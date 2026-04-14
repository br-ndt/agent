"""Discord adapter — discord.py async client.

The bot responds to messages in channels where it's mentioned (@botname)
or in DMs. This avoids spamming every channel it's in.
"""

import asyncio
from io import BytesIO

import discord
import structlog

from .base import BaseAdapter, IncomingMessage
from agent.providers import BaseProvider
from agent.relevance import RelevanceFilter

log = structlog.get_logger()

MAX_LEN = 2000

# File extensions we treat as text (inlined into the message)
_TEXT_EXTENSIONS = {
    ".txt", ".md", ".py", ".js", ".ts", ".jsx", ".tsx", ".json", ".yaml",
    ".yml", ".toml", ".cfg", ".ini", ".conf", ".sh", ".bash", ".zsh",
    ".html", ".css", ".scss", ".less", ".xml", ".svg", ".csv", ".sql",
    ".rs", ".go", ".java", ".kt", ".c", ".cpp", ".h", ".hpp", ".cs",
    ".rb", ".php", ".lua", ".r", ".swift", ".scala", ".ex", ".exs",
    ".hs", ".ml", ".clj", ".erl", ".elm", ".vue", ".svelte", ".astro",
    ".env", ".gitignore", ".dockerignore", ".dockerfile", ".makefile",
    ".log", ".diff", ".patch", ".rst", ".tex", ".org",
}


def _is_text_attachment(content_type: str, filename: str) -> bool:
    """Check if an attachment is a text file we should inline."""
    if content_type.startswith("text/"):
        return True
    # application/json, application/xml, etc.
    if content_type in (
        "application/json", "application/xml", "application/javascript",
        "application/x-yaml", "application/toml", "application/sql",
        "application/x-sh", "application/x-python",
    ):
        return True
    # Fall back to extension check
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in _TEXT_EXTENSIONS


def _split_message(text: str, limit: int = MAX_LEN) -> list[str]:
    """Split text into chunks that respect Discord's character limit.

    Tries to break at, in order: paragraph boundaries (blank lines),
    line boundaries, then word boundaries. Preserves open code fences
    across chunks so markdown rendering isn't broken.
    """
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    remaining = text

    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break

        # Find the best split point within the limit
        window = remaining[:limit]
        split_at = -1

        # Prefer blank-line break (paragraph boundary)
        pos = window.rfind("\n\n")
        if pos > limit // 4:
            split_at = pos + 1  # keep one newline on current chunk

        # Fall back to any newline
        if split_at == -1:
            pos = window.rfind("\n")
            if pos > limit // 4:
                split_at = pos + 1

        # Fall back to a space
        if split_at == -1:
            pos = window.rfind(" ")
            if pos > limit // 4:
                split_at = pos + 1

        # Last resort: hard cut
        if split_at == -1:
            split_at = limit

        chunk = remaining[:split_at]
        remaining = remaining[split_at:]

        # If we split inside an unclosed code fence, close it on this
        # chunk and re-open it on the next so both render correctly.
        fence_count = chunk.count("```")
        if fence_count % 2 == 1:
            chunk += "\n```"
            remaining = "```\n" + remaining

        chunks.append(chunk)

    return chunks


class DiscordAdapter(BaseAdapter):
    def __init__(
        self,
        token: str,
        allowed_ids: set[str] | None = None,
        other_bot_ids: list[str] | None = None,
        relevance_provider: BaseProvider | None = None,
        relevance_model: str = "gemini-2.5-flash",
        other_bots: dict[str, str] | None = None,
        sibling_ids: set[str] | None = None,
    ):
        self.token = token
        self.allowed_ids = allowed_ids
        self.relevance_provider = relevance_provider
        self.relevance_model = relevance_model
        self.other_bots = other_bots or {}
        # Bare Discord IDs (no "discord:" prefix) of sibling instances — other
        # bots running this same agent. Used to inject collaboration guidance
        # only when siblings share a channel.
        self.sibling_ids = {s.split(":", 1)[1] if ":" in s else s for s in (sibling_ids or set())}
        self._relevance_filter = None

        intents = discord.Intents.default()
        intents.message_content = True  # Required to read message text
        intents.members = True

        self.client = discord.Client(intents=intents)
        self._on_message = None
        self._ready = asyncio.Event()

        self._admin_ids: set[str] = set()  # populated from router

    async def start(self, on_message):
        self._on_message = on_message

        @self.client.event
        async def on_ready():
            if self.relevance_provider:
                self._relevance_filter = RelevanceFilter(
                    bot_name=self.client.user.display_name,
                    bot_id=str(self.client.user.id),
                    provider=self.relevance_provider,
                    model=self.relevance_model,
                    other_bot_names=self.other_bots,
                )
                log.info("relevance_filter_enabled")
            log.info(
                "discord_connected",
                user=str(self.client.user),
                guilds=len(self.client.guilds),
            )
            self._ready.set()

        @self.client.event
        async def on_message(message: discord.Message):
            # Ignore our own messages
            if message.author == self.client.user:
                return

            # Respond to DMs always, or channel messages where we're mentioned
            is_dm = isinstance(message.channel, discord.DMChannel)
            bot_id = str(self.client.user.id)
            is_mentioned = (
                self.client.user in message.mentions
                or f"<@{bot_id}>" in message.content
                or f"<@!{bot_id}>" in message.content
            )

            # Check if any role the bot has was mentioned
            is_role_mentioned = False
            bot_role_ids = set()

            if message.guild and message.guild.me:
                bot_role_ids = {role.id for role in message.guild.me.roles}
                pinged_role_ids = {role.id for role in message.role_mentions}

                log.debug(
                    "role_check",
                    bot_has_roles=list(bot_role_ids),
                    message_pinged_roles=list(pinged_role_ids),
                )

                if bot_role_ids.intersection(pinged_role_ids):
                    is_role_mentioned = True
                    log.info("bot_role_pinged")

            log.info(
                "discord_message_pre_filter",
                sender=message.author.display_name,
                sender_id=str(message.author.id),
                is_bot=message.author.bot,
                is_dm=is_dm,
                is_mentioned=is_mentioned,
                is_role_mentioned=is_role_mentioned,
                text_preview=message.content[:100],
            )

            # Relevance filter
            if self._relevance_filter:
                relevant, action = await self._relevance_filter.is_relevant(
                    IncomingMessage(
                        platform="discord",
                        sender_id=str(message.author.id),
                        sender_name=message.author.display_name,
                        text=message.content,
                        chat_id=str(message.channel.id),
                    ),
                    is_dm=is_dm,
                    is_mentioned=is_mentioned,
                    is_role_mentioned=is_role_mentioned,
                )

                if not relevant:
                    return

                if action == "react":
                    emoji = await self._pick_reaction(message.content, guild=message.guild)
                    try:
                        await message.add_reaction(emoji)
                        log.info("discord_reacted", emoji=emoji, sender=message.author.display_name)
                    except discord.Forbidden:
                        log.error("discord_react_forbidden",
                                  hint="Bot lacks 'Add Reactions' permission in this channel")
                    except discord.HTTPException as e:
                        log.error("discord_react_failed", error=str(e), emoji=emoji)
                    return

                if action == "wait":
                    await message.channel.send("Got it, standing by.")
                    return
            else:
                # Fallback if no relevance filter configured
                if not is_dm and not is_mentioned and not is_role_mentioned:
                    return

            # Strip the direct bot mentions
            text = message.content
            if self.client.user:
                text = text.replace(f"<@{bot_id}>", "")
                text = text.replace(f"<@!{bot_id}>", "")

            # Strip the role mentions so the LLM doesn't read the <@&ID> tags
            if message.guild and message.role_mentions:
                for role in message.role_mentions:
                    if role.id in bot_role_ids:
                        text = text.replace(f"<@&{role.id}>", "")

            text = text.strip()

            if not text and (is_role_mentioned or is_mentioned):
                text = "[System: You were pinged by the user.]"
            elif not text:
                return

            user_id = str(message.author.id)
            composite_id = f"discord:{user_id}"

            # Allow known bots from other_bots config to pass access control
            is_known_bot = False
            if self.other_bots:
                known_bot_ids = set()
                if isinstance(self.other_bots, dict):
                    known_bot_ids = {k.split(":", 1)[-1] for k in self.other_bots}
                elif isinstance(self.other_bots, list):
                    known_bot_ids = {str(b).split(":", 1)[-1] for b in self.other_bots}
                is_known_bot = user_id in known_bot_ids

            if self.allowed_ids and composite_id not in self.allowed_ids and not is_known_bot:
                log.debug("discord_unknown_user_dropped", user_id=user_id)
                return

            # ── Fetch recent channel history so the bot sees what was
            # said before it was mentioned (other bots, other users) ──
            channel_context = ""
            if not is_dm:
                try:
                    recent: list[discord.Message] = []
                    async for hist_msg in message.channel.history(
                        limit=15, before=message
                    ):
                        recent.append(hist_msg)
                    recent.reverse()  # oldest first

                    if recent:
                        lines = []
                        for m in recent:
                            tag = "[BOT]" if m.author.bot else "[USER]"
                            content = m.content[:500] if m.content else ""
                            # Note any image attachments so the bot knows
                            # images were posted, even if it can't see them
                            # inline in the history context.
                            img_atts = [
                                a for a in m.attachments
                                if (a.content_type or "").startswith("image/")
                            ]
                            if img_atts:
                                img_note = f" [attached {len(img_atts)} image(s)]"
                                content = (content or "") + img_note
                            if content:
                                lines.append(f"{tag} {m.author.display_name}: {content}")
                        if lines:
                            channel_context = (
                                "[Recent channel messages (oldest first):\n"
                                + "\n".join(lines)
                                + "\n]\n\n"
                            )
                except Exception as e:
                    log.warning("discord_history_fetch_failed", error=str(e))

            system_context = ""
            if message.guild and hasattr(message.channel, "members"):
                channel_members = []
                siblings_present: list[str] = []
                for m in message.channel.members:
                    if m.id == self.client.user.id:
                        continue
                    member_type = "[BOT]" if m.bot else "[USER]"
                    channel_members.append(f"{member_type} {m.display_name} (<@{m.id}>)")
                    if str(m.id) in self.sibling_ids:
                        siblings_present.append(f"{m.display_name} (<@{m.id}>)")

                if len(channel_members) > 50:
                    channel_members = channel_members[:50]

                if channel_members:
                    system_context = (
                        "\n\n[System Context: To ping a user, use their exact ID format. Users in this channel: "
                        + ", ".join(channel_members)
                        + "]"
                    )

                if siblings_present:
                    system_context += (
                        "\n\n[Sibling Instances Present: "
                        + ", ".join(siblings_present)
                        + ". These are other instances of you running the same agent. "
                        "You can @-ping them to collaborate on ambitious multi-step projects — "
                        "e.g. divide research, parallelize coding, or have one plan while another implements.\n"
                        "Coordination pattern (git worktrees):\n"
                        "- One instance initiates: creates the repo and makes the initial commit on `main`.\n"
                        "- Siblings sync work back via worktrees — `git worktree add ../<task>-<name> -b <name>/<task>` "
                        "off the shared repo, work in isolation, then merge back to `main` when done.\n"
                        "- Pull/rebase before merging; reconcile rather than overwrite if two of you touched the same file.\n"
                        "Coordination rules:\n"
                        "- Before starting, declare scope in-channel (\"I'll take the frontend, you take the API schema\").\n"
                        "- Announce decisions that affect shared state (schema changes, API contracts, config keys).\n"
                        "- If unsure who should do something, ask — better than duplicating work or conflicting edits.]"
                    )

            final_text = channel_context + text + system_context

            # Extract attachments: images/audio as binary, text files inlined
            attachments = []
            text_attachments = []

            # If the user replied to a message, pull images from the
            # referenced message so the bot can see what they're referring to.
            if message.reference and message.reference.message_id:
                try:
                    ref_msg = message.reference.resolved
                    if ref_msg is None:
                        ref_msg = await message.channel.fetch_message(
                            message.reference.message_id
                        )
                    if ref_msg and ref_msg.attachments:
                        for att in ref_msg.attachments:
                            ct = att.content_type or ""
                            if ct.startswith("image/"):
                                try:
                                    file_bytes = await att.read()
                                    attachments.append(
                                        {
                                            "data": file_bytes,
                                            "mime_type": ct,
                                            "filename": att.filename or "",
                                        }
                                    )
                                    log.info(
                                        "discord_reply_image_extracted",
                                        filename=att.filename,
                                        size=len(file_bytes),
                                    )
                                except Exception as e:
                                    log.warning(
                                        "discord_reply_attachment_failed",
                                        error=str(e),
                                    )
                except Exception as e:
                    log.warning("discord_reference_fetch_failed", error=str(e))

            for attachment in message.attachments:
                ct = attachment.content_type or ""
                fname = attachment.filename or ""

                if ct.startswith("image/") or ct.startswith("audio/"):
                    try:
                        file_bytes = await attachment.read()
                        attachments.append(
                            {
                                "data": file_bytes,
                                "mime_type": ct,
                                "filename": fname,
                            }
                        )
                        kind = "image" if ct.startswith("image/") else "audio"
                        log.info(
                            f"discord_{kind}_extracted",
                            filename=fname,
                            size=len(file_bytes),
                        )
                    except Exception as e:
                        log.error(
                            "discord_attachment_download_failed",
                            filename=fname,
                            error=str(e),
                        )
                elif _is_text_attachment(ct, fname):
                    try:
                        file_bytes = await attachment.read()
                        text_content = file_bytes.decode("utf-8", errors="replace")
                        # Cap at 15k chars to avoid blowing up context
                        if len(text_content) > 15000:
                            text_content = text_content[:15000] + "\n\n... (truncated)"
                        text_attachments.append((fname, text_content))
                        log.info(
                            "discord_text_extracted",
                            filename=fname,
                            size=len(file_bytes),
                        )
                    except Exception as e:
                        log.error(
                            "discord_text_download_failed",
                            filename=fname,
                            error=str(e),
                        )

            # Inline text attachments into the message
            if text_attachments:
                parts = []
                for fname, content in text_attachments:
                    parts.append(f"[Attached file: {fname}]\n```\n{content}\n```")
                final_text = "\n\n".join(parts) + "\n\n" + final_text

            msg = IncomingMessage(
                platform="discord",
                sender_id=user_id,
                sender_name=message.author.display_name,
                text=final_text,
                chat_id=str(message.channel.id),
                attachments=attachments,
            )

            log.info(
                "discord_message_received",
                sender=msg.sender_name,
                channel=str(message.channel),
                is_dm=is_dm,
                text_len=len(text),
                has_attachments=len(attachments) > 0,
            )

            try:
                async with message.channel.typing():
                    await self._on_message(msg)
            except Exception as e:
                log.error("discord_on_message_error", error=str(e), exc_info=True)

        await self.client.start(self.token)

    async def _pick_reaction(self, text: str, guild: discord.Guild | None = None) -> str | discord.Emoji:
        """Use a cheap LLM call to pick an appropriate emoji reaction.

        Prefers server custom emoji when available.
        """
        if not self.relevance_provider:
            return "\U0001f44d"  # thumbs up fallback

        # Build custom emoji menu if we have a guild
        custom_emoji_map: dict[str, discord.Emoji] = {}
        emoji_menu = ""
        if guild and guild.emojis:
            for e in guild.emojis:
                if e.available:
                    custom_emoji_map[e.name.lower()] = e
            if custom_emoji_map:
                names = ", ".join(f":{name}:" for name in custom_emoji_map)
                emoji_menu = (
                    f"\n\nPrefer these server emoji when they fit: {names}\n"
                    "To pick one, reply with its exact name including colons, e.g. :poggers:"
                )

        try:
            response = await self.relevance_provider.complete(
                messages=[{"role": "user", "content": text[:300]}],
                system=(
                    "Pick ONE emoji that best reacts to this message. "
                    "Reply with ONLY the emoji or emoji name, nothing else."
                    f"{emoji_menu}"
                ),
                model=self.relevance_model,
                max_tokens=15,
                temperature=0.7,
            )
            pick = response.content.strip().strip(":")
            # Check if it matches a custom emoji
            if pick.lower() in custom_emoji_map:
                return custom_emoji_map[pick.lower()]
            # Fall back to treating it as a unicode emoji
            raw = response.content.strip()
            # Reject strings that are just colons/whitespace — Discord
            # tries to parse ":" as a custom emoji with an empty snowflake ID.
            if raw and len(raw) <= 8 and raw.strip(":"):
                return raw
        except Exception as e:
            log.debug("react_emoji_pick_failed", error=str(e))

        return "\U0001f44d"

    async def send(self, chat_id: str, text: str):
        log.debug("discord_send_attempt", chat_id=chat_id, text_len=len(text))
        try:
            channel = self.client.get_channel(int(chat_id))
            if not channel:
                channel = await self.client.fetch_channel(int(chat_id))
            for chunk in _split_message(text):
                await channel.send(chunk)
            log.info("discord_message_sent", chat_id=chat_id)
        except Exception as e:
            log.error(
                "discord_send_error", chat_id=chat_id, error=str(e), exc_info=True
            )

    async def send_images(self, chat_id: str, images: list[bytes]):
        """Send image bytes to a Discord channel."""
        log.debug("discord_send_images_attempt", chat_id=chat_id, count=len(images))
        try:
            channel = self.client.get_channel(int(chat_id))
            if not channel:
                channel = await self.client.fetch_channel(int(chat_id))

            for idx, image_bytes in enumerate(images):
                file = discord.File(
                    fp=BytesIO(image_bytes), filename=f"generated_{idx}.png"
                )
                await channel.send(file=file)
                log.info(
                    "discord_image_sent",
                    chat_id=chat_id,
                    index=idx,
                    size=len(image_bytes),
                )
        except Exception as e:
            log.error(
                "discord_send_images_error",
                chat_id=chat_id,
                error=str(e),
                exc_info=True,
            )

    async def stop(self):
        log.info("discord_adapter_stopping")
        await self.client.close()
