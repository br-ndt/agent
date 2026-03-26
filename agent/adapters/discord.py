"""Discord adapter — discord.py async client.

The bot responds to messages in channels where it's mentioned (@botname)
or in DMs. This avoids spamming every channel it's in.
"""

import asyncio

import discord
import structlog

from .base import BaseAdapter, IncomingMessage
from agent.providers import BaseProvider
from agent.relevance import RelevanceFilter

log = structlog.get_logger()


class DiscordAdapter(BaseAdapter):
    def __init__(self, token: str, allowed_ids: set[str] | None = None,
                 other_bot_ids: list[str] | None = None,
                 relevance_provider: BaseProvider | None = None,
                 relevance_model: str = "gemini-2.5-flash",
                 other_bots: dict[str, str] | None = None):
        self.token = token
        self.allowed_ids = allowed_ids
        self.relevance_provider = relevance_provider
        self.relevance_model = relevance_model
        self.other_bots = other_bots or {}
        self._relevance_filter = None

        intents = discord.Intents.default()
        intents.message_content = True  # Required to read message text
        intents.members = True

        self.client = discord.Client(intents=intents)
        self._on_message = None
        self._ready = asyncio.Event()

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
            log.info("discord_connected",
                     user=str(self.client.user),
                     guilds=len(self.client.guilds))
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
                # Extract raw IDs for a safe mathematical intersection
                bot_role_ids = {role.id for role in message.guild.me.roles}
                pinged_role_ids = {role.id for role in message.role_mentions}
                
                log.debug("role_check", 
                          bot_has_roles=list(bot_role_ids), 
                          message_pinged_roles=list(pinged_role_ids))

                if bot_role_ids.intersection(pinged_role_ids):
                    is_role_mentioned = True
                    log.info("bot_role_pinged") 

            # Relevance filter — replaces the simple bail-out check
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
                    is_mentioned=is_mentioned or is_role_mentioned,
                )

                if not relevant:
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

            if self.allowed_ids and composite_id not in self.allowed_ids:
                log.debug("discord_unknown_user_dropped", user_id=user_id)
                return

            system_context = ""
            if message.guild and hasattr(message.channel, 'members'):
                # Get non-bot members who can see this channel
                channel_members = [
                    f"{m.display_name} (<@{m.id}>)" 
                    for m in message.channel.members 
                    if not m.bot
                ]
                
                # Cap the list if you're in a massive server to save LLM context window tokens
                if len(channel_members) > 50:
                    channel_members = channel_members[:50]
                
                if channel_members:
                    system_context = "\n\n[System Context: To ping a user, use their exact ID format. Users in this channel: " + ", ".join(channel_members) + "]"

            # Append the system context to the text sent to the agent
            final_text = text + system_context

            msg = IncomingMessage(
                platform="discord",
                sender_id=user_id,
                sender_name=message.author.display_name,
                text=final_text,
                chat_id=str(message.channel.id),
            )

            log.info("discord_message_received",
                     sender=msg.sender_name,
                     channel=str(message.channel),
                     is_dm=is_dm,
                     text_len=len(text))

            try:
                # Show typing indicator while processing
                async with message.channel.typing():
                    await self._on_message(msg)
            except Exception as e:
                log.error("discord_on_message_error", error=str(e), exc_info=True)

        # Start the client (this blocks, so we run it as a task)
        await self.client.start(self.token)

    async def send(self, chat_id: str, text: str):
        log.debug("discord_send_attempt", chat_id=chat_id, text_len=len(text))
        try:
            channel = self.client.get_channel(int(chat_id))
            if not channel:
                channel = await self.client.fetch_channel(int(chat_id))
            for i in range(0, len(text), 1900):
                chunk = text[i:i + 1900]
                await channel.send(chunk)
            log.info("discord_message_sent", chat_id=chat_id)
        except Exception as e:
            log.error("discord_send_error", chat_id=chat_id, error=str(e), exc_info=True)

    async def stop(self):
        log.info("discord_adapter_stopping")
        await self.client.close()