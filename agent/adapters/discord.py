"""Discord adapter — discord.py async client.

The bot responds to messages in channels where it's mentioned (@botname)
or in DMs. This avoids spamming every channel it's in.
"""

import asyncio

import discord
import structlog

from .base import BaseAdapter, IncomingMessage

log = structlog.get_logger()


class DiscordAdapter(BaseAdapter):
    def __init__(self, token: str, allowed_ids: set[str] | None = None):
        self.token = token
        self.allowed_ids = allowed_ids

        intents = discord.Intents.default()
        intents.message_content = True  # Required to read message text

        self.client = discord.Client(intents=intents)
        self._on_message = None
        self._ready = asyncio.Event()

    async def start(self, on_message):
        self._on_message = on_message

        @self.client.event
        async def on_ready():
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

            if not is_dm and not is_mentioned:
                return

            # Strip the bot mention from the message text
            text = message.content
            if self.client.user:
                text = text.replace(f"<@{self.client.user.id}>", "").strip()

            if not text:
                return

            user_id = str(message.author.id)
            composite_id = f"discord:{user_id}"

            # Access control
            if self.allowed_ids and composite_id not in self.allowed_ids:
                log.debug("discord_unknown_user_dropped", user_id=user_id)
                return

            msg = IncomingMessage(
                platform="discord",
                sender_id=user_id,
                sender_name=message.author.display_name,
                text=text,
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