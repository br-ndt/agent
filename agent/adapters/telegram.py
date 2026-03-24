"""Telegram adapter — python-telegram-bot async."""

import structlog
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes

from .base import BaseAdapter, IncomingMessage

log = structlog.get_logger()


class TelegramAdapter(BaseAdapter):
    def __init__(self, token: str, allowed_ids: set[str] | None = None):
        self.token = token
        self.allowed_ids = allowed_ids  # Set of "telegram:<id>" strings
        self.app = None
        self._on_message = None

    async def start(self, on_message):
        self._on_message = on_message

        self.app = ApplicationBuilder().token(self.token).build()
        self.app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message)
        )

        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling(drop_pending_updates=True)

        log.info("telegram_adapter_started")

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not update.message or not update.message.text:
            return

        user_id = str(update.effective_user.id)
        composite_id = f"telegram:{user_id}"

        # Access control: silently drop unknown users
        if self.allowed_ids and composite_id not in self.allowed_ids:
            log.debug("telegram_unknown_user_dropped", user_id=user_id)
            return

        msg = IncomingMessage(
            platform="telegram",
            sender_id=user_id,
            sender_name=update.effective_user.first_name or "Unknown",
            text=update.message.text,
            chat_id=str(update.effective_chat.id),
        )

        log.info("telegram_message_received",
                 sender=msg.sender_name, text_len=len(msg.text))

        await self._on_message(msg)

    async def send(self, chat_id: str, text: str):
        if not self.app:
            return
        # Telegram max message length is 4096
        for i in range(0, len(text), 4000):
            chunk = text[i:i + 4000]
            await self.app.bot.send_message(chat_id=int(chat_id), text=chunk)

    async def stop(self):
        if self.app:
            log.info("telegram_adapter_stopping")
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()
