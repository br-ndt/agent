"""CLI adapter — local readline REPL for testing."""

import asyncio
import sys

from .base import BaseAdapter, IncomingMessage


class CLIAdapter(BaseAdapter):
    def __init__(self):
        self._running = False

    async def start(self, on_message):
        self._running = True
        loop = asyncio.get_event_loop()

        print("─" * 50)
        print("Agent CLI — type a message, or 'quit' to exit")
        print("─" * 50)

        while self._running:
            try:
                # Use run_in_executor so we don't block the event loop
                line = await loop.run_in_executor(None, sys.stdin.readline)
            except EOFError:
                break

            line = line.strip()
            if not line:
                continue
            if line.lower() in ("quit", "exit", "/quit"):
                break

            msg = IncomingMessage(
                platform="cli",
                sender_id="local",
                sender_name="You",
                text=line,
                chat_id="cli",
            )
            await on_message(msg)

        print("\nGoodbye!")

    async def send(self, chat_id: str, text: str):
        print(f"\n{'─' * 50}")
        print(text)
        print(f"{'─' * 50}\n")

    async def stop(self):
        self._running = False
