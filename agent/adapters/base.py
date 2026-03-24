"""Abstract base for chat adapters."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Awaitable


@dataclass
class IncomingMessage:
    platform: str              # "telegram", "discord", "cli"
    sender_id: str             # Platform-specific user ID
    sender_name: str           # Display name
    text: str
    chat_id: str = ""          # Where to send the reply
    attachments: list = field(default_factory=list)


class BaseAdapter(ABC):
    @abstractmethod
    async def start(self, on_message: Callable[[IncomingMessage], Awaitable[None]]):
        """Start listening. Call on_message for each incoming message."""
        ...

    @abstractmethod
    async def send(self, chat_id: str, text: str):
        """Send a message back."""
        ...

    @abstractmethod
    async def stop(self):
        """Graceful shutdown."""
        ...
