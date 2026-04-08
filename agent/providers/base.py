"""Abstract base for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class LLMResponse:
    content: str
    model: str
    usage: dict = field(default_factory=dict)
    tool_calls: list = field(default_factory=list)
    images: list[bytes] = field(default_factory=list)


class BaseProvider(ABC):
    @abstractmethod
    async def complete(
        self,
        messages: list[dict],
        system: str = "",
        model: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        tools: list[dict] | None = None,
        cwd: str | None = None,
    ) -> LLMResponse: ...
