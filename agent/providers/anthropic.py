"""Anthropic (Claude) provider."""

import anthropic
from .base import BaseProvider, LLMResponse


class AnthropicProvider(BaseProvider):
    def __init__(self, api_key: str):
        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    async def complete(
        self,
        messages: list[dict],
        system: str = "",
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if temperature is not None:
            kwargs["temperature"] = temperature
        # Tool use support can be added here later

        response = await self.client.messages.create(**kwargs)

        # Extract text content (skip tool_use blocks for now)
        text_parts = [
            block.text for block in response.content if hasattr(block, "text")
        ]

        return LLMResponse(
            content="\n".join(text_parts),
            model=response.model,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )
