"""Google (Gemini) provider."""

import structlog

from google import genai
from google.genai import types
from .base import BaseProvider, LLMResponse

log = structlog.get_logger()


class GoogleProvider(BaseProvider):
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    async def complete(
        self,
        messages: list[dict],
        system: str = "",
        model: str = "gemini-2.5-flash",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        # Convert from OpenAI-style messages to Gemini format
        contents = []
        for msg in messages:
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append(types.Content(
                role=role,
                parts=[types.Part(text=msg["content"])],
            ))

        config = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        if system:
            config.system_instruction = system

        response = await self.client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

        text = response.text or ""
        log.info(
            "google_response",
            model=model,
            content_len=len(text),
        )
        usage = {}
        if response.usage_metadata:
            usage = {
                "input_tokens": response.usage_metadata.prompt_token_count or 0,
                "output_tokens": response.usage_metadata.candidates_token_count or 0,
            }

        return LLMResponse(content=text, model=model, usage=usage)
