"""Google (Gemini) provider — supports text, image, and audio inputs."""

import asyncio
import tempfile
import structlog
from pathlib import Path

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
        cwd: str | None = None,
        **kwargs,
    ) -> LLMResponse:
        # Convert from OpenAI-style messages to Gemini format
        # Messages can include image paths via {"role": "user", "content": ..., "images": [...]}
        contents = []
        for msg in messages:
            role = "model" if msg["role"] == "assistant" else "user"
            parts = [types.Part(text=msg["content"])]

            # Attach images if present
            for img in msg.get("images", []):
                img_path = Path(img) if isinstance(img, str) else None
                if img_path and img_path.exists():
                    mime = "image/png" if img_path.suffix == ".png" else "image/jpeg"
                    parts.append(types.Part.from_bytes(
                        data=img_path.read_bytes(),
                        mime_type=mime,
                    ))
                elif isinstance(img, bytes):
                    parts.append(types.Part.from_bytes(
                        data=img,
                        mime_type="image/png",
                    ))

            # Attach audio if present — use File API for reliability
            for aud in msg.get("audio", []):
                data = aud["data"] if isinstance(aud, dict) else aud
                mime = aud.get("mime_type", "audio/mpeg") if isinstance(aud, dict) else "audio/mpeg"
                filename = aud.get("filename", "audio.mp3") if isinstance(aud, dict) else "audio.mp3"
                try:
                    file_part = await self._upload_audio(data, mime, filename)
                    parts.append(file_part)
                    log.info("audio_attached", filename=filename, size=len(data), mime=mime)
                except Exception as exc:
                    log.error("audio_upload_failed", error=str(exc), filename=filename)
                    # Fall back to inline bytes
                    parts.append(types.Part.from_bytes(data=data, mime_type=mime))

            contents.append(types.Content(role=role, parts=parts))

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

        # Extract text and any generated images from response parts.
        # response.text is empty when the model returns image output, so we
        # iterate candidates[0].content.parts directly.
        import base64 as _b64
        text = ""
        image_bytes_list: list[bytes] = []
        try:
            parts = response.candidates[0].content.parts
            for part in parts:
                if hasattr(part, "inline_data") and part.inline_data:
                    data = part.inline_data.data
                    if isinstance(data, str):
                        image_bytes_list.append(_b64.b64decode(data))
                    else:
                        image_bytes_list.append(bytes(data))
                elif hasattr(part, "text") and part.text:
                    text += part.text
        except (AttributeError, IndexError):
            # Fall back to .text for non-generative models
            text = response.text or ""

        log.info(
            "google_response",
            model=model,
            content_len=len(text),
            image_count=len(image_bytes_list),
        )
        usage = {}
        if response.usage_metadata:
            usage = {
                "input_tokens": response.usage_metadata.prompt_token_count or 0,
                "output_tokens": response.usage_metadata.candidates_token_count or 0,
            }

        return LLMResponse(content=text, model=model, usage=usage, images=image_bytes_list)

    async def _upload_audio(self, data: bytes, mime_type: str, filename: str):
        """Upload audio via the Gemini File API and return a Part reference.

        The File API handles large files and is more reliable for audio than
        inline Part.from_bytes.
        """
        # Map common mime types to file extensions
        ext_map = {
            "audio/mpeg": ".mp3", "audio/mp3": ".mp3",
            "audio/wav": ".wav", "audio/x-wav": ".wav",
            "audio/ogg": ".ogg", "audio/flac": ".flac",
            "audio/aac": ".aac", "audio/mp4": ".m4a",
        }
        ext = ext_map.get(mime_type, ".mp3")

        # Write to temp file (Gemini SDK needs a file path for upload)
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        try:
            loop = asyncio.get_running_loop()
            uploaded = await loop.run_in_executor(
                None,
                lambda: self.client.files.upload(
                    file=tmp_path,
                    config=types.UploadFileConfig(mime_type=mime_type),
                ),
            )
            log.info("audio_file_uploaded", name=uploaded.name, size=len(data))
            return types.Part.from_uri(file_uri=uploaded.uri, mime_type=mime_type)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
