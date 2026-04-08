"""Tests for agent.subagent_runner — task execution and tool dispatch."""

import pytest

from agent.config import SubagentConfig
from agent.providers.base import BaseProvider, LLMResponse
from agent.subagent_runner import SubagentRunner


class MockProvider(BaseProvider):
    """Provider that returns canned responses."""

    def __init__(self, response: str = "done", fail: bool = False):
        self.response = response
        self.fail = fail
        self.calls: list[dict] = []
        self.has_native_tools = False

    async def complete(self, **kwargs) -> LLMResponse:
        self.calls.append(kwargs)
        if self.fail:
            raise ConnectionError("mock failure")
        return LLMResponse(content=self.response, model="mock", usage={})


def _make_runner(
    tools: list[str] | None = None,
    provider: MockProvider | None = None,
    fallback: MockProvider | None = None,
    capabilities: list[str] | None = None,
) -> SubagentRunner:
    config = SubagentConfig(
        name="test-agent",
        model="mock-model",
        personality="You are a test agent.",
        tools=tools or [],
        capabilities=capabilities or [],
    )
    return SubagentRunner(
        config=config,
        provider=provider or MockProvider(),
        fallback_provider=fallback,
        fallback_model="fallback-model",
    )


class TestTextOnlyExecution:
    """Subagents without tools use _run_text_only."""

    @pytest.mark.asyncio
    async def test_basic_text_only(self):
        provider = MockProvider(response="hello world")
        runner = _make_runner(provider=provider)
        result = await runner.run("Say hello")
        assert result == "hello world"
        assert len(provider.calls) == 1

    @pytest.mark.asyncio
    async def test_context_prepended_to_prompt(self):
        provider = MockProvider()
        runner = _make_runner(provider=provider)
        await runner.run("Do task", context="Background info")
        prompt = provider.calls[0]["messages"][0]["content"]
        assert "Background info" in prompt
        assert "Do task" in prompt

    @pytest.mark.asyncio
    async def test_personality_passed_as_system(self):
        provider = MockProvider()
        runner = _make_runner(provider=provider)
        await runner.run("test")
        assert provider.calls[0]["system"] == "You are a test agent."

    @pytest.mark.asyncio
    async def test_audio_attachments_passed_to_provider(self):
        provider = MockProvider()
        runner = _make_runner(provider=provider)
        audio = [{"data": b"mp3data", "mime_type": "audio/mpeg", "filename": "t.mp3"}]
        await runner.run("Analyze this", attachments=audio)
        msg = provider.calls[0]["messages"][0]
        assert "audio" in msg
        assert msg["audio"][0]["data"] == b"mp3data"

    @pytest.mark.asyncio
    async def test_image_attachments_passed_to_provider(self):
        provider = MockProvider()
        runner = _make_runner(provider=provider)
        images = [{"data": b"pngdata", "mime_type": "image/png", "filename": "i.png"}]
        await runner.run("Look at this", attachments=images)
        msg = provider.calls[0]["messages"][0]
        assert "images" in msg
        assert msg["images"][0]["data"] == b"pngdata"

    @pytest.mark.asyncio
    async def test_attachments_cleared_after_run(self):
        runner = _make_runner()
        audio = [{"data": b"mp3", "mime_type": "audio/mpeg", "filename": "t.mp3"}]
        await runner.run("test", attachments=audio)
        assert runner._pending_attachments is None

    @pytest.mark.asyncio
    async def test_no_attachments_no_media_keys(self):
        provider = MockProvider()
        runner = _make_runner(provider=provider)
        await runner.run("test")
        msg = provider.calls[0]["messages"][0]
        assert "audio" not in msg
        assert "images" not in msg


class TestFallbackBehavior:
    """Primary failure triggers fallback provider."""

    @pytest.mark.asyncio
    async def test_fallback_on_primary_failure(self):
        primary = MockProvider(fail=True)
        fallback = MockProvider(response="fallback worked")
        runner = _make_runner(provider=primary, fallback=fallback)
        result = await runner.run("test")
        assert result == "fallback worked"
        assert len(primary.calls) == 1
        assert len(fallback.calls) == 1

    @pytest.mark.asyncio
    async def test_no_fallback_raises(self):
        primary = MockProvider(fail=True)
        runner = _make_runner(provider=primary)
        with pytest.raises(ConnectionError):
            await runner.run("test")

    @pytest.mark.asyncio
    async def test_fallback_uses_fallback_model(self):
        primary = MockProvider(fail=True)
        fallback = MockProvider()
        runner = _make_runner(provider=primary, fallback=fallback)
        await runner.run("test")
        assert fallback.calls[0]["model"] == "fallback-model"

    @pytest.mark.asyncio
    async def test_attachments_cleared_even_on_failure(self):
        primary = MockProvider(fail=True)
        runner = _make_runner(provider=primary)
        audio = [{"data": b"mp3", "mime_type": "audio/mpeg", "filename": "t.mp3"}]
        with pytest.raises(ConnectionError):
            await runner.run("test", attachments=audio)
        assert runner._pending_attachments is None


class TestToolDispatch:
    """Correct execution path is chosen based on tools and provider."""

    @pytest.mark.asyncio
    async def test_no_tools_uses_text_only(self):
        provider = MockProvider(response="text only result")
        runner = _make_runner(tools=[], provider=provider)
        result = await runner.run("test")
        assert result == "text only result"

    @pytest.mark.asyncio
    async def test_prompt_format_task_only(self):
        provider = MockProvider()
        runner = _make_runner(provider=provider)
        await runner.run("Just the task")
        content = provider.calls[0]["messages"][0]["content"]
        assert content == "Just the task"

    @pytest.mark.asyncio
    async def test_prompt_format_with_context(self):
        provider = MockProvider()
        runner = _make_runner(provider=provider)
        await runner.run("The task", context="The context")
        content = provider.calls[0]["messages"][0]["content"]
        assert content == "Context:\nThe context\n\nTask:\nThe task"
