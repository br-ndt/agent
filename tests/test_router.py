"""Tests for agent.router — access control and message dispatch."""

import asyncio

import pytest
import pytest_asyncio

from agent.adapters.base import IncomingMessage, BaseAdapter
from agent.config import Config
from agent.router import Router


class FakeAdapter(BaseAdapter):
    """Minimal adapter that records sent messages."""

    def __init__(self):
        self.sent: list[tuple[str, str]] = []
        self.sent_images: list[tuple[str, list]] = []

    async def start(self, on_message):
        pass

    async def send(self, chat_id: str, text: str):
        self.sent.append((chat_id, text))

    async def send_images(self, chat_id: str, images: list[bytes]):
        self.sent_images.append((chat_id, images))

    async def stop(self):
        pass


class FakeOrchestrator:
    """Minimal orchestrator that records handle() calls."""

    def __init__(self, response: str = "ok"):
        self.calls: list[dict] = []
        self.response = response
        self.sessions = {}
        self._active_delegations = {}
        self._skill_tasks = {}
        self.skill_registry = type("R", (), {"list_active": lambda self: [], "list_proposed": lambda self: []})()
        self.skill_executor = type("E", (), {"active_runs": {}})()
        self.error_journal = type("J", (), {"recent": lambda self, n: []})()
        self.diagnostic_store = type("D", (), {"summary": lambda self: "ok", "read_report": lambda self, f: None})()
        self.cost_tracker = None

    async def handle(self, **kwargs) -> dict:
        self.calls.append(kwargs)
        return {"text": self.response, "images": []}


def _make_msg(
    text: str = "hello",
    platform: str = "discord",
    sender_id: str = "user1",
    chat_id: str = "ch1",
    attachments: list | None = None,
) -> IncomingMessage:
    return IncomingMessage(
        platform=platform,
        sender_id=sender_id,
        sender_name="TestUser",
        text=text,
        chat_id=chat_id,
        attachments=attachments or [],
    )


def _make_router(
    admin_ids: set[str] | None = None,
    trusted_ids: set[str] | None = None,
    response: str = "ok",
) -> tuple[Router, FakeOrchestrator, FakeAdapter]:
    config = Config(
        admin_ids=admin_ids or {"discord:admin1"},
        trusted_ids=trusted_ids or {"discord:trusted1"},
    )
    orch = FakeOrchestrator(response=response)
    router = Router(config, orch)
    adapter = FakeAdapter()
    router.register_adapter("discord", adapter)
    router.register_adapter("cli", adapter)
    return router, orch, adapter


class TestGetTier:
    """Access tier resolution."""

    def test_admin_by_id(self):
        router, _, _ = _make_router(admin_ids={"discord:admin1"})
        msg = _make_msg(sender_id="admin1", platform="discord")
        assert router.get_tier(msg) == "admin"

    def test_trusted_by_id(self):
        router, _, _ = _make_router(trusted_ids={"discord:trusted1"})
        msg = _make_msg(sender_id="trusted1", platform="discord")
        assert router.get_tier(msg) == "trusted"

    def test_unknown_user(self):
        router, _, _ = _make_router()
        msg = _make_msg(sender_id="stranger", platform="discord")
        assert router.get_tier(msg) == "unknown"

    def test_cli_always_admin(self):
        router, _, _ = _make_router(admin_ids=set())
        msg = _make_msg(sender_id="whoever", platform="cli")
        assert router.get_tier(msg) == "admin"

    def test_platform_prefix_matters(self):
        router, _, _ = _make_router(admin_ids={"telegram:admin1"})
        # Same ID but wrong platform
        msg = _make_msg(sender_id="admin1", platform="discord")
        assert router.get_tier(msg) == "unknown"


class TestMessageRouting:
    """Message dispatch to orchestrator."""

    @pytest.mark.asyncio
    async def test_admin_message_reaches_orchestrator(self):
        router, orch, adapter = _make_router()
        msg = _make_msg(text="hello", sender_id="admin1")
        await router.handle_message(msg)
        assert len(orch.calls) == 1
        assert "hello" in orch.calls[0]["user_msg"]

    @pytest.mark.asyncio
    async def test_unknown_user_dropped(self):
        router, orch, adapter = _make_router()
        msg = _make_msg(text="hello", sender_id="stranger")
        await router.handle_message(msg)
        assert len(orch.calls) == 0
        assert len(adapter.sent) == 0

    @pytest.mark.asyncio
    async def test_orchestrator_response_sent_to_adapter(self):
        router, orch, adapter = _make_router(response="test response")
        msg = _make_msg(sender_id="admin1")
        await router.handle_message(msg)
        assert any("test response" in text for _, text in adapter.sent)

    @pytest.mark.asyncio
    async def test_images_extracted_from_attachments(self):
        router, orch, adapter = _make_router()
        msg = _make_msg(
            sender_id="admin1",
            attachments=[
                {"data": b"png", "mime_type": "image/png", "filename": "test.png"},
            ],
        )
        await router.handle_message(msg)
        assert orch.calls[0]["images"] is not None
        assert len(orch.calls[0]["images"]) == 1

    @pytest.mark.asyncio
    async def test_audio_extracted_from_attachments(self):
        router, orch, adapter = _make_router()
        msg = _make_msg(
            sender_id="admin1",
            attachments=[
                {"data": b"mp3", "mime_type": "audio/mpeg", "filename": "track.mp3"},
            ],
        )
        await router.handle_message(msg)
        assert orch.calls[0]["audio"] is not None
        assert len(orch.calls[0]["audio"]) == 1

    @pytest.mark.asyncio
    async def test_mixed_attachments_separated(self):
        router, orch, adapter = _make_router()
        msg = _make_msg(
            sender_id="admin1",
            attachments=[
                {"data": b"png", "mime_type": "image/png", "filename": "img.png"},
                {"data": b"mp3", "mime_type": "audio/mpeg", "filename": "song.mp3"},
            ],
        )
        await router.handle_message(msg)
        assert len(orch.calls[0]["images"]) == 1
        assert len(orch.calls[0]["audio"]) == 1

    @pytest.mark.asyncio
    async def test_no_attachments_passes_none(self):
        router, orch, adapter = _make_router()
        msg = _make_msg(sender_id="admin1")
        await router.handle_message(msg)
        assert orch.calls[0]["images"] is None
        assert orch.calls[0]["audio"] is None

    @pytest.mark.asyncio
    async def test_session_id_format(self):
        router, orch, adapter = _make_router()
        msg = _make_msg(sender_id="admin1", platform="discord")
        await router.handle_message(msg)
        assert orch.calls[0]["session_id"] == "discord:admin1"


class TestAdminCommands:
    """Admin command handling."""

    @pytest.mark.asyncio
    async def test_status_command(self):
        router, orch, adapter = _make_router()
        msg = _make_msg(text="STATUS", sender_id="admin1")
        await router.handle_message(msg)
        assert len(orch.calls) == 0  # admin commands don't go to orchestrator
        assert any("running" in text.lower() for _, text in adapter.sent)

    @pytest.mark.asyncio
    async def test_healme_clears_sessions(self):
        router, orch, adapter = _make_router()

        class FakeSession:
            async def clear(self):
                pass

        orch.sessions["test"] = FakeSession()
        msg = _make_msg(text="HEALME", sender_id="admin1")
        await router.handle_message(msg)
        assert len(orch.sessions) == 0
        assert any("cleared" in text.lower() for _, text in adapter.sent)

    @pytest.mark.asyncio
    async def test_non_admin_cant_use_admin_commands(self):
        router, orch, adapter = _make_router()
        msg = _make_msg(text="STATUS", sender_id="trusted1")
        await router.handle_message(msg)
        # Trusted users don't get admin commands — message goes to orchestrator
        assert len(orch.calls) == 1


class TestQuietMode:
    """Quiet mode blocks non-admin messages."""

    @pytest.mark.asyncio
    async def test_quiet_blocks_trusted(self):
        router, orch, adapter = _make_router()
        # Enable quiet mode
        msg = _make_msg(text="QUIET", sender_id="admin1")
        await router.handle_message(msg)

        # Trusted user should be blocked
        msg = _make_msg(text="hello", sender_id="trusted1")
        await router.handle_message(msg)
        assert len(orch.calls) == 0
        assert any("quiet" in text.lower() for _, text in adapter.sent)

    @pytest.mark.asyncio
    async def test_wake_unblocks(self):
        router, orch, adapter = _make_router()
        # Enable then disable quiet
        await router.handle_message(_make_msg(text="QUIET", sender_id="admin1"))
        await router.handle_message(_make_msg(text="WAKE", sender_id="admin1"))

        # Trusted user should now work
        msg = _make_msg(text="hello", sender_id="trusted1")
        await router.handle_message(msg)
        assert len(orch.calls) == 1

    @pytest.mark.asyncio
    async def test_admin_bypasses_quiet(self):
        router, orch, adapter = _make_router()
        await router.handle_message(_make_msg(text="QUIET", sender_id="admin1"))

        # Admin commands still work
        msg = _make_msg(text="STATUS", sender_id="admin1")
        await router.handle_message(msg)
        assert any("running" in text.lower() for _, text in adapter.sent)
