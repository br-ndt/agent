"""Tests for agent.session_store — session persistence with memory eviction."""

import pytest
import pytest_asyncio

from agent.memory import MemoryStore
from agent.session_store import SessionStore, PersistentSession, MAX_HOT, SUMMARIZE_CHUNK


@pytest_asyncio.fixture
async def session_store(tmp_path):
    store = SessionStore(db_path=tmp_path / "sessions.db")
    await store.init()
    yield store
    await store.close()


class TestSessionStoreRoundtrip:
    """Basic save/load/delete on the store."""

    @pytest.mark.asyncio
    async def test_save_and_load(self, session_store):
        history = [{"role": "user", "content": "hello"}]
        await session_store.save("s1", "", history, 1)
        data = await session_store.load("s1")
        assert data["history"] == history
        assert data["message_count"] == 1

    @pytest.mark.asyncio
    async def test_load_missing_session(self, session_store):
        data = await session_store.load("nonexistent")
        assert data["history"] == []
        assert data["message_count"] == 0

    @pytest.mark.asyncio
    async def test_delete(self, session_store):
        await session_store.save("s1", "", [{"role": "user", "content": "hi"}], 1)
        await session_store.delete("s1")
        data = await session_store.load("s1")
        assert data["history"] == []

    @pytest.mark.asyncio
    async def test_list_sessions(self, session_store):
        await session_store.save("s1", "", [], 5)
        await session_store.save("s2", "", [], 10)
        sessions = await session_store.list_sessions()
        ids = {s["session_id"] for s in sessions}
        assert ids == {"s1", "s2"}


class TestPersistentSessionMessages:
    """get_messages_for_llm returns only the hot buffer."""

    @pytest.mark.asyncio
    async def test_messages_are_just_history(self, session_store):
        session = PersistentSession("s1", session_store)
        await session.ensure_loaded()
        await session.add_message("user", "hello")
        await session.add_message("assistant", "hi there")
        messages = session.get_messages_for_llm()
        assert len(messages) == 2
        assert messages[0] == {"role": "user", "content": "hello"}
        assert messages[1] == {"role": "assistant", "content": "hi there"}

    @pytest.mark.asyncio
    async def test_no_summary_injection(self, session_store):
        """Even with old data that has a summary column, we don't inject it."""
        # Simulate old data with a summary in the DB
        await session_store.save("s1", "old flat summary here", [
            {"role": "user", "content": "recent message"},
        ], 50)
        session = PersistentSession("s1", session_store)
        await session.ensure_loaded()
        messages = session.get_messages_for_llm()
        # Should only contain the hot buffer, no summary preamble
        assert len(messages) == 1
        assert messages[0]["content"] == "recent message"
        assert not any("Previous conversation summary" in m["content"] for m in messages)


class TestMemoryEviction:
    """When the hot buffer overflows, old messages go to MemoryStore."""

    @pytest.mark.asyncio
    async def test_eviction_writes_to_memory(self, session_store, memory_store):
        session = PersistentSession("s1", session_store, memory_store=memory_store)
        await session.ensure_loaded()

        # Fill the buffer past MAX_HOT to trigger eviction
        for i in range(MAX_HOT + 5):
            role = "user" if i % 2 == 0 else "assistant"
            await session.add_message(role, f"Message {i}")

        # Hot buffer should have been trimmed
        assert len(session.history) <= MAX_HOT

        # Memory store should have at least one topic from the eviction
        index = await memory_store.get_index("s1")
        assert len(index) >= 1

    @pytest.mark.asyncio
    async def test_eviction_preserves_recent_messages(self, session_store, memory_store):
        session = PersistentSession("s1", session_store, memory_store=memory_store)
        await session.ensure_loaded()

        total = MAX_HOT + 5
        for i in range(total):
            role = "user" if i % 2 == 0 else "assistant"
            await session.add_message(role, f"Message {i}")

        # The most recent messages must still be in the hot buffer
        last_content = session.history[-1]["content"]
        assert last_content == f"Message {total - 1}"

    @pytest.mark.asyncio
    async def test_no_eviction_without_memory_store(self, session_store):
        """Without a memory_store, messages just accumulate (no crash)."""
        session = PersistentSession("s1", session_store, memory_store=None)
        await session.ensure_loaded()

        for i in range(MAX_HOT + 5):
            role = "user" if i % 2 == 0 else "assistant"
            await session.add_message(role, f"Message {i}")

        # All messages stay in history since there's nowhere to evict to
        assert len(session.history) == MAX_HOT + 5

    @pytest.mark.asyncio
    async def test_evicted_content_is_recallable(self, session_store, memory_store):
        """Evicted messages should be retrievable from the memory store."""
        session = PersistentSession("s1", session_store, memory_store=memory_store)
        await session.ensure_loaded()

        # Add a distinctive message that will get evicted
        await session.add_message("user", "The secret password is swordfish")
        # Fill the rest to push it out
        for i in range(MAX_HOT + 5):
            role = "user" if i % 2 == 0 else "assistant"
            await session.add_message(role, f"Filler {i}")

        # The distinctive message should be in a memory topic
        index = await memory_store.get_index("s1")
        found = False
        for pointer in index:
            topic = await memory_store.get_topic("s1", pointer.topic)
            if topic and "swordfish" in topic.content:
                found = True
                break
        assert found, "Evicted message content not found in any memory topic"


class TestMergeOnEvict:
    """Evictions about the same topic should merge instead of creating duplicates."""

    @pytest.mark.asyncio
    async def test_repeated_evictions_merge(self, session_store, memory_store):
        """Talking about the same thing across many messages should produce few topics."""
        # Seed a topic so there's something to merge into
        await memory_store.save_topic(
            "s1", "neondrift-racing",
            "NeonDrift racing game discussion",
            "Discussing the NeonDrift multiplayer racing game features and bugs.",
        )

        session = PersistentSession("s1", session_store, memory_store=memory_store)
        await session.ensure_loaded()

        # Fill with messages about the same topic to trigger multiple evictions
        for i in range(MAX_HOT + SUMMARIZE_CHUNK + 5):
            role = "user" if i % 2 == 0 else "assistant"
            await session.add_message(
                role,
                f"NeonDrift multiplayer racing game: working on sync issue {i}. "
                "The interpolation and server tick need alignment for ghost cars.",
            )

        index = await memory_store.get_index("s1")
        # With merging, we should have far fewer topics than eviction cycles
        # (without merging, each eviction creates a new topic)
        assert len(index) <= 3


class TestSessionClear:
    """clear() must reset state."""

    @pytest.mark.asyncio
    async def test_clear_resets_everything(self, session_store):
        session = PersistentSession("s1", session_store)
        await session.ensure_loaded()
        await session.add_message("user", "hello")
        await session.clear()
        assert session.history == []
        assert session.message_count == 0
        # DB should also be empty
        data = await session_store.load("s1")
        assert data["history"] == []
