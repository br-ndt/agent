"""Tests for agent.memory — pointer-based memory system."""

import pytest

from agent.memory import (
    MemoryPointer,
    MemoryStore,
    build_memory_index_prompt,
    build_topic_summary_for_index,
    MAX_POINTER_LEN,
)


class TestMemoryStoreRoundtrip:
    """Data saved to the store must be retrievable with correct content."""

    @pytest.mark.asyncio
    async def test_save_and_recall_topic(self, memory_store: MemoryStore):
        await memory_store.save_topic(
            session_id="s1",
            topic="auth-redesign",
            summary="Redesigning auth to use JWT",
            content="Full details about the auth redesign including migration plan...",
            tags=["auth", "backend"],
        )
        topic = await memory_store.get_topic("s1", "auth-redesign")
        assert topic is not None
        assert topic.content == "Full details about the auth redesign including migration plan..."
        assert topic.topic == "auth-redesign"

    @pytest.mark.asyncio
    async def test_topic_not_found_returns_none(self, memory_store: MemoryStore):
        result = await memory_store.get_topic("s1", "nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_upsert_overwrites_content(self, memory_store: MemoryStore):
        await memory_store.save_topic("s1", "t1", "v1 summary", "version 1 content")
        await memory_store.save_topic("s1", "t1", "v2 summary", "version 2 content")
        topic = await memory_store.get_topic("s1", "t1")
        assert topic.content == "version 2 content"

    @pytest.mark.asyncio
    async def test_session_isolation(self, memory_store: MemoryStore):
        """Topics in one session must not leak to another."""
        await memory_store.save_topic("s1", "secret", "s1 summary", "s1 private data")
        await memory_store.save_topic("s2", "secret", "s2 summary", "s2 private data")
        t1 = await memory_store.get_topic("s1", "secret")
        t2 = await memory_store.get_topic("s2", "secret")
        assert t1.content == "s1 private data"
        assert t2.content == "s2 private data"


class TestGlobalMemory:
    """Global memory is accessible from any session."""

    @pytest.mark.asyncio
    async def test_global_topic_accessible_from_any_session(self, memory_store: MemoryStore):
        await memory_store.save_topic(
            session_id="s1",
            topic="project-convention",
            summary="We use snake_case everywhere",
            content="All functions and variables use snake_case.",
            global_=True,
        )
        # Accessible from s2 which never saved it
        topic = await memory_store.get_topic("s2", "project-convention")
        assert topic is not None
        assert "snake_case" in topic.content

    @pytest.mark.asyncio
    async def test_global_index_returns_global_entries(self, memory_store: MemoryStore):
        await memory_store.upsert_global_pointer("gp1", "global pointer 1", "content 1")
        await memory_store.upsert_global_pointer("gp2", "global pointer 2", "content 2")
        index = await memory_store.get_global_index()
        topics = {p.topic for p in index}
        assert "gp1" in topics
        assert "gp2" in topics


class TestMemoryIndex:
    """The index is the lightweight always-in-context map."""

    @pytest.mark.asyncio
    async def test_index_reflects_saved_topics(self, memory_store: MemoryStore):
        await memory_store.save_topic("s1", "topic-a", "Summary A", "Content A")
        await memory_store.save_topic("s1", "topic-b", "Summary B", "Content B")
        index = await memory_store.get_index("s1")
        topics = {p.topic for p in index}
        assert topics == {"topic-a", "topic-b"}

    @pytest.mark.asyncio
    async def test_access_count_increments(self, memory_store: MemoryStore):
        await memory_store.save_topic("s1", "hot-topic", "summary", "content")
        # Access it multiple times
        await memory_store.get_topic("s1", "hot-topic")
        await memory_store.get_topic("s1", "hot-topic")
        index = await memory_store.get_index("s1")
        pointer = next(p for p in index if p.topic == "hot-topic")
        # Initial save (1) + 2 accesses = 3
        assert pointer.access_count >= 3

    @pytest.mark.asyncio
    async def test_delete_session_clears_all(self, memory_store: MemoryStore):
        await memory_store.save_topic("s1", "t1", "sum", "content")
        await memory_store.save_topic("s1", "t2", "sum", "content")
        await memory_store.delete_session_memory("s1")
        index = await memory_store.get_index("s1")
        assert index == []
        assert await memory_store.get_topic("s1", "t1") is None


class TestMemorySearch:
    """Keyword search across topics and tags."""

    @pytest.mark.asyncio
    async def test_search_by_topic_name(self, memory_store: MemoryStore):
        await memory_store.save_topic("s1", "auth-migration", "Moving to JWT", "details")
        await memory_store.save_topic("s1", "database-schema", "New tables", "details")
        results = await memory_store.search_topics("s1", "auth")
        assert len(results) == 1
        assert results[0].topic == "auth-migration"

    @pytest.mark.asyncio
    async def test_search_by_summary_content(self, memory_store: MemoryStore):
        await memory_store.save_topic("s1", "t1", "JWT token rotation plan", "details")
        await memory_store.save_topic("s1", "t2", "Database indexing strategy", "details")
        results = await memory_store.search_topics("s1", "JWT")
        assert len(results) == 1
        assert results[0].topic == "t1"

    @pytest.mark.asyncio
    async def test_search_returns_empty_for_no_match(self, memory_store: MemoryStore):
        await memory_store.save_topic("s1", "t1", "summary", "content")
        results = await memory_store.search_topics("s1", "zzz_no_match_zzz")
        assert results == []


class TestMergeOnEvict:
    """Related topics should be merged instead of creating duplicates."""

    @pytest.mark.asyncio
    async def test_find_related_topic_by_content_overlap(self, memory_store: MemoryStore):
        await memory_store.save_topic(
            "s1", "neondrift-multiplayer",
            "NeonDrift multiplayer sync issues",
            "The NeonDrift multiplayer system has desync issues with remote car positions "
            "and interpolation timing. Players report ghost cars teleporting.",
        )
        related = await memory_store.find_related_topic(
            "s1", "neondrift-desync",
            "Fixed the NeonDrift multiplayer desync by correcting delta baseline. "
            "Remote car interpolation now uses server tick alignment.",
        )
        assert related == "neondrift-multiplayer"

    @pytest.mark.asyncio
    async def test_no_related_topic_for_unrelated_content(self, memory_store: MemoryStore):
        await memory_store.save_topic(
            "s1", "auth-redesign", "Auth migration to JWT", "JWT token rotation and migration plan"
        )
        related = await memory_store.find_related_topic(
            "s1", "deploy-pipeline",
            "Setting up the deploy pipeline with Docker and systemd service files.",
        )
        assert related is None

    @pytest.mark.asyncio
    async def test_merge_appends_content(self, memory_store: MemoryStore):
        await memory_store.save_topic(
            "s1", "project-notes", "Project setup notes", "Initial project setup with SQLite."
        )
        await memory_store.merge_into_topic(
            "s1", "project-notes",
            "Added Redis caching layer.",
            "Project notes updated with caching",
        )
        topic = await memory_store.get_topic("s1", "project-notes")
        assert "Initial project setup" in topic.content
        assert "Redis caching" in topic.content

    @pytest.mark.asyncio
    async def test_merge_caps_content_size(self, memory_store: MemoryStore):
        await memory_store.save_topic(
            "s1", "big-topic", "Big topic", "x" * 9000
        )
        await memory_store.merge_into_topic(
            "s1", "big-topic", "y" * 5000, "More content"
        )
        topic = await memory_store.get_topic("s1", "big-topic")
        assert len(topic.content) <= 10000

    @pytest.mark.asyncio
    async def test_index_stays_compact_with_merging(self, memory_store: MemoryStore):
        """Multiple evictions about the same topic should merge, not create N entries."""
        await memory_store.save_topic(
            "s1", "neondrift-work",
            "NeonDrift game development",
            "Working on the NeonDrift racing game multiplayer features.",
        )
        # Simulate multiple evictions about the same topic
        for i in range(5):
            related = await memory_store.find_related_topic(
                "s1", f"neondrift-chunk-{i}",
                f"More NeonDrift multiplayer work: fixing issue number {i} "
                "with racing game interpolation and server sync.",
            )
            if related:
                await memory_store.merge_into_topic(
                    "s1", related,
                    f"Fixed NeonDrift issue {i}",
                    f"NeonDrift fix {i}",
                )
            else:
                await memory_store.save_topic(
                    "s1", f"neondrift-chunk-{i}",
                    f"NeonDrift fix {i}",
                    f"Fixed NeonDrift issue {i}",
                )

        index = await memory_store.get_index("s1")
        # Should have merged into the original, not created 5 new entries
        assert len(index) <= 2  # original + at most 1 if first didn't match


class TestBuildMemoryIndexPrompt:
    """The prompt builder must produce a parseable, bounded index."""

    def test_empty_index(self):
        prompt = build_memory_index_prompt([], None)
        assert "No memories stored yet" in prompt

    def test_session_pointers_listed(self):
        pointers = [
            MemoryPointer(topic="auth", summary="Auth redesign notes"),
            MemoryPointer(topic="deploy", summary="Deploy pipeline config"),
        ]
        prompt = build_memory_index_prompt(pointers)
        assert "auth" in prompt
        assert "deploy" in prompt
        assert "Session memories" in prompt

    def test_global_pointers_listed_separately(self):
        session = [MemoryPointer(topic="local", summary="Local thing")]
        global_ = [MemoryPointer(topic="global", summary="Cross-session thing")]
        prompt = build_memory_index_prompt(session, global_)
        assert "Session memories" in prompt
        assert "Cross-session knowledge" in prompt

    def test_recall_instructions_present(self):
        prompt = build_memory_index_prompt([])
        assert "<recall" in prompt
        assert "<remember" in prompt


class TestMemoryPointerTruncation:
    """Pointer lines must stay within MAX_POINTER_LEN."""

    def test_long_summary_truncated(self):
        pointer = MemoryPointer(topic="t", summary="x" * 200)
        line = pointer.to_index_line()
        assert len(line) <= MAX_POINTER_LEN

    def test_short_summary_not_truncated(self):
        pointer = MemoryPointer(topic="t", summary="short")
        line = pointer.to_index_line()
        assert "..." not in line


class TestBuildTopicSummaryForIndex:
    """The summarization helper extracts topic/summary/content from messages."""

    def test_extracts_topic_from_user_message(self):
        messages = [
            {"role": "user", "content": "How do I fix the login bug?"},
            {"role": "assistant", "content": "Try checking the session cookie."},
        ]
        topic, summary, content = build_topic_summary_for_index(messages)
        assert "login" in topic.lower() or "how-do-i-fix" in topic.lower()
        assert len(summary) <= MAX_POINTER_LEN
        assert "login" in content.lower()

    def test_uses_hint_when_provided(self):
        messages = [{"role": "user", "content": "anything"}]
        topic, _, _ = build_topic_summary_for_index(messages, topic_hint="my-custom-topic")
        assert topic == "my-custom-topic"

    def test_fallback_topic_when_no_user_message(self):
        messages = [{"role": "assistant", "content": "I can help with that."}]
        topic, _, _ = build_topic_summary_for_index(messages)
        assert topic.startswith("conversation")
