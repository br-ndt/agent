"""Tests for agent.memory — pointer-based memory system."""

import pytest

from agent.memory import (
    ChromaMemoryBackend,
    MemoryPointer,
    MemoryStore,
    _CHROMA_AVAILABLE,
    build_memory_index_prompt,
    build_topic_summary_for_index,
    infer_room,
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


@pytest.mark.skipif(not _CHROMA_AVAILABLE, reason="chromadb not installed")
class TestSemanticSearch:
    """ChromaDB-backed semantic search over memory topics."""

    @pytest.mark.asyncio
    async def test_semantic_search_finds_related_topic(self, memory_store: MemoryStore):
        """Semantic search should find a topic even when the query uses different words."""
        await memory_store.save_topic(
            "s1", "auth-jwt-migration",
            "Migrating authentication from sessions to JSON Web Tokens",
            "We are replacing cookie-based session auth with JWT tokens. "
            "The refresh token rotation uses RS256 signing. Migration plan "
            "includes a 2-week dual-auth window for backward compatibility.",
            tags=["auth", "security"],
        )
        await memory_store.save_topic(
            "s1", "deploy-pipeline",
            "CI/CD pipeline configuration with Docker",
            "Docker compose setup for staging and production. Uses GitHub "
            "Actions for CI, with blue-green deploys via systemd.",
            tags=["infra", "deploy"],
        )

        # Query uses "login tokens" — semantically close to JWT auth, not keyword-matching
        results = await memory_store.search_topics("s1", "login token security")
        assert len(results) >= 1
        assert results[0].topic == "auth-jwt-migration"

    @pytest.mark.asyncio
    async def test_semantic_search_ranks_by_relevance(self, memory_store: MemoryStore):
        """The most semantically relevant topic should rank first."""
        await memory_store.save_topic(
            "s1", "database-indexing",
            "PostgreSQL index optimization",
            "Added composite indexes on users(email, created_at) and "
            "orders(user_id, status). Query time dropped from 800ms to 12ms.",
        )
        await memory_store.save_topic(
            "s1", "api-rate-limiting",
            "Rate limiting with Redis sliding window",
            "Implemented sliding window rate limiter using Redis sorted sets. "
            "100 requests per minute per API key.",
        )
        await memory_store.save_topic(
            "s1", "db-migration-v3",
            "Database schema migration to v3",
            "Added new columns for user preferences and notification settings. "
            "Ran ALTER TABLE on the users and notifications tables.",
        )

        results = await memory_store.search_topics("s1", "slow SQL query performance")
        assert len(results) >= 1
        # Database indexing is most relevant to query performance
        assert results[0].topic == "database-indexing"

    @pytest.mark.asyncio
    async def test_semantic_search_respects_session_isolation(self, memory_store: MemoryStore):
        """Semantic search should only return topics from the queried session + global."""
        await memory_store.save_topic(
            "s1", "secret-project",
            "Top secret project details",
            "This is session 1 only data about a confidential project.",
        )
        await memory_store.save_topic(
            "s2", "other-project",
            "Session 2 project details",
            "This is session 2 data about a different project.",
        )

        results = await memory_store.search_topics("s2", "secret confidential project")
        topics = [r.topic for r in results]
        assert "secret-project" not in topics

    @pytest.mark.asyncio
    async def test_global_topics_found_from_any_session(self, memory_store: MemoryStore):
        """Global topics should be discoverable via semantic search from any session."""
        await memory_store.save_topic(
            "s1", "coding-standards",
            "Team coding conventions and style guide",
            "All Python code uses snake_case. Max line length 100. "
            "Type hints required on all public functions.",
            global_=True,
        )

        # Search from a different session
        results = await memory_store.search_topics("s2", "code style naming conventions")
        assert len(results) >= 1
        assert results[0].topic == "coding-standards"

    @pytest.mark.asyncio
    async def test_chroma_backend_direct_search(self, memory_store: MemoryStore):
        """Test the ChromaMemoryBackend.search method directly."""
        assert memory_store.chroma.available

        await memory_store.chroma.upsert(
            session_id="s1",
            topic="test-vectors",
            summary="Testing vector embeddings",
            content="This is about machine learning and neural network embeddings.",
        )

        hits = await memory_store.chroma.search("deep learning embeddings", session_id="s1")
        assert len(hits) >= 1
        assert hits[0]["topic"] == "test-vectors"
        assert "distance" in hits[0]

    @pytest.mark.asyncio
    async def test_delete_session_clears_chroma(self, memory_store: MemoryStore):
        """Deleting a session should also remove ChromaDB embeddings."""
        await memory_store.save_topic(
            "s1", "ephemeral-topic",
            "This will be deleted",
            "Temporary content for testing deletion.",
        )

        # Verify it's searchable
        results = await memory_store.search_topics("s1", "temporary deletion test")
        assert len(results) >= 1

        await memory_store.delete_session_memory("s1")

        # Should no longer be found
        hits = await memory_store.chroma.search("temporary deletion test", session_id="s1")
        assert len(hits) == 0


@pytest.mark.skipif(not _CHROMA_AVAILABLE, reason="chromadb not installed")
class TestChromaBackfill:
    """Backfill seeds ChromaDB from existing SQLite data on first start."""

    @pytest.mark.asyncio
    async def test_backfill_populates_chroma_on_first_init(self, tmp_path):
        """Topics saved before ChromaDB was added become searchable after backfill."""
        db_path = tmp_path / "backfill.db"
        chroma_dir = tmp_path / "chroma_backfill"

        # Step 1: Create a store, save topics, then close — simulating pre-upgrade state
        store1 = MemoryStore(db_path=db_path, chroma_dir=chroma_dir)
        await store1.init()
        # Wipe chroma to simulate it not existing before
        store1.chroma._collection.delete(where={"session_id": {"$ne": ""}})

        await store1.save_topic(
            "s1", "neural-nets",
            "Deep learning architecture notes",
            "Transformer attention mechanisms and positional encoding details.",
            tags=["ml"],
        )
        await store1.save_topic(
            "s1", "api-design",
            "REST API conventions",
            "Use plural nouns for resources. Always return JSON. Paginate with cursor.",
            tags=["backend"],
            global_=True,
        )
        # Disable chroma and clear it to simulate pre-upgrade SQLite-only state
        import asyncio
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, lambda: store1.chroma._collection.delete(
                where={"session_id": {"$ne": ""}}
            )
        )
        await store1.close()

        # Step 2: Re-open — backfill should detect empty chroma and populate it
        store2 = MemoryStore(db_path=db_path, chroma_dir=chroma_dir)
        await store2.init()

        # Semantic search should now find the backfilled topics
        results = await store2.search_topics("s1", "machine learning transformers")
        assert len(results) >= 1
        assert results[0].topic == "neural-nets"

        # Global topic should also be findable from another session
        results = await store2.search_topics("s2", "REST endpoint conventions")
        assert len(results) >= 1
        assert results[0].topic == "api-design"

        await store2.close()

    @pytest.mark.asyncio
    async def test_backfill_is_noop_when_already_populated(self, tmp_path):
        """Backfill should not duplicate data on subsequent starts."""
        db_path = tmp_path / "noop.db"
        chroma_dir = tmp_path / "chroma_noop"

        store1 = MemoryStore(db_path=db_path, chroma_dir=chroma_dir)
        await store1.init()
        await store1.save_topic("s1", "topic-a", "Summary A", "Content A")
        await store1.close()

        # Re-open — backfill should see existing data and skip
        store2 = MemoryStore(db_path=db_path, chroma_dir=chroma_dir)
        await store2.init()

        import asyncio
        loop = asyncio.get_running_loop()
        count = await loop.run_in_executor(
            None, lambda: store2.chroma._collection.count()
        )
        # Should have exactly 1 document, not duplicated
        assert count == 1
        await store2.close()


class TestChromaGracefulDegradation:
    """MemoryStore should work fine even when ChromaDB is unavailable."""

    @pytest.mark.asyncio
    async def test_search_falls_back_to_keyword(self, tmp_path):
        """When chroma is not available, search_topics should use keyword fallback."""
        store = MemoryStore(
            db_path=tmp_path / "mem_fallback.db",
            chroma_dir=tmp_path / "chroma_fallback",
        )
        await store.init()

        # Forcibly disable chroma to test fallback
        store.chroma._collection = None

        await store.save_topic("s1", "auth-migration", "JWT auth migration", "details")
        results = await store.search_topics("s1", "auth")
        assert len(results) == 1
        assert results[0].topic == "auth-migration"
        await store.close()

    @pytest.mark.asyncio
    async def test_save_topic_works_without_chroma(self, tmp_path):
        """save_topic should not raise even when chroma is disabled."""
        store = MemoryStore(
            db_path=tmp_path / "mem_nochroma.db",
            chroma_dir=tmp_path / "chroma_nochroma",
        )
        await store.init()
        store.chroma._collection = None

        # Should not raise
        await store.save_topic("s1", "test", "test summary", "test content")
        topic = await store.get_topic("s1", "test")
        assert topic is not None
        assert topic.content == "test content"
        await store.close()


class TestRoomInference:
    """Room auto-inference from topic name, tags, and content."""

    def test_auth_room_from_topic(self):
        assert infer_room("jwt-token-rotation", ["auth"], "") == "auth"

    def test_data_room_from_content(self):
        room = infer_room("schema-update", [], "ALTER TABLE users ADD COLUMN sql migration")
        assert room == "data"

    def test_deploy_room_from_tags(self):
        assert infer_room("release-v2", ["deploy", "docker"], "") == "deploy"

    def test_frontend_room(self):
        assert infer_room("button-refactor", ["ui"], "React component JSX render") == "frontend"

    def test_api_room(self):
        assert infer_room("new-endpoint", ["api"], "REST route handler request") == "api"

    def test_general_fallback(self):
        assert infer_room("random-notes", [], "some unrelated content") == "general"

    def test_requires_minimum_hits(self):
        # Single keyword hit shouldn't be enough
        assert infer_room("something", [], "just one mention of docker") == "general"


class TestRoomStorage:
    """Room is stored and retrieved correctly."""

    @pytest.mark.asyncio
    async def test_save_topic_auto_infers_room(self, memory_store: MemoryStore):
        await memory_store.save_topic(
            "s1", "jwt-migration",
            "Migrating to JWT tokens",
            "We are replacing session auth with JWT token rotation and OAuth2.",
            tags=["auth", "security"],
        )
        index = await memory_store.get_index("s1")
        pointer = next(p for p in index if p.topic == "jwt-migration")
        assert pointer.room == "auth"

    @pytest.mark.asyncio
    async def test_save_topic_explicit_room(self, memory_store: MemoryStore):
        await memory_store.save_topic(
            "s1", "custom-topic",
            "Some topic",
            "Content here",
            room="infra",
        )
        index = await memory_store.get_index("s1")
        pointer = next(p for p in index if p.topic == "custom-topic")
        assert pointer.room == "infra"

    @pytest.mark.asyncio
    async def test_global_topic_gets_room(self, memory_store: MemoryStore):
        await memory_store.save_topic(
            "s1", "db-conventions",
            "Database naming conventions",
            "All database tables use snake_case. SQL migration scripts run via postgres ORM model.",
            global_=True,
        )
        global_index = await memory_store.get_global_index()
        pointer = next(p for p in global_index if p.topic == "db-conventions")
        assert pointer.room == "data"


@pytest.mark.skipif(not _CHROMA_AVAILABLE, reason="chromadb not installed")
class TestRoomScopedSearch:
    """Room-scoped search narrows results to the relevant room."""

    @pytest.mark.asyncio
    async def test_scoped_search_filters_by_room(self, memory_store: MemoryStore):
        await memory_store.save_topic(
            "s1", "jwt-auth-setup",
            "JWT authentication setup",
            "Configuring JWT token signing with RS256 for login authentication.",
            tags=["auth"],
            room="auth",
        )
        await memory_store.save_topic(
            "s1", "docker-deploy",
            "Docker deployment pipeline",
            "Setting up Docker compose for staging with GitHub Actions CI/CD.",
            tags=["deploy"],
            room="deploy",
        )

        # Search scoped to auth room — query is relevant to auth content
        results = await memory_store.search_topics(
            "s1", "JWT token signing authentication login", room="auth"
        )
        topics = [r.topic for r in results]
        assert "jwt-auth-setup" in topics
        assert "docker-deploy" not in topics

    @pytest.mark.asyncio
    async def test_unscoped_search_finds_all(self, memory_store: MemoryStore):
        await memory_store.save_topic(
            "s1", "auth-thing", "Auth",
            "JWT login auth session token password credential",
            room="auth",
        )
        await memory_store.save_topic(
            "s1", "deploy-thing", "Deploy",
            "Docker deploy pipeline container CI build release",
            room="deploy",
        )

        # Without room scope, a broad query finds results across rooms
        results = await memory_store.search_topics("s1", "JWT authentication token")
        assert len(results) >= 1


class TestHierarchicalIndexPrompt:
    """The prompt builder groups pointers by room."""

    def test_groups_by_room(self):
        pointers = [
            MemoryPointer(topic="jwt-setup", summary="JWT config", room="auth"),
            MemoryPointer(topic="login-fix", summary="Fixed login", room="auth"),
            MemoryPointer(topic="docker-ci", summary="Docker CI", room="deploy"),
        ]
        prompt = build_memory_index_prompt(pointers)
        # auth group should appear before deploy (alphabetical)
        auth_pos = prompt.index("**auth**")
        deploy_pos = prompt.index("**deploy**")
        assert auth_pos < deploy_pos
        assert "jwt-setup" in prompt
        assert "docker-ci" in prompt

    def test_general_room_last(self):
        pointers = [
            MemoryPointer(topic="misc", summary="Misc notes", room="general"),
            MemoryPointer(topic="api-routes", summary="API routes", room="api"),
        ]
        prompt = build_memory_index_prompt(pointers)
        api_pos = prompt.index("**api**")
        general_pos = prompt.index("**general**")
        assert api_pos < general_pos

    def test_single_room_no_redundant_header(self):
        pointers = [
            MemoryPointer(topic="a", summary="A", room="general"),
            MemoryPointer(topic="b", summary="B", room="general"),
        ]
        prompt = build_memory_index_prompt(pointers)
        # When all pointers are in "general", no room header needed
        assert "**general**" not in prompt

    def test_backward_compat_empty_room(self):
        pointers = [
            MemoryPointer(topic="old-topic", summary="Old summary"),
        ]
        prompt = build_memory_index_prompt(pointers)
        assert "old-topic" in prompt


class TestRoomBackfill:
    """Existing topics get rooms assigned on init."""

    @pytest.mark.asyncio
    async def test_backfill_assigns_rooms(self, tmp_path):
        db_path = tmp_path / "rooms.db"
        store = MemoryStore(db_path=db_path, chroma_dir=tmp_path / "chroma_rooms")
        await store.init()

        # Save topics (they'll get rooms auto-assigned)
        await store.save_topic(
            "s1", "jwt-auth", "JWT setup", "JWT token auth login session", tags=["auth"],
        )
        await store.save_topic(
            "s1", "docker-deploy", "Docker setup", "Docker container deploy pipeline CI",
            tags=["deploy"],
        )
        await store.close()

        # Simulate pre-migration state by clearing rooms
        import aiosqlite
        async with aiosqlite.connect(db_path) as db:
            await db.execute("UPDATE memory_index SET room = ''")
            await db.execute("UPDATE memory_topics SET room = ''")
            await db.commit()

        # Re-open — backfill should assign rooms
        store2 = MemoryStore(db_path=db_path, chroma_dir=tmp_path / "chroma_rooms")
        await store2.init()

        index = await store2.get_index("s1")
        rooms = {p.topic: p.room for p in index}
        assert rooms["jwt-auth"] == "auth"
        assert rooms["docker-deploy"] == "deploy"
        await store2.close()
