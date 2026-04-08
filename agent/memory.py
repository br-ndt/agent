"""Pointer-based memory system — inspired by Claude Code's MEMORY.md architecture.

Instead of summarizing old messages into one flat paragraph, this system
maintains a lightweight index of topic pointers (~150 chars each) that is
always loaded into context. Full topic content is stored in separate
"topic files" (SQLite rows) and fetched on-demand when the LLM asks for
them via <recall topic="..."> operations.

This replaces the single-paragraph summarization in session_store.py with
a much more token-efficient and precise long-term memory.

Integration points:
  - PersistentSession.get_messages_for_llm() injects the memory index
  - Orchestrator.handle() parses <recall> and <remember> operations
  - Summarization still runs, but now writes topic entries instead of
    a single summary string
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import aiosqlite
import structlog

log = structlog.get_logger()

DB_PATH = Path(__file__).resolve().parent.parent / "state" / "memory.db"

# Each pointer line in the index should be under this many chars
MAX_POINTER_LEN = 150
# Maximum number of pointers in the index (oldest get promoted to "deep" storage)
MAX_INDEX_ENTRIES = 40
# Maximum topic content size to inject into context
MAX_TOPIC_CONTENT = 2000

# ── Room inference ────────────────────────────────────────────

ROOM_KEYWORDS: dict[str, set[str]] = {
    "auth": {
        "auth", "login", "logout", "jwt", "token", "session", "oauth",
        "password", "credential", "permission", "rbac", "role", "signup",
        "register", "sso", "saml", "mfa", "2fa", "cookie",
    },
    "data": {
        "database", "sql", "postgres", "mysql", "sqlite", "mongo",
        "schema", "migration", "table", "index", "query", "redis",
        "cache", "caching", "orm", "model", "column", "row",
    },
    "api": {
        "api", "endpoint", "rest", "graphql", "route", "request",
        "response", "middleware", "handler", "controller", "grpc",
        "webhook", "cors", "rate-limit", "ratelimit", "payload",
    },
    "frontend": {
        "react", "vue", "svelte", "angular", "css", "html", "dom",
        "component", "ui", "ux", "layout", "style", "tailwind",
        "button", "form", "modal", "render", "browser", "jsx", "tsx",
    },
    "deploy": {
        "deploy", "ci", "cd", "docker", "kubernetes", "k8s", "pipeline",
        "systemd", "nginx", "terraform", "ansible", "github-actions",
        "workflow", "build", "release", "artifact", "helm", "container",
    },
    "infra": {
        "server", "hosting", "dns", "monitoring", "logging", "grafana",
        "prometheus", "alert", "metric", "uptime", "load-balancer",
        "ssl", "tls", "certificate", "s3", "aws", "gcp", "azure",
        "cloud", "vpc", "firewall", "network",
    },
    "testing": {
        "test", "testing", "pytest", "jest", "spec", "assert",
        "mock", "fixture", "coverage", "e2e", "integration", "unit",
        "ci", "regression", "snapshot",
    },
}


def infer_room(topic: str, tags: list[str] | None = None, content: str = "") -> str:
    """Infer a room name from topic metadata using keyword scoring.

    Returns the best-matching room name, or "general" if no strong match.
    """
    # Build a bag of lowercase words from all inputs
    text = f"{topic.replace('-', ' ')} {' '.join(tags or '')} {content[:500]}"
    words = set(text.lower().split())

    best_room = "general"
    best_score = 0

    for room, keywords in ROOM_KEYWORDS.items():
        score = len(words & keywords)
        if score > best_score:
            best_score = score
            best_room = room

    # Require at least 2 keyword hits to avoid false positives
    return best_room if best_score >= 2 else "general"


@dataclass
class MemoryPointer:
    """A single entry in the memory index — points to a topic."""

    topic: str  # short identifier, e.g. "project-auth-redesign"
    summary: str  # ~150 char description of what this topic contains
    last_accessed: float = 0.0
    access_count: int = 0
    tags: list[str] = field(default_factory=list)
    room: str = ""  # hierarchical grouping, e.g. "auth", "deploy", "data"

    def to_index_line(self) -> str:
        """Format as a single line for context injection."""
        tags_str = f" [{', '.join(self.tags)}]" if self.tags else ""
        line = f"- {self.topic}: {self.summary}{tags_str}"
        if len(line) > MAX_POINTER_LEN:
            line = line[: MAX_POINTER_LEN - 3] + "..."
        return line


@dataclass
class TopicMemory:
    """Full content for a topic — fetched on demand."""

    topic: str
    content: str  # the actual detailed memory
    created_at: float = 0.0
    updated_at: float = 0.0
    source_session: str = ""  # which session created this


class MemoryStore:
    """SQLite-backed storage for the pointer index and topic content."""

    def __init__(self, db_path: Path | None = None, chroma_dir: Path | None = None):
        self.db_path = db_path or DB_PATH
        self._db: aiosqlite.Connection | None = None
        self.chroma = ChromaMemoryBackend(persist_dir=chroma_dir)

    async def init(self):
        """Create tables if needed."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path)
        await self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS memory_index (
                session_id TEXT NOT NULL,
                topic TEXT NOT NULL,
                summary TEXT NOT NULL,
                tags TEXT DEFAULT '[]',
                last_accessed REAL,
                access_count INTEGER DEFAULT 0,
                created_at REAL,
                PRIMARY KEY (session_id, topic)
            );

            CREATE TABLE IF NOT EXISTS memory_topics (
                session_id TEXT NOT NULL,
                topic TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at REAL,
                updated_at REAL,
                source_session TEXT DEFAULT '',
                PRIMARY KEY (session_id, topic)
            );

            CREATE TABLE IF NOT EXISTS memory_global (
                topic TEXT PRIMARY KEY,
                summary TEXT NOT NULL,
                content TEXT NOT NULL,
                tags TEXT DEFAULT '[]',
                last_accessed REAL,
                access_count INTEGER DEFAULT 0,
                created_at REAL,
                updated_at REAL
            );
            """
        )
        await self._db.commit()

        # Schema migration: add room column to existing tables
        import sqlite3
        for table in ("memory_index", "memory_topics", "memory_global"):
            try:
                await self._db.execute(
                    f"ALTER TABLE {table} ADD COLUMN room TEXT DEFAULT ''"
                )
                await self._db.commit()
                log.info("schema_migration", table=table, column="room")
            except sqlite3.OperationalError:
                pass  # column already exists

        log.info("memory_store_ready", db=str(self.db_path))

        # Initialize ChromaDB semantic backend (non-blocking; degrades gracefully)
        await self.chroma.init()

        # Backfill: if ChromaDB is available but empty, seed it from SQLite
        if self.chroma.available:
            await self._backfill_chroma()

        # Backfill: assign rooms to any topics that don't have one
        await self._backfill_rooms()

    # ── Backfill ──────────────────────────────────────────────

    async def _backfill_chroma(self):
        """One-time migration: copy all existing SQLite topics into ChromaDB.

        Only runs when ChromaDB collection is empty (first start after upgrade).
        Subsequent starts are a no-op.
        """
        loop = asyncio.get_running_loop()
        count = await loop.run_in_executor(
            None, lambda: self.chroma._collection.count()
        )
        if count > 0:
            return  # already populated

        if not self._db:
            return

        # Gather all session topics
        async with self._db.execute(
            """SELECT mt.session_id, mt.topic, mi.summary, mt.content, mi.tags
               FROM memory_topics mt
               LEFT JOIN memory_index mi
                   ON mt.session_id = mi.session_id AND mt.topic = mi.topic"""
        ) as cursor:
            session_rows = await cursor.fetchall()

        # Gather all global topics
        async with self._db.execute(
            "SELECT topic, summary, content, tags FROM memory_global"
        ) as cursor:
            global_rows = await cursor.fetchall()

        total = len(session_rows) + len(global_rows)
        if total == 0:
            return

        log.info("chroma_backfill_start", topics=total)

        for sid, topic, summary, content, tags_json in session_rows:
            try:
                tags = json.loads(tags_json) if tags_json else []
            except (json.JSONDecodeError, TypeError):
                tags = []
            await self.chroma.upsert(
                session_id=sid,
                topic=topic,
                summary=summary or "",
                content=content or "",
                tags=tags,
            )

        for topic, summary, content, tags_json in global_rows:
            try:
                tags = json.loads(tags_json) if tags_json else []
            except (json.JSONDecodeError, TypeError):
                tags = []
            await self.chroma.upsert(
                session_id="global",
                topic=topic,
                summary=summary or "",
                content=content or "",
                tags=tags,
                global_=True,
            )

        log.info("chroma_backfill_done", topics=total)

    async def _backfill_rooms(self):
        """Assign rooms to any topics that don't have one yet.

        Runs on every init but only touches rows with empty room fields,
        so it's effectively a no-op after the first pass.
        """
        if not self._db:
            return

        # Find session topics without a room
        async with self._db.execute(
            """SELECT mt.session_id, mt.topic, mi.tags, mt.content
               FROM memory_topics mt
               LEFT JOIN memory_index mi
                   ON mt.session_id = mi.session_id AND mt.topic = mi.topic
               WHERE mt.room = '' OR mt.room IS NULL"""
        ) as cursor:
            session_rows = await cursor.fetchall()

        # Find global topics without a room
        async with self._db.execute(
            "SELECT topic, tags, content FROM memory_global WHERE room = '' OR room IS NULL"
        ) as cursor:
            global_rows = await cursor.fetchall()

        total = len(session_rows) + len(global_rows)
        if total == 0:
            return

        log.info("room_backfill_start", topics=total)

        for sid, topic, tags_json, content in session_rows:
            try:
                tags = json.loads(tags_json) if tags_json else []
            except (json.JSONDecodeError, TypeError):
                tags = []
            room = infer_room(topic, tags, content or "")
            await self._db.execute(
                "UPDATE memory_topics SET room = ? WHERE session_id = ? AND topic = ?",
                (room, sid, topic),
            )
            await self._db.execute(
                "UPDATE memory_index SET room = ? WHERE session_id = ? AND topic = ?",
                (room, sid, topic),
            )
            # Update ChromaDB metadata too
            if self.chroma.available:
                await self.chroma.upsert(
                    session_id=sid, topic=topic, summary="",
                    content=content or "", tags=tags, room=room,
                )

        for topic, tags_json, content in global_rows:
            try:
                tags = json.loads(tags_json) if tags_json else []
            except (json.JSONDecodeError, TypeError):
                tags = []
            room = infer_room(topic, tags, content or "")
            await self._db.execute(
                "UPDATE memory_global SET room = ? WHERE topic = ?",
                (room, topic),
            )
            if self.chroma.available:
                await self.chroma.upsert(
                    session_id="global", topic=topic, summary="",
                    content=content or "", tags=tags, room=room, global_=True,
                )

        await self._db.commit()
        log.info("room_backfill_done", topics=total)

    # ── Index operations ─────────────────────────────────────

    async def get_index(self, session_id: str) -> list[MemoryPointer]:
        """Load the memory index for a session."""
        if not self._db:
            return []

        async with self._db.execute(
            """SELECT topic, summary, tags, last_accessed, access_count, room
               FROM memory_index WHERE session_id = ?
               ORDER BY last_accessed DESC LIMIT ?""",
            (session_id, MAX_INDEX_ENTRIES),
        ) as cursor:
            rows = await cursor.fetchall()

        pointers = []
        for topic, summary, tags_json, last_accessed, access_count, room in rows:
            try:
                tags = json.loads(tags_json)
            except (json.JSONDecodeError, TypeError):
                tags = []
            pointers.append(
                MemoryPointer(
                    topic=topic,
                    summary=summary,
                    last_accessed=last_accessed or 0.0,
                    access_count=access_count or 0,
                    tags=tags,
                    room=room or "",
                )
            )
        return pointers

    async def get_global_index(self) -> list[MemoryPointer]:
        """Load the global memory index (cross-session knowledge)."""
        if not self._db:
            return []

        async with self._db.execute(
            """SELECT topic, summary, tags, last_accessed, access_count, room
               FROM memory_global
               ORDER BY last_accessed DESC LIMIT ?""",
            (MAX_INDEX_ENTRIES,),
        ) as cursor:
            rows = await cursor.fetchall()

        pointers = []
        for topic, summary, tags_json, last_accessed, access_count, room in rows:
            try:
                tags = json.loads(tags_json)
            except (json.JSONDecodeError, TypeError):
                tags = []
            pointers.append(
                MemoryPointer(
                    topic=topic,
                    summary=summary,
                    last_accessed=last_accessed or 0.0,
                    access_count=access_count or 0,
                    tags=tags,
                    room=room or "",
                )
            )
        return pointers

    async def upsert_pointer(
        self, session_id: str, topic: str, summary: str,
        tags: list[str] | None = None, room: str = "",
    ):
        """Create or update a pointer in the session index."""
        if not self._db:
            return

        now = time.time()
        tags_json = json.dumps(tags or [])

        await self._db.execute(
            """INSERT INTO memory_index (session_id, topic, summary, tags, last_accessed, access_count, created_at, room)
               VALUES (?, ?, ?, ?, ?, 1, ?, ?)
               ON CONFLICT(session_id, topic) DO UPDATE SET
                   summary = excluded.summary,
                   tags = excluded.tags,
                   last_accessed = excluded.last_accessed,
                   access_count = access_count + 1,
                   room = CASE WHEN excluded.room != '' THEN excluded.room ELSE room END""",
            (session_id, topic, summary[:MAX_POINTER_LEN], tags_json, now, now, room),
        )
        await self._db.commit()

    async def upsert_global_pointer(
        self, topic: str, summary: str, content: str,
        tags: list[str] | None = None, room: str = "",
    ):
        """Create or update a global (cross-session) memory entry."""
        if not self._db:
            return

        now = time.time()
        tags_json = json.dumps(tags or [])

        await self._db.execute(
            """INSERT INTO memory_global (topic, summary, content, tags, last_accessed, access_count, created_at, updated_at, room)
               VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?)
               ON CONFLICT(topic) DO UPDATE SET
                   summary = excluded.summary,
                   content = excluded.content,
                   tags = excluded.tags,
                   last_accessed = excluded.last_accessed,
                   access_count = access_count + 1,
                   updated_at = excluded.updated_at,
                   room = CASE WHEN excluded.room != '' THEN excluded.room ELSE room END""",
            (topic, summary[:MAX_POINTER_LEN], content, tags_json, now, now, now, room),
        )
        await self._db.commit()

    # ── Topic operations ─────────────────────────────────────

    async def get_topic(self, session_id: str, topic: str) -> TopicMemory | None:
        """Fetch full topic content. Updates access time."""
        if not self._db:
            return None

        # Try session-specific first, then global
        async with self._db.execute(
            "SELECT content, created_at, updated_at, source_session FROM memory_topics WHERE session_id = ? AND topic = ?",
            (session_id, topic),
        ) as cursor:
            row = await cursor.fetchone()

        if row:
            content, created_at, updated_at, source = row
            # Update access time on the pointer
            await self._db.execute(
                "UPDATE memory_index SET last_accessed = ?, access_count = access_count + 1 WHERE session_id = ? AND topic = ?",
                (time.time(), session_id, topic),
            )
            await self._db.commit()
            return TopicMemory(
                topic=topic,
                content=content,
                created_at=created_at or 0.0,
                updated_at=updated_at or 0.0,
                source_session=source or "",
            )

        # Try global memory
        async with self._db.execute(
            "SELECT content, created_at, updated_at FROM memory_global WHERE topic = ?",
            (topic,),
        ) as cursor:
            row = await cursor.fetchone()

        if row:
            content, created_at, updated_at = row
            await self._db.execute(
                "UPDATE memory_global SET last_accessed = ?, access_count = access_count + 1 WHERE topic = ?",
                (time.time(), topic),
            )
            await self._db.commit()
            return TopicMemory(
                topic=topic,
                content=content,
                created_at=created_at or 0.0,
                updated_at=updated_at or 0.0,
            )

        return None

    async def save_topic(
        self,
        session_id: str,
        topic: str,
        summary: str,
        content: str,
        tags: list[str] | None = None,
        global_: bool = False,
        room: str = "",
    ):
        """Save topic content and update the index pointer."""
        if not self._db:
            return

        # Auto-infer room if not provided
        if not room:
            room = infer_room(topic, tags, content)

        now = time.time()

        # Save topic content
        await self._db.execute(
            """INSERT INTO memory_topics (session_id, topic, content, created_at, updated_at, source_session, room)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(session_id, topic) DO UPDATE SET
                   content = excluded.content,
                   updated_at = excluded.updated_at,
                   room = CASE WHEN excluded.room != '' THEN excluded.room ELSE room END""",
            (session_id, topic, content[:10000], now, now, session_id, room),
        )

        # Update index pointer
        await self.upsert_pointer(session_id, topic, summary, tags, room=room)

        # Optionally save to global memory too
        if global_:
            await self.upsert_global_pointer(topic, summary, content[:10000], tags, room=room)

        await self._db.commit()

        # Dual-write to ChromaDB for semantic search
        await self.chroma.upsert(
            session_id=session_id,
            topic=topic,
            summary=summary,
            content=content[:10000],
            tags=tags,
            global_=global_,
            room=room,
        )

        log.info(
            "memory_topic_saved",
            session_id=session_id,
            topic=topic,
            room=room,
            content_len=len(content),
            global_=global_,
        )

    async def find_related_topic(
        self, session_id: str, topic: str, content: str, threshold: float = 0.3
    ) -> str | None:
        """Find an existing topic that's related enough to merge into.

        Uses keyword overlap between the new content and existing topic
        summaries/names. Returns the topic name to merge into, or None.

        threshold: minimum fraction of new content words found in existing topic.
        """
        if not self._db:
            return None

        pointers = await self.get_index(session_id)
        if not pointers:
            return None

        # Extract significant words from new content (skip short/common words)
        new_words = set(
            w.lower() for w in content.split()
            if len(w) > 3 and w.isalpha()
        )
        if not new_words:
            return None

        # Batch-fetch all topic content in one query to avoid N+1
        topic_contents: dict[str, str] = {}
        if self._db:
            placeholders = ",".join("?" for _ in pointers)
            topic_names = [p.topic for p in pointers]
            async with self._db.execute(
                f"SELECT topic, content FROM memory_topics WHERE session_id = ? AND topic IN ({placeholders})",
                [session_id] + topic_names,
            ) as cursor:
                async for row_topic, row_content in cursor:
                    topic_contents[row_topic] = row_content or ""

        best_match = None
        best_score = 0.0

        for p in pointers:
            if p.topic == topic:
                continue  # don't match self

            # Build word set from existing topic name + summary + content
            existing_words = set(
                w.lower() for w in f"{p.topic} {p.summary}".replace("-", " ").split()
                if len(w) > 3 and w.isalpha()
            )

            existing_content = topic_contents.get(p.topic, "")
            if existing_content:
                existing_words.update(
                    w.lower() for w in existing_content.split()
                    if len(w) > 3 and w.isalpha()
                )

            if not existing_words:
                continue

            overlap = new_words & existing_words
            score = len(overlap) / len(new_words)

            if score > best_score:
                best_score = score
                best_match = p.topic

        if best_score >= threshold:
            log.info(
                "memory_merge_candidate",
                session_id=session_id,
                new_topic=topic,
                merge_into=best_match,
                score=f"{best_score:.2f}",
            )
            return best_match

        return None

    async def merge_into_topic(
        self, session_id: str, target_topic: str, new_content: str, new_summary: str
    ):
        """Append new content to an existing topic and update its summary."""
        if not self._db:
            return

        existing = await self.get_topic(session_id, target_topic)
        if not existing:
            return

        # Append with a separator, cap total size
        merged = existing.content + f"\n\n---\n\n{new_content}"
        if len(merged) > 10000:
            # Keep the most recent content, trim from the front
            merged = "..." + merged[-(10000 - 3):]

        now = time.time()
        await self._db.execute(
            """UPDATE memory_topics SET content = ?, updated_at = ?
               WHERE session_id = ? AND topic = ?""",
            (merged, now, session_id, target_topic),
        )

        # Update the summary to reflect the merged content
        updated_summary = f"{new_summary[:70]} (+ earlier context)"
        await self._db.execute(
            """UPDATE memory_index SET summary = ?, last_accessed = ?
               WHERE session_id = ? AND topic = ?""",
            (updated_summary[:MAX_POINTER_LEN], now, session_id, target_topic),
        )
        await self._db.commit()

        log.info(
            "memory_topic_merged",
            session_id=session_id,
            target=target_topic,
            new_content_len=len(new_content),
            merged_len=len(merged),
        )

    async def search_topics(
        self, session_id: str, query: str, limit: int = 5, room: str = "",
    ) -> list[MemoryPointer]:
        """Search memory topics — prefers semantic (ChromaDB) with keyword fallback.

        If room is provided, search is scoped to that room for higher precision.
        """

        # Try semantic search first
        if self.chroma.available:
            hits = await self.chroma.search(
                query, session_id=session_id, room=room, limit=limit, max_distance=0.8
            )
            if hits:
                # Convert chroma hits back to MemoryPointers for the existing interface
                pointers = []
                seen = set()
                for hit in hits:
                    topic = hit["topic"]
                    if topic in seen:
                        continue
                    seen.add(topic)
                    pointers.append(
                        MemoryPointer(
                            topic=topic,
                            summary=hit.get("snippet", "")[:MAX_POINTER_LEN],
                            tags=hit.get("tags", []),
                            room=hit.get("room", ""),
                        )
                    )
                if pointers:
                    log.debug(
                        "memory_semantic_search",
                        query=query[:60],
                        room=room or "*",
                        hits=len(pointers),
                    )
                    return pointers

        # Fallback: keyword search across topic summaries and tags
        if not self._db:
            return []

        query_lower = query.lower()
        all_pointers = await self.get_index(session_id)
        global_pointers = await self.get_global_index()

        results = []
        seen = set()

        for p in all_pointers + global_pointers:
            if p.topic in seen:
                continue
            # If room filter is active, skip non-matching pointers
            if room and p.room and p.room != room:
                continue
            score = 0
            if query_lower in p.summary.lower():
                score += 2
            if query_lower in p.topic.lower():
                score += 3
            for tag in p.tags:
                if query_lower in tag.lower():
                    score += 1
            if score > 0:
                results.append((score, p))
                seen.add(p.topic)

        results.sort(key=lambda x: (-x[0], -x[1].last_accessed))
        return [p for _, p in results[:limit]]

    # ── Lifecycle ─────────────────────────────────────────────

    async def delete_session_memory(self, session_id: str):
        """Clear all memory for a session."""
        if not self._db:
            return
        await self._db.execute(
            "DELETE FROM memory_index WHERE session_id = ?", (session_id,)
        )
        await self._db.execute(
            "DELETE FROM memory_topics WHERE session_id = ?", (session_id,)
        )
        await self._db.commit()
        await self.chroma.delete_session(session_id)

    async def close(self):
        if self._db:
            await self._db.close()


# ── Semantic search backend (ChromaDB) ────────────────────────────

CHROMA_DIR = Path(__file__).resolve().parent.parent / "state" / "chroma"

try:
    import chromadb

    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False


class ChromaMemoryBackend:
    """Vector-search layer over memory topics using ChromaDB.

    Stores embeddings alongside the existing SQLite memory store so that
    recall can fall back to semantic similarity when exact topic names
    or keyword matches fail.

    ChromaDB's default embedding function (all-MiniLM-L6-v2 via
    sentence-transformers) runs locally — no API calls required.
    """

    def __init__(self, persist_dir: Path | None = None):
        self._persist_dir = persist_dir or CHROMA_DIR
        self._client: "chromadb.ClientAPI | None" = None
        self._collection: "chromadb.Collection | None" = None

    @property
    def available(self) -> bool:
        return _CHROMA_AVAILABLE and self._collection is not None

    async def init(self):
        """Initialize ChromaDB client and collection.

        Runs the (synchronous) ChromaDB setup in a thread so we don't
        block the event loop on first-time model download.
        """
        if not _CHROMA_AVAILABLE:
            log.warning("chroma_not_installed", hint="pip install chromadb")
            return

        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, self._init_sync)
            log.info("chroma_backend_ready", persist_dir=str(self._persist_dir))
        except Exception as exc:
            log.error("chroma_init_failed", error=str(exc))

    def _init_sync(self):
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self._persist_dir))
        self._collection = self._client.get_or_create_collection(
            name="memory_topics",
            metadata={"hnsw:space": "cosine"},
        )

    # ── Write ──────────────────────────────────────────────

    async def upsert(
        self,
        session_id: str,
        topic: str,
        summary: str,
        content: str,
        tags: list[str] | None = None,
        global_: bool = False,
        room: str = "",
    ):
        """Upsert a topic's embedding into the vector store."""
        if not self.available:
            return

        # Combine summary + content for a richer embedding
        doc_text = f"{topic}: {summary}\n\n{content}"[:8000]
        doc_id = f"{session_id}::{topic}"

        metadata = {
            "session_id": session_id,
            "topic": topic,
            "global": global_,
            "tags": ",".join(tags or []),
            "room": room or "",
            "updated_at": time.time(),
        }

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: self._collection.upsert(
                ids=[doc_id],
                documents=[doc_text],
                metadatas=[metadata],
            ),
        )

        # For global topics, also store under a global:: prefix so cross-session
        # searches can find them without knowing the original session_id.
        if global_:
            global_id = f"global::{topic}"
            global_meta = {**metadata, "session_id": "global"}
            await loop.run_in_executor(
                None,
                lambda: self._collection.upsert(
                    ids=[global_id],
                    documents=[doc_text],
                    metadatas=[global_meta],
                ),
            )

    # ── Search ─────────────────────────────────────────────

    async def search(
        self,
        query: str,
        session_id: str | None = None,
        room: str = "",
        limit: int = 5,
        max_distance: float = 1.0,
    ) -> list[dict]:
        """Semantic search over stored topics.

        Returns list of dicts with keys: topic, session_id, distance, room, snippet.
        Results are sorted by relevance (lowest distance first).
        If session_id is provided, results from that session and global topics are
        returned; otherwise all topics are searched.
        If room is provided, results are scoped to that room only.
        Hits with cosine distance > max_distance are filtered out.
        """
        if not self.available:
            return []

        # Build where filter: session scope + optional room scope
        conditions = []
        if session_id:
            conditions.append({
                "$or": [
                    {"session_id": {"$eq": session_id}},
                    {"session_id": {"$eq": "global"}},
                ]
            })
        if room:
            conditions.append({"room": {"$eq": room}})

        where_filter = None
        if len(conditions) == 1:
            where_filter = conditions[0]
        elif len(conditions) > 1:
            where_filter = {"$and": conditions}

        loop = asyncio.get_running_loop()
        try:
            results = await loop.run_in_executor(
                None,
                lambda: self._collection.query(
                    query_texts=[query],
                    n_results=limit,
                    where=where_filter,
                    include=["metadatas", "distances", "documents"],
                ),
            )
        except Exception as exc:
            log.error("chroma_search_failed", error=str(exc))
            return []

        hits = []
        if not results or not results["ids"] or not results["ids"][0]:
            return hits

        for i, doc_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            distance = results["distances"][0][i] if results["distances"] else 1.0
            doc = results["documents"][0][i] if results["documents"] else ""

            if distance > max_distance:
                continue

            hits.append({
                "topic": meta.get("topic", doc_id),
                "session_id": meta.get("session_id", ""),
                "distance": distance,
                "room": meta.get("room", ""),
                "tags": meta.get("tags", "").split(",") if meta.get("tags") else [],
                "snippet": doc[:200] if doc else "",
            })

        return hits

    # ── Lifecycle ──────────────────────────────────────────

    async def delete_session(self, session_id: str):
        """Remove all embeddings for a session."""
        if not self.available:
            return

        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                None,
                lambda: self._collection.delete(
                    where={"session_id": {"$eq": session_id}},
                ),
            )
        except Exception as exc:
            log.error("chroma_delete_failed", error=str(exc))


def _group_pointers_by_room(pointers: list[MemoryPointer]) -> list[str]:
    """Group pointers by room for hierarchical display."""
    from collections import defaultdict

    rooms: dict[str, list[MemoryPointer]] = defaultdict(list)
    for p in pointers:
        rooms[p.room or "general"].append(p)

    lines = []
    # Sort rooms alphabetically, but put "general" last
    sorted_rooms = sorted(rooms.keys(), key=lambda r: (r == "general", r))
    use_headers = len(sorted_rooms) > 1 or (len(sorted_rooms) == 1 and "general" not in sorted_rooms)

    for room in sorted_rooms:
        if use_headers:
            lines.append(f"**{room}**")
        for p in rooms[room]:
            lines.append(p.to_index_line())

    return lines


def build_memory_index_prompt(
    pointers: list[MemoryPointer], global_pointers: list[MemoryPointer] | None = None
) -> str:
    """Build the memory index string for injection into the system prompt.

    This is the lightweight "map of your mind" that's always in context.
    The LLM can use <recall topic="..."> to fetch full details.
    """
    lines = ["## Memory Index"]
    lines.append(
        "You have access to stored memories from past conversations. "
        "To recall details on any topic, emit: <recall topic=\"topic-name\">why you need it</recall>"
    )
    lines.append(
        "To remember something for later, emit: "
        '<remember topic="topic-name" tags="tag1,tag2">what to remember</remember>'
    )
    lines.append(
        "To store knowledge that persists across all sessions, add global=\"true\": "
        '<remember topic="topic-name" global="true">cross-session knowledge</remember>'
    )

    if pointers:
        lines.append("\n### Session memories")
        lines.extend(_group_pointers_by_room(pointers))

    if global_pointers:
        lines.append("\n### Cross-session knowledge")
        lines.extend(_group_pointers_by_room(global_pointers))

    if not pointers and not global_pointers:
        lines.append("\nNo memories stored yet.")

    return "\n".join(lines)


def build_topic_summary_for_index(
    messages: list[dict], topic_hint: str = ""
) -> tuple[str, str, str]:
    """Given a chunk of messages, extract a topic name, summary, and full content.

    This is used by the summarization pipeline to create memory entries
    instead of flat summaries.

    Returns: (topic, summary, content)
    """
    # Build the full content from messages
    content_parts = []
    for msg in messages:
        role = msg.get("role", "unknown")
        text = msg.get("content", "")[:500]
        content_parts.append(f"{role}: {text}")

    content = "\n".join(content_parts)

    # Extract a topic name from the first substantive user message
    topic = topic_hint
    if not topic:
        for msg in messages:
            if msg.get("role") == "user":
                text = msg["content"].strip()
                # Skip system-injected messages
                if text.startswith("[") or text.startswith("Results from"):
                    continue
                # Use first ~40 chars as topic slug
                slug = text[:40].lower().strip()
                slug = slug.replace(" ", "-").replace("/", "-")
                # Remove non-alphanumeric except hyphens
                slug = "".join(c for c in slug if c.isalnum() or c == "-")
                topic = slug or "conversation"
                break

    if not topic:
        topic = f"conversation-{int(time.time())}"

    # Summary is first ~150 chars of the content
    summary = content[:MAX_POINTER_LEN].replace("\n", " ").strip()

    return topic, summary, content