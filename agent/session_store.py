"""Session persistence with pointer-based memory.

Two-tier memory:
  1. Hot buffer: last N messages in-memory (what the LLM sees directly)
  2. Memory topics: structured topic entries in the MemoryStore, surfaced
     via the lightweight pointer index in the orchestrator's system prompt

When history exceeds MAX_HOT messages, the oldest chunk is written to the
MemoryStore as a searchable topic. The orchestrator fetches detail on
demand via <recall> — no flat summary is injected into the message list.

Storage: SQLite (one row per session). Lightweight, no extra services.
"""

import json
import time
from pathlib import Path

import aiosqlite
import structlog

log = structlog.get_logger()

# Tuning knobs
MAX_HOT = 40  # Messages kept in full detail
SUMMARIZE_CHUNK = 10  # How many old messages to summarize at once
SAVE_INTERVAL = 30  # Auto-save every N seconds

DB_PATH = Path(__file__).resolve().parent.parent / "state" / "sessions.db"


class SessionStore:
    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or DB_PATH
        self._db: aiosqlite.Connection | None = None

    async def init(self):
        """Create the DB and table if needed."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path)
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                summary TEXT DEFAULT '',
                hot_messages TEXT DEFAULT '[]',
                last_active REAL,
                message_count INTEGER DEFAULT 0
            )
        """
        )
        await self._db.commit()
        log.info("session_store_ready", db=str(self.db_path))

    async def load(self, session_id: str) -> dict:
        """Load a session. Returns {summary, history, message_count}."""
        if not self._db:
            return {"summary": "", "history": [], "message_count": 0}

        async with self._db.execute(
            "SELECT summary, hot_messages, message_count FROM sessions WHERE session_id = ?",
            (session_id,),
        ) as cursor:
            row = await cursor.fetchone()

        if not row:
            return {"summary": "", "history": [], "message_count": 0}

        summary, hot_json, count = row
        try:
            history = json.loads(hot_json)
        except json.JSONDecodeError:
            history = []

        log.debug(
            "session_loaded",
            session_id=session_id,
            summary_len=len(summary),
            hot_messages=len(history),
        )

        return {"summary": summary, "history": history, "message_count": count}

    async def save(
        self, session_id: str, summary: str, history: list[dict], message_count: int
    ):
        """Persist a session."""
        if not self._db:
            return

        hot_json = json.dumps(history)
        await self._db.execute(
            """
            INSERT INTO sessions (session_id, summary, hot_messages, last_active, message_count)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                summary = excluded.summary,
                hot_messages = excluded.hot_messages,
                last_active = excluded.last_active,
                message_count = excluded.message_count
        """,
            (session_id, summary, hot_json, time.time(), message_count),
        )
        await self._db.commit()

    async def delete(self, session_id: str):
        """Delete a session."""
        if self._db:
            await self._db.execute(
                "DELETE FROM sessions WHERE session_id = ?", (session_id,)
            )
            await self._db.commit()

    async def list_sessions(self) -> list[dict]:
        """List all sessions with metadata."""
        if not self._db:
            return []

        async with self._db.execute(
            "SELECT session_id, last_active, message_count, length(summary) FROM sessions"
        ) as cursor:
            rows = await cursor.fetchall()

        return [
            {
                "session_id": r[0],
                "last_active": r[1],
                "message_count": r[2],
                "summary_chars": r[3],
            }
            for r in rows
        ]

    async def close(self):
        if self._db:
            await self._db.close()


class PersistentSession:
    """Wraps a single user's conversation with summarization."""

    def __init__(
        self,
        session_id: str,
        store: SessionStore,
        memory_store=None,
    ):
        self.session_id = session_id
        self.store = store
        self.memory_store = memory_store

        self.history: list[dict] = []
        self.message_count: int = 0
        self._loaded = False

        # Vision support
        self.images: list[dict] = []
        self.last_generated_image: bytes | None = None

    async def ensure_loaded(self):
        """Lazy-load from DB on first access."""
        if self._loaded:
            return
        data = await self.store.load(self.session_id)
        self.history = data["history"]
        self.message_count = data["message_count"]
        self._loaded = True

    def get_messages_for_llm(self) -> list[dict]:
        """Build the message list the LLM actually sees.

        Context from older messages is handled by the memory index
        in the system prompt, not by injecting a summary here.
        """
        return list(self.history)

    async def add_message(self, role: str, content: str):
        """Add a message and write old messages to memory if buffer is full."""
        self.history.append({"role": role, "content": content})
        self.message_count += 1

        # Evict oldest messages into memory topics when buffer overflows
        if len(self.history) > MAX_HOT and self.memory_store:
            await self._evict_to_memory()

        # Auto-save
        await self.store.save(
            self.session_id, "", self.history, self.message_count
        )

    async def _evict_to_memory(self):
        """Write the oldest messages to the MemoryStore, merging into related topics when possible."""
        to_evict = self.history[:SUMMARIZE_CHUNK]
        self.history = self.history[SUMMARIZE_CHUNK:]

        try:
            from agent.memory import build_topic_summary_for_index
            topic, summary, content = build_topic_summary_for_index(
                to_evict, topic_hint=""
            )

            # Try to merge into an existing related topic instead of creating a new one
            related = await self.memory_store.find_related_topic(
                self.session_id, topic, content
            )

            if related:
                await self.memory_store.merge_into_topic(
                    self.session_id, related, content, summary
                )
                log.info(
                    "session_evicted_merged",
                    session_id=self.session_id,
                    evicted_messages=len(to_evict),
                    merged_into=related,
                )
            else:
                await self.memory_store.save_topic(
                    session_id=self.session_id,
                    topic=topic,
                    summary=summary,
                    content=content,
                )
                log.info(
                    "session_evicted_new_topic",
                    session_id=self.session_id,
                    evicted_messages=len(to_evict),
                    topic=topic,
                )
        except Exception as e:
            log.warning("memory_eviction_failed", error=str(e))


    async def clear(self):
        """Reset the session."""
        self.history = []
        self.message_count = 0
        await self.store.delete(self.session_id)
