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


@dataclass
class MemoryPointer:
    """A single entry in the memory index — points to a topic."""

    topic: str  # short identifier, e.g. "project-auth-redesign"
    summary: str  # ~150 char description of what this topic contains
    last_accessed: float = 0.0
    access_count: int = 0
    tags: list[str] = field(default_factory=list)

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

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or DB_PATH
        self._db: aiosqlite.Connection | None = None

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
        log.info("memory_store_ready", db=str(self.db_path))

    # ── Index operations ─────────────────────────────────────

    async def get_index(self, session_id: str) -> list[MemoryPointer]:
        """Load the memory index for a session."""
        if not self._db:
            return []

        async with self._db.execute(
            """SELECT topic, summary, tags, last_accessed, access_count
               FROM memory_index WHERE session_id = ?
               ORDER BY last_accessed DESC LIMIT ?""",
            (session_id, MAX_INDEX_ENTRIES),
        ) as cursor:
            rows = await cursor.fetchall()

        pointers = []
        for topic, summary, tags_json, last_accessed, access_count in rows:
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
                )
            )
        return pointers

    async def get_global_index(self) -> list[MemoryPointer]:
        """Load the global memory index (cross-session knowledge)."""
        if not self._db:
            return []

        async with self._db.execute(
            """SELECT topic, summary, tags, last_accessed, access_count
               FROM memory_global
               ORDER BY last_accessed DESC LIMIT ?""",
            (MAX_INDEX_ENTRIES,),
        ) as cursor:
            rows = await cursor.fetchall()

        pointers = []
        for topic, summary, tags_json, last_accessed, access_count in rows:
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
                )
            )
        return pointers

    async def upsert_pointer(
        self, session_id: str, topic: str, summary: str, tags: list[str] | None = None
    ):
        """Create or update a pointer in the session index."""
        if not self._db:
            return

        now = time.time()
        tags_json = json.dumps(tags or [])

        await self._db.execute(
            """INSERT INTO memory_index (session_id, topic, summary, tags, last_accessed, access_count, created_at)
               VALUES (?, ?, ?, ?, ?, 1, ?)
               ON CONFLICT(session_id, topic) DO UPDATE SET
                   summary = excluded.summary,
                   tags = excluded.tags,
                   last_accessed = excluded.last_accessed,
                   access_count = access_count + 1""",
            (session_id, topic, summary[:MAX_POINTER_LEN], tags_json, now, now),
        )
        await self._db.commit()

    async def upsert_global_pointer(
        self, topic: str, summary: str, content: str, tags: list[str] | None = None
    ):
        """Create or update a global (cross-session) memory entry."""
        if not self._db:
            return

        now = time.time()
        tags_json = json.dumps(tags or [])

        await self._db.execute(
            """INSERT INTO memory_global (topic, summary, content, tags, last_accessed, access_count, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, 1, ?, ?)
               ON CONFLICT(topic) DO UPDATE SET
                   summary = excluded.summary,
                   content = excluded.content,
                   tags = excluded.tags,
                   last_accessed = excluded.last_accessed,
                   access_count = access_count + 1,
                   updated_at = excluded.updated_at""",
            (topic, summary[:MAX_POINTER_LEN], content, tags_json, now, now, now),
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
    ):
        """Save topic content and update the index pointer."""
        if not self._db:
            return

        now = time.time()

        # Save topic content
        await self._db.execute(
            """INSERT INTO memory_topics (session_id, topic, content, created_at, updated_at, source_session)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(session_id, topic) DO UPDATE SET
                   content = excluded.content,
                   updated_at = excluded.updated_at""",
            (session_id, topic, content[:10000], now, now, session_id),
        )

        # Update index pointer
        await self.upsert_pointer(session_id, topic, summary, tags)

        # Optionally save to global memory too
        if global_:
            await self.upsert_global_pointer(topic, summary, content[:10000], tags)

        await self._db.commit()
        log.info(
            "memory_topic_saved",
            session_id=session_id,
            topic=topic,
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

        best_match = None
        best_score = 0.0

        for p in pointers:
            if p.topic == topic:
                continue  # don't match self

            # Build word set from existing topic name + summary
            existing_words = set(
                w.lower() for w in f"{p.topic} {p.summary}".replace("-", " ").split()
                if len(w) > 3 and w.isalpha()
            )

            # Also check the stored content for better matching
            existing_topic = await self.get_topic(session_id, p.topic)
            if existing_topic:
                existing_words.update(
                    w.lower() for w in existing_topic.content.split()
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
        self, session_id: str, query: str, limit: int = 5
    ) -> list[MemoryPointer]:
        """Simple keyword search across topic summaries and tags."""
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

    async def close(self):
        if self._db:
            await self._db.close()


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
        for p in pointers:
            lines.append(p.to_index_line())

    if global_pointers:
        lines.append("\n### Cross-session knowledge")
        for p in global_pointers:
            lines.append(p.to_index_line())

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