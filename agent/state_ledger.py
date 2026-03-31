"""State ledger — ground truth for the managed mesh.

Tracks verifiable facts about what has actually happened in the system:
  - Files created/modified by subagents (with checksums)
  - Delegations completed and their outcomes
  - User-established facts and preferences
  - Workspace state snapshots

The orchestrator queries this ledger to validate subagent claims
(e.g., "I wrote the file" → did the file actually appear?) and to
provide accurate context to downstream agents.

This is the "Physics Engine" from the Managed Mesh pattern: it
maintains the authoritative state of the world, and the orchestrator
checks every action against it.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import aiosqlite
import structlog

log = structlog.get_logger()

DB_PATH = Path(__file__).resolve().parent.parent / "state" / "ledger.db"


class EntryType(str, Enum):
    FILE_CREATED = "file_created"
    FILE_MODIFIED = "file_modified"
    FILE_DELETED = "file_deleted"
    DELEGATION_COMPLETED = "delegation_completed"
    DELEGATION_FAILED = "delegation_failed"
    SKILL_EXECUTED = "skill_executed"
    USER_FACT = "user_fact"  # something the user told us to remember
    SYSTEM_EVENT = "system_event"


@dataclass
class LedgerEntry:
    """A single fact recorded in the ledger."""

    entry_type: EntryType
    key: str  # unique identifier (e.g., file path, delegation ID)
    value: str  # the fact content
    metadata: dict = field(default_factory=dict)
    session_id: str = ""
    agent: str = ""
    timestamp: float = 0.0

    def to_context_line(self) -> str:
        """Format as a short context line for the orchestrator."""
        age = time.time() - self.timestamp if self.timestamp else 0
        age_str = (
            f"{int(age)}s ago"
            if age < 60
            else f"{int(age / 60)}m ago"
            if age < 3600
            else f"{int(age / 3600)}h ago"
        )
        return f"[{self.entry_type.value}] {self.key}: {self.value[:100]} ({age_str}, by {self.agent})"


class StateLedger:
    """SQLite-backed ledger for ground truth state tracking."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or DB_PATH
        self._db: aiosqlite.Connection | None = None

    async def init(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path)
        await self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS ledger (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_type TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                session_id TEXT DEFAULT '',
                agent TEXT DEFAULT '',
                timestamp REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_ledger_type ON ledger(entry_type);
            CREATE INDEX IF NOT EXISTS idx_ledger_key ON ledger(key);
            CREATE INDEX IF NOT EXISTS idx_ledger_session ON ledger(session_id);
            CREATE INDEX IF NOT EXISTS idx_ledger_time ON ledger(timestamp DESC);

            CREATE TABLE IF NOT EXISTS workspace_files (
                path TEXT PRIMARY KEY,
                checksum TEXT NOT NULL,
                size INTEGER NOT NULL,
                agent TEXT DEFAULT '',
                last_modified REAL NOT NULL
            );
            """
        )
        await self._db.commit()
        log.info("state_ledger_ready", db=str(self.db_path))

    # ── Recording ─────────────────────────────────────────────

    async def record(
        self,
        entry_type: EntryType,
        key: str,
        value: str,
        metadata: dict | None = None,
        session_id: str = "",
        agent: str = "",
    ) -> int:
        """Record a fact in the ledger. Returns the entry ID."""
        if not self._db:
            return -1

        now = time.time()
        meta_json = json.dumps(metadata or {})

        cursor = await self._db.execute(
            """INSERT INTO ledger (entry_type, key, value, metadata, session_id, agent, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (entry_type.value, key, value[:5000], meta_json, session_id, agent, now),
        )
        await self._db.commit()
        entry_id = cursor.lastrowid

        log.debug(
            "ledger_recorded",
            type=entry_type.value,
            key=key,
            agent=agent,
            id=entry_id,
        )
        return entry_id

    async def record_file(self, path: str, agent: str = ""):
        """Record or update a file's existence and checksum.

        Call this after a subagent claims to have written a file.
        """
        if not self._db:
            return

        file_path = Path(path)
        if not file_path.exists():
            # File doesn't exist — record as a failed claim
            await self.record(
                EntryType.FILE_DELETED,
                key=str(path),
                value="File does not exist (claimed but not found)",
                agent=agent,
            )
            return False

        # Compute checksum
        content = file_path.read_bytes()
        checksum = hashlib.sha256(content).hexdigest()[:16]

        await self._db.execute(
            """INSERT INTO workspace_files (path, checksum, size, agent, last_modified)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(path) DO UPDATE SET
                   checksum = excluded.checksum,
                   size = excluded.size,
                   agent = excluded.agent,
                   last_modified = excluded.last_modified""",
            (str(path), checksum, len(content), agent, time.time()),
        )
        await self._db.commit()

        await self.record(
            EntryType.FILE_CREATED,
            key=str(path),
            value=f"size={len(content)}, sha256={checksum}",
            agent=agent,
        )
        return True

    # ── Querying ──────────────────────────────────────────────

    async def get_recent(
        self,
        limit: int = 20,
        entry_type: EntryType | None = None,
        session_id: str | None = None,
    ) -> list[LedgerEntry]:
        """Query recent ledger entries."""
        if not self._db:
            return []

        conditions = []
        params: list = []

        if entry_type:
            conditions.append("entry_type = ?")
            params.append(entry_type.value)
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)

        async with self._db.execute(
            f"SELECT entry_type, key, value, metadata, session_id, agent, timestamp FROM ledger {where} ORDER BY timestamp DESC LIMIT ?",
            params,
        ) as cursor:
            rows = await cursor.fetchall()

        return [
            LedgerEntry(
                entry_type=EntryType(r[0]),
                key=r[1],
                value=r[2],
                metadata=json.loads(r[3]) if r[3] else {},
                session_id=r[4],
                agent=r[5],
                timestamp=r[6],
            )
            for r in rows
        ]

    async def verify_file(self, path: str) -> dict:
        """Check if a file exists and matches the ledger.

        Returns {"exists": bool, "matches": bool, "details": str}
        """
        if not self._db:
            return {"exists": False, "matches": False, "details": "Ledger not initialized"}

        file_path = Path(path)
        actual_exists = file_path.exists()

        async with self._db.execute(
            "SELECT checksum, size, agent FROM workspace_files WHERE path = ?",
            (str(path),),
        ) as cursor:
            row = await cursor.fetchone()

        if not row:
            return {
                "exists": actual_exists,
                "matches": False,
                "details": "Not in ledger" + (" but file exists on disk" if actual_exists else ""),
            }

        recorded_checksum, recorded_size, agent = row

        if not actual_exists:
            return {
                "exists": False,
                "matches": False,
                "details": f"Recorded by {agent} but file missing from disk",
            }

        # Check current state
        content = file_path.read_bytes()
        current_checksum = hashlib.sha256(content).hexdigest()[:16]

        matches = current_checksum == recorded_checksum
        return {
            "exists": True,
            "matches": matches,
            "details": (
                f"Matches ledger (by {agent})"
                if matches
                else f"Modified since last recorded by {agent}"
            ),
        }

    async def get_workspace_state(self, workspace_prefix: str = "") -> list[dict]:
        """Get all tracked files, optionally filtered by path prefix."""
        if not self._db:
            return []

        if workspace_prefix:
            async with self._db.execute(
                "SELECT path, checksum, size, agent, last_modified FROM workspace_files WHERE path LIKE ?",
                (f"{workspace_prefix}%",),
            ) as cursor:
                rows = await cursor.fetchall()
        else:
            async with self._db.execute(
                "SELECT path, checksum, size, agent, last_modified FROM workspace_files"
            ) as cursor:
                rows = await cursor.fetchall()

        return [
            {
                "path": r[0],
                "checksum": r[1],
                "size": r[2],
                "agent": r[3],
                "last_modified": r[4],
            }
            for r in rows
        ]

    # ── Validation ────────────────────────────────────────────

    async def validate_delegation_result(
        self,
        agent: str,
        result: str,
        workspace: str = "",
        session_id: str = "",
    ) -> dict:
        """Validate a subagent's claimed result against ground truth.

        Checks:
          1. If the agent claims to have written files, do they exist?
          2. If the result references specific data, is it consistent
             with the ledger?

        Returns {"valid": bool, "issues": list[str], "verified_files": list[str]}
        """
        issues = []
        verified_files = []

        # Check for file creation claims
        import re

        # Common patterns agents use to report file creation
        file_patterns = [
            r"(?:wrote|created|saved|generated)\s+(?:file\s+)?[`'\"]?([^\s`'\"]+\.\w+)",
            r"(?:output|written to)\s+[`'\"]?([^\s`'\"]+\.\w+)",
        ]

        claimed_files = set()
        for pattern in file_patterns:
            for match in re.finditer(pattern, result, re.IGNORECASE):
                claimed_files.add(match.group(1))

        for claimed in claimed_files:
            # Resolve relative to workspace if provided
            if workspace and not Path(claimed).is_absolute():
                full_path = Path(workspace) / claimed
            else:
                full_path = Path(claimed)

            if full_path.exists():
                await self.record_file(str(full_path), agent=agent)
                verified_files.append(str(full_path))
            else:
                issues.append(f"Agent '{agent}' claimed to write '{claimed}' but file not found")

        # Record the delegation result
        await self.record(
            EntryType.DELEGATION_COMPLETED if not issues else EntryType.DELEGATION_FAILED,
            key=f"delegation:{agent}",
            value=result[:500],
            metadata={"issues": issues, "verified_files": verified_files},
            session_id=session_id,
            agent=agent,
        )

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "verified_files": verified_files,
        }

    def build_context_summary(self, entries: list[LedgerEntry]) -> str:
        """Build a compact context block from recent ledger entries."""
        if not entries:
            return ""

        lines = ["## Recent system state"]
        for entry in entries[:15]:
            lines.append(entry.to_context_line())
        return "\n".join(lines)

    # ── Lifecycle ─────────────────────────────────────────────

    async def close(self):
        if self._db:
            await self._db.close()