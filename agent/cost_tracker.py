"""Cost and usage tracker — logs every LLM call to SQLite.

Tracks per-request: timestamp, provider, model, tokens, cost, session, agent.
Queryable via methods for per-user, per-provider, per-day breakdowns.

Usage:
    tracker = CostTracker()
    await tracker.init()
    await tracker.log(provider="claude_cli", model="opus", usage={...},
                      session_id="discord:123", agent="coder")
    summary = await tracker.get_summary(days=7)
"""

import time
from pathlib import Path

import aiosqlite
import structlog

log = structlog.get_logger()

DB_PATH = Path(__file__).resolve().parent.parent / "state" / "costs.db"


class CostTracker:
    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or DB_PATH
        self._db: aiosqlite.Connection | None = None

    async def init(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path)
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS usage_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                cache_read_tokens INTEGER DEFAULT 0,
                cost_usd REAL DEFAULT 0,
                session_id TEXT DEFAULT '',
                agent TEXT DEFAULT '',
                duration_ms INTEGER DEFAULT 0
            )
        """)
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_usage_timestamp ON usage_log(timestamp)
        """)
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_usage_session ON usage_log(session_id)
        """)
        await self._db.commit()
        log.info("cost_tracker_ready", db=str(self.db_path))

    async def log_call(
        self,
        provider: str,
        model: str,
        usage: dict,
        session_id: str = "",
        agent: str = "",
        duration_ms: int = 0,
    ):
        """Log a single LLM call."""
        if not self._db:
            return

        await self._db.execute(
            """INSERT INTO usage_log
               (timestamp, provider, model, input_tokens, output_tokens,
                cache_read_tokens, cost_usd, session_id, agent, duration_ms)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                time.time(),
                provider,
                model,
                usage.get("input_tokens", 0),
                usage.get("output_tokens", 0),
                usage.get("cache_read_tokens", 0),
                usage.get("cost_usd", 0),
                session_id,
                agent,
                duration_ms,
            ),
        )
        await self._db.commit()

    async def get_summary(self, days: int = 7) -> dict:
        """Get usage summary for the last N days."""
        if not self._db:
            return {}

        cutoff = time.time() - (days * 86400)

        # Total cost
        async with self._db.execute(
            "SELECT COALESCE(SUM(cost_usd), 0), COUNT(*) FROM usage_log WHERE timestamp > ?",
            (cutoff,),
        ) as cursor:
            row = await cursor.fetchone()
            total_cost = row[0]
            total_calls = row[1]

        # Per-provider breakdown
        async with self._db.execute(
            """SELECT provider, model,
                      COUNT(*) as calls,
                      COALESCE(SUM(cost_usd), 0) as cost,
                      COALESCE(SUM(input_tokens), 0) as input_tok,
                      COALESCE(SUM(output_tokens), 0) as output_tok,
                      COALESCE(AVG(duration_ms), 0) as avg_duration
               FROM usage_log WHERE timestamp > ?
               GROUP BY provider, model
               ORDER BY cost DESC""",
            (cutoff,),
        ) as cursor:
            by_provider = [
                {
                    "provider": r[0],
                    "model": r[1],
                    "calls": r[2],
                    "cost_usd": round(r[3], 4),
                    "input_tokens": r[4],
                    "output_tokens": r[5],
                    "avg_duration_ms": round(r[6]),
                }
                for r in await cursor.fetchall()
            ]

        # Per-session (user) breakdown
        async with self._db.execute(
            """SELECT session_id,
                      COUNT(*) as calls,
                      COALESCE(SUM(cost_usd), 0) as cost,
                      COALESCE(SUM(input_tokens + output_tokens), 0) as total_tokens
               FROM usage_log WHERE timestamp > ? AND session_id != ''
               GROUP BY session_id
               ORDER BY cost DESC
               LIMIT 20""",
            (cutoff,),
        ) as cursor:
            by_session = [
                {
                    "session_id": r[0],
                    "calls": r[1],
                    "cost_usd": round(r[2], 4),
                    "total_tokens": r[3],
                }
                for r in await cursor.fetchall()
            ]

        # Per-agent breakdown
        async with self._db.execute(
            """SELECT agent,
                      COUNT(*) as calls,
                      COALESCE(SUM(cost_usd), 0) as cost,
                      COALESCE(AVG(duration_ms), 0) as avg_duration
               FROM usage_log WHERE timestamp > ? AND agent != ''
               GROUP BY agent
               ORDER BY cost DESC""",
            (cutoff,),
        ) as cursor:
            by_agent = [
                {
                    "agent": r[0],
                    "model": "",
                    "calls": r[1],
                    "cost_usd": round(r[2], 4),
                    "avg_duration_ms": round(r[3]),
                }
                for r in await cursor.fetchall()
            ]

        # Daily totals (last N days)
        async with self._db.execute(
            """SELECT date(timestamp, 'unixepoch') as day,
                      COUNT(*) as calls,
                      COALESCE(SUM(cost_usd), 0) as cost
               FROM usage_log WHERE timestamp > ?
               GROUP BY day
               ORDER BY day DESC""",
            (cutoff,),
        ) as cursor:
            by_day = [
                {"date": r[0], "calls": r[1], "cost_usd": round(r[2], 4)}
                for r in await cursor.fetchall()
            ]

        return {
            "period_days": days,
            "total_cost_usd": round(total_cost, 4),
            "total_calls": total_calls,
            "by_provider": by_provider,
            "by_session": by_session,
            "by_agent": by_agent,
            "by_day": by_day,
        }

    async def get_today(self) -> dict:
        """Quick summary for today only."""
        return await self.get_summary(days=1)

    async def close(self):
        if self._db:
            await self._db.close()