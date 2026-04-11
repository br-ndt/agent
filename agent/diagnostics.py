"""Diagnostics — error journal and diagnostic report store.

Two components:
  1. ErrorJournal: append-only JSONL file of system events (state/errors.jsonl).
     Keeps the last 24 hours or 1000 entries (whichever is larger).
     Older entries are rotated into state/journal_archive/ as dated files.
     Supports multiple log levels: debug, info, warning, error.
  2. DiagnosticStore: markdown reports written by sysadmin to _diagnostics/.
     Each report is a timestamped file with structured frontmatter.
"""

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import structlog

log = structlog.get_logger()

BASE_DIR = Path(__file__).resolve().parent.parent
ERRORS_PATH = BASE_DIR / "state" / "errors.jsonl"
ARCHIVE_DIR = BASE_DIR / "state" / "journal_archive"
DIAGNOSTICS_DIR = BASE_DIR / "skills" / "_diagnostics"

MAX_ENTRIES = 1000
MAX_AGE_HOURS = 24
# Check rotation every N writes to avoid stat-ing the file on every log call
_ROTATION_CHECK_INTERVAL = 50

VALID_LEVELS = ("debug", "info", "warning", "error")


class ErrorJournal:
    """Append-only structured journal with automatic archive rotation.

    Backwards-compatible: ``record()`` still works and writes level="error".
    Use ``log()`` for entries at any level.

    Rotation policy: when the active file exceeds ``MAX_ENTRIES``, entries
    older than ``MAX_AGE_HOURS`` are moved to a dated archive file under
    ``state/journal_archive/``.
    """

    def __init__(self, path: Path | None = None, archive_dir: Path | None = None):
        self.path = path or ERRORS_PATH
        self.archive_dir = archive_dir or ARCHIVE_DIR
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self._write_count = 0

    # ── writing ────────────────────────────────────────────────────

    def record(
        self,
        event: str,
        error: str,
        context: dict | None = None,
    ) -> None:
        """Append an error entry. Non-blocking, never raises."""
        self.log("error", event, error, context)

    def log(
        self,
        level: str,
        event: str,
        message: str,
        context: dict | None = None,
    ) -> None:
        """Append a journal entry at any level. Non-blocking, never raises."""
        try:
            entry = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "level": level if level in VALID_LEVELS else "info",
                "event": event,
                "error": message,  # kept as "error" key for backwards compat
                **(context or {}),
            }
            with open(self.path, "a") as f:
                f.write(json.dumps(entry) + "\n")
            self._write_count += 1
            if self._write_count >= _ROTATION_CHECK_INTERVAL:
                self._write_count = 0
                self._maybe_rotate()
        except Exception:
            pass  # never break the caller

    # ── reading (active journal) ───────────────────────────────────

    def recent(self, n: int = 50) -> list[dict]:
        """Read the last N entries (all levels)."""
        if not self.path.exists():
            return []
        try:
            lines = self.path.read_text().strip().splitlines()
            return [json.loads(line) for line in lines[-n:]]
        except Exception:
            return []

    def query(
        self,
        *,
        level: str | None = None,
        event: str | None = None,
        page: int = 1,
        per_page: int = 50,
    ) -> dict:
        """Query journal entries with filtering and pagination.

        Returns ``{"entries": [...], "total": int, "page": int,
        "per_page": int, "pages": int, "levels": [...], "events": [...]}``.
        """
        return self._query_file(
            self.path, level=level, event=event, page=page, per_page=per_page,
        )

    # ── archive reading ────────────────────────────────────────────

    def list_archives(self) -> list[dict]:
        """List archive files, newest first.

        Returns ``[{"filename": "2026-04-10.jsonl", "date": "2026-04-10",
        "size": 48201, "lines": 312}, ...]``.
        """
        archives = []
        for f in sorted(self.archive_dir.glob("*.jsonl"), reverse=True):
            try:
                line_count = sum(1 for _ in open(f))
            except Exception:
                line_count = 0
            archives.append({
                "filename": f.name,
                "date": f.stem,  # e.g. "2026-04-10"
                "size": f.stat().st_size,
                "lines": line_count,
            })
        return archives

    def query_archive(
        self,
        filename: str,
        *,
        level: str | None = None,
        event: str | None = None,
        page: int = 1,
        per_page: int = 50,
    ) -> dict:
        """Query a specific archive file with the same interface as ``query()``."""
        # Sanitize filename to prevent path traversal
        safe = Path(filename).name
        path = self.archive_dir / safe
        if not path.exists() or path.suffix != ".jsonl":
            return {
                "entries": [], "total": 0, "page": 1,
                "per_page": per_page, "pages": 0,
                "levels": [], "events": [],
            }
        return self._query_file(
            path, level=level, event=event, page=page, per_page=per_page,
        )

    # ── shared query logic ─────────────────────────────────────────

    @staticmethod
    def _query_file(
        path: Path,
        *,
        level: str | None = None,
        event: str | None = None,
        page: int = 1,
        per_page: int = 50,
    ) -> dict:
        empty = {
            "entries": [], "total": 0, "page": 1,
            "per_page": per_page, "pages": 0,
            "levels": [], "events": [],
        }
        if not path.exists():
            return empty
        try:
            lines = path.read_text().strip().splitlines()
        except Exception:
            return empty

        all_entries: list[dict] = []
        level_set: set[str] = set()
        event_set: set[str] = set()
        for raw in reversed(lines):
            try:
                entry = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                continue
            entry.setdefault("level", "error")
            level_set.add(entry["level"])
            event_set.add(entry.get("event", "unknown"))
            all_entries.append(entry)

        filtered = all_entries
        if level:
            allowed = {l.strip() for l in level.split(",")}
            filtered = [e for e in filtered if e.get("level") in allowed]
        if event:
            allowed_events = {ev.strip() for ev in event.split(",")}
            filtered = [e for e in filtered if e.get("event") in allowed_events]

        total = len(filtered)
        pages = max(1, (total + per_page - 1) // per_page)
        page = max(1, min(page, pages))
        start = (page - 1) * per_page
        page_entries = filtered[start : start + per_page]

        return {
            "entries": page_entries,
            "total": total,
            "page": page,
            "per_page": per_page,
            "pages": pages,
            "levels": sorted(level_set),
            "events": sorted(event_set),
        }

    # ── rotation ───────────────────────────────────────────────────

    def _maybe_rotate(self) -> None:
        """Rotate old entries out of the active journal into dated archives."""
        try:
            if not self.path.exists():
                return
            lines = self.path.read_text().strip().splitlines()
            if len(lines) <= MAX_ENTRIES:
                return

            cutoff = datetime.now(timezone.utc) - timedelta(hours=MAX_AGE_HOURS)
            cutoff_iso = cutoff.isoformat()

            keep: list[str] = []
            archive: dict[str, list[str]] = {}  # date -> lines

            for line in lines:
                try:
                    entry = json.loads(line)
                    ts = entry.get("ts", "")
                except (json.JSONDecodeError, ValueError):
                    continue

                if ts >= cutoff_iso:
                    keep.append(line)
                else:
                    # Bucket by date for the archive filename
                    date_key = ts[:10]  # "2026-04-10"
                    archive.setdefault(date_key, []).append(line)

            # Also enforce MAX_ENTRIES on the keep set — if we have more than
            # MAX_ENTRIES even within the time window, archive the oldest
            if len(keep) > MAX_ENTRIES:
                overflow = keep[:-MAX_ENTRIES]
                keep = keep[-MAX_ENTRIES:]
                for line in overflow:
                    try:
                        ts = json.loads(line).get("ts", "")[:10]
                    except Exception:
                        ts = "unknown"
                    archive.setdefault(ts, []).append(line)

            if not archive:
                return  # nothing to rotate

            # Append to dated archive files
            for date_key, archived_lines in archive.items():
                archive_path = self.archive_dir / f"{date_key}.jsonl"
                with open(archive_path, "a") as f:
                    f.write("\n".join(archived_lines) + "\n")

            # Rewrite active journal with only the kept entries
            self.path.write_text("\n".join(keep) + "\n" if keep else "")
        except Exception:
            pass


class DiagnosticStore:
    """Reads diagnostic reports written by sysadmin to _diagnostics/."""

    def __init__(self, directory: Path | None = None):
        self.directory = directory or DIAGNOSTICS_DIR
        self.directory.mkdir(parents=True, exist_ok=True)

    def list_reports(self, limit: int = 20) -> list[dict]:
        """List recent diagnostic reports (newest first)."""
        reports = []
        for f in sorted(self.directory.glob("*.md"), reverse=True):
            reports.append({
                "filename": f.name,
                "modified": datetime.fromtimestamp(
                    f.stat().st_mtime, tz=timezone.utc
                ).isoformat(),
                "size": f.stat().st_size,
            })
            if len(reports) >= limit:
                break
        return reports

    def read_report(self, filename: str) -> str | None:
        """Read a specific diagnostic report."""
        path = self.directory / filename
        if path.exists() and path.parent.resolve() == self.directory.resolve():
            return path.read_text()
        return None

    def summary(self) -> str:
        """One-line summary for admin commands."""
        reports = self.list_reports(limit=5)
        if not reports:
            return "No diagnostic reports."
        lines = [f"**{len(self.list_reports(limit=100))} diagnostic reports** (latest):"]
        for r in reports:
            lines.append(f"  - `{r['filename']}` ({r['modified'][:16]})")
        return "\n".join(lines)


def build_diagnosis_prompt(
    event: str,
    error: str,
    skill_name: str = "",
    step_id: str = "",
    step_output: str = "",
    context: str = "",
) -> str:
    """Build a prompt for sysadmin to diagnose a failure."""
    parts = [
        "## Diagnosis Request\n",
        f"**Event:** {event}\n",
        f"**Error:** {error}\n",
    ]
    if skill_name:
        parts.append(f"**Skill:** {skill_name}\n")
    if step_id:
        parts.append(f"**Failed step:** {step_id}\n")
    if step_output:
        parts.append(f"\n### Step output (last 2000 chars)\n```\n{step_output[-2000:]}\n```\n")
    if context:
        parts.append(f"\n### Additional context\n{context}\n")

    parts.append(
        "\n## Instructions\n"
        "1. Read the relevant skill file (if this is a skill failure)\n"
        "2. Read the error journal at ../state/errors.jsonl for recent patterns\n"
        "3. Read any relevant agent source files under ../agent/ to understand the failure\n"
        "4. Write a diagnostic report to _diagnostics/ (within your workspace) with a descriptive filename like:\n"
        f"   `_diagnostics/{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}-{event.replace('_', '-')}.md`\n\n"
        "The report should include:\n"
        "- **Summary**: one-line description of what went wrong\n"
        "- **Root cause**: your analysis of why it failed\n"
        "- **Evidence**: relevant code/config/log snippets\n"
        "- **Recommended fix**: specific changes needed (file, line, what to change)\n"
        "- **Severity**: low / medium / high / critical\n"
    )
    return "".join(parts)
