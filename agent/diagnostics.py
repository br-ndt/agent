"""Diagnostics — error journal and diagnostic report store.

Two components:
  1. ErrorJournal: append-only JSONL file of system errors (state/errors.jsonl).
     Capped at ~5000 lines. Readable by sysadmin for pattern detection.
  2. DiagnosticStore: markdown reports written by sysadmin to _diagnostics/.
     Each report is a timestamped file with structured frontmatter.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import structlog

log = structlog.get_logger()

BASE_DIR = Path(__file__).resolve().parent.parent
ERRORS_PATH = BASE_DIR / "state" / "errors.jsonl"
DIAGNOSTICS_DIR = BASE_DIR / "skills" / "_diagnostics"

MAX_JOURNAL_LINES = 5000


class ErrorJournal:
    """Append-only error log that sysadmin can read via file_ops."""

    def __init__(self, path: Path | None = None):
        self.path = path or ERRORS_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        event: str,
        error: str,
        context: dict | None = None,
    ) -> None:
        """Append an error entry. Non-blocking, never raises."""
        try:
            entry = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "event": event,
                "error": error,
                **(context or {}),
            }
            with open(self.path, "a") as f:
                f.write(json.dumps(entry) + "\n")
            self._maybe_truncate()
        except Exception as e:
            log.warning("error_journal_write_failed", error=str(e))

    def recent(self, n: int = 50) -> list[dict]:
        """Read the last N error entries."""
        if not self.path.exists():
            return []
        try:
            lines = self.path.read_text().strip().splitlines()
            return [json.loads(line) for line in lines[-n:]]
        except Exception:
            return []

    def _maybe_truncate(self) -> None:
        """Keep the journal under MAX_JOURNAL_LINES."""
        try:
            if not self.path.exists():
                return
            lines = self.path.read_text().strip().splitlines()
            if len(lines) > MAX_JOURNAL_LINES:
                # Keep the most recent half
                keep = lines[len(lines) // 2 :]
                self.path.write_text("\n".join(keep) + "\n")
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
