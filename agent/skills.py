"""Skill system — structured, executable skills with registry and step runner.

Evolves from the original skills.py. Skills now support structured steps
(shell commands, agent reasoning, conditionals) in addition to the original
markdown-only format. Backward compatible with existing SKILL.md files.

Storage: YAML manifests on disk (git-tracked), metadata + run history in SQLite.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import asyncio
import json
import sqlite3
import time
import yaml
import re
import structlog

log = structlog.get_logger()

DEFAULT_SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"
DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "state" / "skills.db"


def sanitize_skill_name(name: str) -> str:
    """
    Sanitize a skill name for use as a directory/file name.

    - Takes only the first line (in case newlines are present)
    - Removes/replaces invalid characters
    - Truncates to reasonable length
    - Converts to lowercase with hyphens
    """
    # Take only first line (before any newline)
    name = name.split("\n")[0].strip()

    # Remove any system context markers that might have leaked in
    if "[system" in name.lower():
        name = name.split("[System")[0].strip()
    if "[bot]" in name.lower():
        name = name.split("[")[0].strip()

    # Truncate to max 50 chars for the base name
    name = name[:50].strip()

    # Replace spaces and invalid characters
    name = name.lower()
    name = name.replace(" ", "-")

    # Remove any remaining invalid filesystem characters

    name = re.sub(r"[^a-z0-9\-_]", "", name)

    # Ensure it doesn't start/end with hyphen
    name = name.strip("-")

    # Fallback if empty
    if not name:
        name = "unnamed-skill"

    return name


# ── Data Models ─────────────────────────────────────────────────────


@dataclass
class SkillStep:
    """A single step in a skill's execution plan."""

    id: str
    type: str  # "shell", "agent", "conditional", "parallel"
    description: str = ""
    command: str = ""  # for shell steps
    prompt: str = ""  # for agent steps
    on_failure: str = "abort"  # "abort", "continue", "retry"
    condition: str = ""  # for conditional steps
    branches: dict[str, list[str]] = field(default_factory=dict)
    sub_steps: list["SkillStep"] = field(default_factory=list)  # for parallel
    subagent: str = ""  # per-step subagent override (falls back to skill.subagent)


@dataclass
class StepResult:
    """Result of executing a single step."""

    step_id: str
    status: str  # "success", "failed", "skipped"
    output: str = ""
    error: str = ""
    duration_ms: int = 0


@dataclass
class SkillRun:
    """Record of a complete skill execution."""

    skill_name: str
    session_id: str = ""
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    status: str = "running"  # "running", "success", "failed", "aborted"
    step_results: list[StepResult] = field(default_factory=list)
    error: str = ""
    triggered_by: str = ""
    current_step: str = ""  # id of the step currently executing
    total_steps: int = 0


@dataclass
class Skill:
    """A skill definition. Backward compatible with the original Skill dataclass.

    New fields (steps, executor config) are optional — old-style skills
    with just markdown content still work.
    """

    name: str
    description: str
    triggers: list[str] = field(default_factory=list)
    subagent: str = ""
    allowed_tools: list[str] = field(default_factory=list)
    content: str = ""  # markdown instructions (original format)
    path: Path | None = None

    # New structured fields
    version: int = 1
    author: str = ""
    tags: list[str] = field(default_factory=list)
    steps: list[SkillStep] = field(default_factory=list)
    timeout: int = 300
    status: str = "active"  # "active", "proposed", "disabled"

    # Executor overrides
    model_override: str = ""
    tools_override: list[str] = field(default_factory=list)
    mount_paths: list[str] = field(default_factory=list)
    env_passthrough: list[str] = field(default_factory=list)
    output_format: str = "summary"  # "summary", "full", "structured"
    include_step_outputs: bool = True

    @property
    def has_steps(self) -> bool:
        """Whether this skill has structured steps (vs. just markdown)."""
        return len(self.steps) > 0

    def to_markdown(self) -> str:
        """Serialize skill to markdown with YAML frontmatter."""
        frontmatter: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "triggers": self.triggers,
            "subagent": self.subagent,
            "allowed_tools": self.allowed_tools,
        }
        # Only include new fields if they have non-default values
        if self.version > 1:
            frontmatter["version"] = self.version
        if self.author:
            frontmatter["author"] = self.author
        if self.tags:
            frontmatter["tags"] = self.tags
        if self.steps:
            frontmatter["steps"] = [_step_to_dict(s) for s in self.steps]
        if self.timeout != 300:
            frontmatter["timeout"] = self.timeout
        if self.model_override:
            frontmatter["model_override"] = self.model_override
        if self.tools_override:
            frontmatter["tools_override"] = self.tools_override
        if self.mount_paths:
            frontmatter["mount_paths"] = self.mount_paths
        if self.env_passthrough:
            frontmatter["env_passthrough"] = self.env_passthrough

        fm_str = yaml.dump(frontmatter, sort_keys=False).strip()
        return f"---\n{fm_str}\n---\n\n{self.content}"

    def catalog_entry(self) -> str:
        """Compact representation for injection into orchestrator system prompt."""
        triggers_str = (
            ", ".join(f'"{t}"' for t in self.triggers) if self.triggers else "none"
        )
        if self.has_steps:
            step_flow = " → ".join(s.id for s in self.steps)
            return (
                f"### {self.name}\n"
                f"{self.description} | Triggers: {triggers_str}\n"
                f"Executor: {self.subagent} | Steps: {step_flow}"
            )
        else:
            return (
                f"### {self.name}\n"
                f"{self.description} | Triggers: {triggers_str}\n"
                f"Executor: {self.subagent}"
            )


def _step_to_dict(step: SkillStep) -> dict:
    """Serialize a SkillStep to a dict for YAML output."""
    d: dict[str, Any] = {"id": step.id, "type": step.type}
    if step.description:
        d["description"] = step.description
    if step.subagent:
        d["subagent"] = step.subagent
    if step.command:
        d["command"] = step.command
    if step.prompt:
        d["prompt"] = step.prompt
    if step.on_failure != "abort":
        d["on_failure"] = step.on_failure
    if step.condition:
        d["condition"] = step.condition
    if step.branches:
        d["branches"] = step.branches
    if step.sub_steps:
        d["sub_steps"] = [_step_to_dict(s) for s in step.sub_steps]
    return d


def _dict_to_step(d: dict) -> SkillStep:
    """Parse a SkillStep from a dict (YAML frontmatter)."""
    return SkillStep(
        id=d.get("id", ""),
        type=d.get("type", "shell"),
        description=d.get("description", ""),
        command=d.get("command", ""),
        prompt=d.get("prompt", ""),
        on_failure=d.get("on_failure", "abort"),
        condition=d.get("condition", ""),
        branches=d.get("branches", {}),
        sub_steps=[_dict_to_step(s) for s in d.get("sub_steps", [])],
        subagent=d.get("subagent", ""),
    )


# ── Loading / Saving ────────────────────────────────────────────────


def load_skills(skills_dir: Path | None = None) -> list[Skill]:
    """Load all skills from the directory. Backward compatible.

    Each skill is expected at `skills_dir/<skill_folder>/SKILL.md`.
    """
    directory = skills_dir or DEFAULT_SKILLS_DIR
    skills: list[Skill] = []

    if not directory.exists():
        log.debug("skills_dir_not_found", path=str(directory))
        return skills

    for skill_folder in directory.iterdir():
        if not skill_folder.is_dir():
            continue
        # Skip proposed skills directory
        if skill_folder.name == "_proposed":
            continue

        skill_file = skill_folder / "SKILL.md"
        if not skill_file.exists():
            continue

        try:
            skill = _load_skill_file(skill_file, skill_folder)
            if skill:
                skills.append(skill)
                log.info(
                    "skill_loaded",
                    name=skill.name,
                    subagent=skill.subagent,
                    has_steps=skill.has_steps,
                )
        except Exception as e:
            log.error("skill_load_failed", path=str(skill_file), error=str(e))

    return skills


def load_proposed_skills(skills_dir: Path | None = None) -> list[Skill]:
    """Load skills from the _proposed/ staging directory."""
    directory = (skills_dir or DEFAULT_SKILLS_DIR) / "_proposed"
    if not directory.exists():
        return []

    skills = []
    for skill_folder in directory.iterdir():
        if not skill_folder.is_dir():
            continue
        skill_file = skill_folder / "SKILL.md"
        if not skill_file.exists():
            continue
        try:
            skill = _load_skill_file(skill_file, skill_folder)
            if skill:
                skill.status = "proposed"
                skills.append(skill)
        except Exception as e:
            log.error("proposed_skill_load_failed", path=str(skill_file), error=str(e))

    return skills


def _load_skill_file(skill_file: Path, skill_folder: Path) -> Skill | None:
    """Parse a single SKILL.md file into a Skill object."""
    raw_text = skill_file.read_text(encoding="utf-8")
    if not raw_text.startswith("---"):
        log.warning("skill_missing_frontmatter", path=str(skill_file))
        return None

    _, fm_raw, content = raw_text.split("---", 2)
    fm = yaml.safe_load(fm_raw)
    content = content.strip()

    # Parse steps if present
    steps = []
    for step_dict in fm.get("steps", []):
        steps.append(_dict_to_step(step_dict))

    return Skill(
        name=fm.get("name", skill_folder.name),
        description=fm.get("description", ""),
        triggers=fm.get("triggers", []),
        subagent=fm.get("subagent", ""),
        allowed_tools=fm.get("allowed_tools", []),
        content=content,
        path=skill_file,
        version=fm.get("version", 1),
        author=fm.get("author", ""),
        tags=fm.get("tags", []),
        steps=steps,
        timeout=fm.get("timeout", 300),
        status=fm.get("status", "active"),
        model_override=fm.get("model_override", ""),
        tools_override=fm.get("tools_override", []),
        mount_paths=fm.get("mount_paths", []),
        env_passthrough=fm.get("env_passthrough", []),
        output_format=fm.get("output_format", "summary"),
        include_step_outputs=fm.get("include_step_outputs", True),
    )


def save_skill(skill: Skill, skills_dir: Path | None = None) -> Path:
    """Save a skill to the directory."""
    directory = skills_dir or DEFAULT_SKILLS_DIR
    skill_folder = directory / sanitize_skill_name(skill.name)
    skill_folder.mkdir(parents=True, exist_ok=True)

    skill_file = skill_folder / "SKILL.md"
    skill_file.write_text(skill.to_markdown(), encoding="utf-8")

    log.info("skill_saved", name=skill.name, path=str(skill_file))
    return skill_file


def save_proposed_skill(skill: Skill, skills_dir: Path | None = None) -> Path:
    """Save a skill to the _proposed/ staging directory."""
    directory = (skills_dir or DEFAULT_SKILLS_DIR) / "_proposed"
    skill_folder = directory / sanitize_skill_name(skill.name)
    skill_folder.mkdir(parents=True, exist_ok=True)

    skill_file = skill_folder / "SKILL.md"
    skill_file.write_text(skill.to_markdown(), encoding="utf-8")

    log.info("proposed_skill_saved", name=skill.name, path=str(skill_file))
    return skill_file


def approve_skill(name: str, skills_dir: Path | None = None) -> bool:
    """Move a proposed skill from _proposed/ to the active skills directory."""
    directory = skills_dir or DEFAULT_SKILLS_DIR
    safe_name = sanitize_skill_name(name)
    proposed_dir = directory / "_proposed" / safe_name
    active_dir = directory / safe_name

    if not proposed_dir.exists():
        log.warning("proposed_skill_not_found", name=name)
        return False

    # Move the folder
    import shutil

    if active_dir.exists():
        shutil.rmtree(active_dir)
    shutil.move(str(proposed_dir), str(active_dir))

    log.info("skill_approved", name=name)
    return True


def reject_skill(name: str, skills_dir: Path | None = None) -> bool:
    """Delete a proposed skill."""
    directory = (
        (skills_dir or DEFAULT_SKILLS_DIR) / "_proposed" / sanitize_skill_name(name)
    )
    if not directory.exists():
        return False

    import shutil

    shutil.rmtree(directory)
    log.info("skill_rejected", name=name)
    return True


# ── Registry (SQLite index + search) ────────────────────────────────


class SkillRegistry:
    """In-memory skill registry backed by SQLite for search and run history."""

    def __init__(self, skills_dir: Path | None = None, db_path: Path | None = None):
        self.skills_dir = skills_dir or DEFAULT_SKILLS_DIR
        self.db_path = db_path or DEFAULT_DB_PATH
        self.skills: dict[str, Skill] = {}
        self._init_db()
        self.reload()

    def _init_db(self) -> None:
        """Create the SQLite tables if they don't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS skill_meta (
                    name TEXT PRIMARY KEY,
                    last_used TIMESTAMP,
                    use_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS skill_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    skill_name TEXT NOT NULL,
                    session_id TEXT,
                    started_at TIMESTAMP NOT NULL,
                    completed_at TIMESTAMP,
                    status TEXT NOT NULL,
                    step_results TEXT,
                    error TEXT,
                    triggered_by TEXT
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS skill_search USING fts5(
                    name, description, tags, triggers
                );
            """
            )

    def reload(self) -> None:
        """Re-scan skills directory and rebuild the in-memory index."""
        self.skills.clear()

        # Load active skills
        for skill in load_skills(self.skills_dir):
            self.skills[skill.name.lower()] = skill

        # Load proposed skills
        for skill in load_proposed_skills(self.skills_dir):
            self.skills[skill.name.lower()] = skill

        # Rebuild FTS index
        self._rebuild_search_index()

        log.info("skill_registry_reloaded", count=len(self.skills))

    def _rebuild_search_index(self) -> None:
        """Rebuild the FTS5 search index from current skills."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("DELETE FROM skill_search")
            for skill in self.skills.values():
                conn.execute(
                    "INSERT INTO skill_search (name, description, tags, triggers) VALUES (?, ?, ?, ?)",
                    (
                        skill.name,
                        skill.description,
                        " ".join(skill.tags),
                        " ".join(skill.triggers),
                    ),
                )

    def find(self, query: str, top_k: int = 3) -> list[Skill]:
        """Find skills matching a natural language query using FTS5."""
        with sqlite3.connect(str(self.db_path)) as conn:
            # FTS5 search across name, description, tags, triggers
            rows = conn.execute(
                """SELECT name, rank FROM skill_search
                   WHERE skill_search MATCH ? ORDER BY rank LIMIT ?""",
                (query, top_k),
            ).fetchall()

        results = []
        for name, _rank in rows:
            skill = self.skills.get(name.lower())
            if skill and skill.status == "active":
                results.append(skill)
        return results

    def find_by_trigger(self, message: str) -> Skill | None:
        """Check if any skill's trigger phrases appear in a message."""
        msg_lower = message.lower()
        for skill in self.skills.values():
            if skill.status != "active":
                continue
            for trigger in skill.triggers:
                if trigger.lower() in msg_lower:
                    return skill
        return None

    def get(self, name: str) -> Skill | None:
        """Get a specific skill by name."""
        return self.skills.get(name.lower())

    def list_active(self) -> list[Skill]:
        """List all active skills."""
        return [s for s in self.skills.values() if s.status == "active"]

    def list_proposed(self) -> list[Skill]:
        """List skills awaiting approval."""
        return [s for s in self.skills.values() if s.status == "proposed"]

    def approve(self, name: str) -> bool:
        """Approve a proposed skill — move it to active."""
        if approve_skill(name, self.skills_dir):
            self.reload()
            return True
        return False

    def reject(self, name: str) -> bool:
        """Reject a proposed skill — delete it."""
        if reject_skill(name, self.skills_dir):
            self.reload()
            return True
        return False

    def disable(self, name: str) -> bool:
        """Disable a skill without deleting files."""
        skill = self.get(name)
        if not skill:
            return False
        skill.status = "disabled"
        if skill.path:
            save_skill(skill, self.skills_dir)
        self.reload()
        return True

    def enable(self, name: str) -> bool:
        """Re-enable a disabled skill."""
        skill = self.get(name)
        if not skill:
            return False
        skill.status = "active"
        if skill.path:
            save_skill(skill, self.skills_dir)
        self.reload()
        return True

    def record_run(self, run: SkillRun) -> None:
        """Log a skill execution to the database."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """INSERT INTO skill_runs
                   (skill_name, session_id, started_at, completed_at, status, step_results, error, triggered_by)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run.skill_name,
                    run.session_id,
                    run.started_at.isoformat(),
                    run.completed_at.isoformat() if run.completed_at else None,
                    run.status,
                    json.dumps(
                        [
                            {
                                "step_id": r.step_id,
                                "status": r.status,
                                "output": r.output[:500],
                                "duration_ms": r.duration_ms,
                            }
                            for r in run.step_results
                        ]
                    ),
                    run.error,
                    run.triggered_by,
                ),
            )
            # Update use count
            conn.execute(
                """INSERT INTO skill_meta (name, last_used, use_count)
                   VALUES (?, CURRENT_TIMESTAMP, 1)
                   ON CONFLICT(name) DO UPDATE SET
                   last_used = CURRENT_TIMESTAMP, use_count = use_count + 1""",
                (run.skill_name,),
            )

    def get_run_history(self, skill_name: str, limit: int = 10) -> list[dict]:
        """Get recent runs for a skill."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT * FROM skill_runs WHERE skill_name = ?
                   ORDER BY started_at DESC LIMIT ?""",
                (skill_name, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def build_catalog(self) -> str:
        """Build the skill catalog string for orchestrator system prompt injection."""
        active = self.list_active()
        if not active:
            return "No skills currently available."
        return "\n\n".join(s.catalog_entry() for s in active)


# ── Skill Executor ──────────────────────────────────────────────────


class SkillExecutor:
    """Executes a skill's structured steps.

    For shell steps: runs the command via the subagent's bash tool.
    For agent steps: sends the prompt to the subagent with context from prior steps.
    For skills without steps: falls back to the original delegation behavior.
    """

    def __init__(self, registry: SkillRegistry):
        self.registry = registry
        self.active_runs: dict[str, SkillRun] = {}  # session_id -> current run

    def get_status(self, session_id: str) -> str | None:
        """Get a human-readable status string for an active run, or None."""
        run = self.active_runs.get(session_id)
        if not run:
            return None
        completed = len(run.step_results)
        elapsed = (datetime.now(timezone.utc) - run.started_at).total_seconds()
        lines = [f"**{run.skill_name}** — {run.status} ({elapsed:.0f}s elapsed)"]
        for r in run.step_results:
            icon = "✅" if r.status == "success" else "❌" if r.status == "failed" else "⏭"
            lines.append(f"  {icon} {r.step_id} ({r.duration_ms}ms)")
        if run.current_step:
            lines.append(f"  ⏳ {run.current_step} (running...)")
        remaining = run.total_steps - completed - (1 if run.current_step else 0)
        if remaining > 0:
            lines.append(f"  ⬚ {remaining} step(s) remaining")
        return "\n".join(lines)

    async def execute(
        self,
        skill: Skill,
        task_context: str,
        subagent_runners: "dict[str, SubagentRunner]",
        session_id: str = "",
        triggered_by: str = "",
    ) -> SkillRun:
        """Execute a skill and return the run record.

        Progress is tracked on the SkillRun object (queryable via get_status)
        rather than pushed to the chat.
        """
        run = SkillRun(
            skill_name=skill.name,
            session_id=session_id,
            triggered_by=triggered_by,
            total_steps=len(skill.steps),
        )
        self.active_runs[session_id] = run

        try:
            return await self._execute_inner(
                skill, task_context, subagent_runners, run
            )
        finally:
            self.active_runs.pop(session_id, None)

    async def _execute_inner(
        self,
        skill: Skill,
        task_context: str,
        subagent_runners: "dict[str, SubagentRunner]",
        run: SkillRun,
    ) -> SkillRun:
        if not skill.has_steps:
            default_runner = subagent_runners.get(skill.subagent)
            if not default_runner:
                run.status = "failed"
                run.error = f"Subagent '{skill.subagent}' not found"
                run.completed_at = datetime.now(timezone.utc)
                self.registry.record_run(run)
                return run
            return await self._execute_unstructured(
                skill, task_context, default_runner, run
            )

        total = len(skill.steps)
        prior_outputs: list[str] = []

        for i, step in enumerate(skill.steps, 1):
            subagent_name = step.subagent or skill.subagent
            subagent_runner = subagent_runners.get(subagent_name)

            if not subagent_runner:
                result = StepResult(
                    step_id=step.id,
                    status="failed",
                    error=f"Subagent '{subagent_name}' not found",
                )
                run.step_results.append(result)
                run.status = "failed"
                run.error = result.error
                break

            run.current_step = step.id
            log.info("skill_step_starting",
                     skill=run.skill_name, step=step.id,
                     index=f"{i}/{total}", subagent=subagent_name)

            start = time.monotonic()

            try:
                if step.type == "shell":
                    result = await self._run_shell_step(
                        step, subagent_runner, prior_outputs, task_context
                    )
                elif step.type == "agent":
                    result = await self._run_agent_step(
                        step, skill, subagent_runner, prior_outputs, task_context
                    )
                else:
                    result = StepResult(
                        step_id=step.id,
                        status="skipped",
                        output=f"Unknown step type: {step.type}",
                    )
            except asyncio.TimeoutError:
                result = StepResult(step_id=step.id, status="failed", error="Timed out")
            except Exception as e:
                result = StepResult(step_id=step.id, status="failed", error=str(e))

            result.duration_ms = int((time.monotonic() - start) * 1000)
            run.step_results.append(result)
            run.current_step = ""

            log.info("skill_step_done",
                     skill=run.skill_name, step=step.id,
                     status=result.status, duration_ms=result.duration_ms)

            prior_outputs.append(
                f'Step "{step.id}" [{subagent_name}]: {result.output[:1000] if result.output else result.error}'
            )

            if result.status == "failed":
                if step.on_failure == "abort":
                    run.status = "failed"
                    run.error = f"Step '{step.id}' failed: {result.error}"
                    break
                elif step.on_failure == "retry":
                    log.info("skill_step_retrying", skill=run.skill_name, step=step.id)
                    run.current_step = step.id
                    try:
                        if step.type == "shell":
                            result = await self._run_shell_step(
                                step, subagent_runner, prior_outputs, task_context
                            )
                        elif step.type == "agent":
                            result = await self._run_agent_step(
                                step, skill, subagent_runner,
                                prior_outputs, task_context,
                            )
                        result.duration_ms = int((time.monotonic() - start) * 1000)
                        run.step_results[-1] = result
                        run.current_step = ""
                        if result.status == "failed":
                            run.status = "failed"
                            run.error = (
                                f"Step '{step.id}' failed after retry: {result.error}"
                            )
                            break
                    except Exception as e:
                        run.status = "failed"
                        run.error = f"Step '{step.id}' retry failed: {e}"
                        run.current_step = ""
                        break
        else:
            # All steps ran without crashing. But check if the final step's
            # output indicates nothing was actually accomplished.
            last = run.step_results[-1] if run.step_results else None
            if last and last.output:
                out_lower = last.output.lower()
                no_op_signals = [
                    "nothing verified",
                    "no code changes",
                    "skipping",
                    "all_complete",
                    "nothing was implemented",
                    "no tasks",
                    "0 tasks",
                ]
                if any(sig in out_lower for sig in no_op_signals):
                    run.status = "no_op"
                else:
                    run.status = "success"
            else:
                run.status = "success"

        run.completed_at = datetime.now(timezone.utc)
        self.registry.record_run(run)

        return run

    async def _execute_unstructured(
        self,
        skill: Skill,
        task_context: str,
        subagent_runner: "SubagentRunner",
        run: SkillRun,
    ) -> SkillRun:
        """Execute a skill that has no structured steps (original behavior)."""
        run.current_step = "main"
        start = time.monotonic()
        prompt = (
            f"## Skill: {skill.name}\n\n"
            f"{skill.content}\n\n"
            f"## Task\n{task_context}"
        )

        try:
            result, _imgs = await subagent_runner.run(prompt, context="")
            run.step_results.append(
                StepResult(
                    step_id="main",
                    status="success",
                    output=result,
                    duration_ms=int((time.monotonic() - start) * 1000),
                )
            )
            run.status = "success"
        except Exception as e:
            run.step_results.append(
                StepResult(
                    step_id="main",
                    status="failed",
                    error=str(e),
                    duration_ms=int((time.monotonic() - start) * 1000),
                )
            )
            run.status = "failed"
            run.error = str(e)

        run.completed_at = datetime.now(timezone.utc)
        self.registry.record_run(run)
        return run

    async def _run_shell_step(
        self,
        step: SkillStep,
        subagent_runner: "SubagentRunner",
        prior_outputs: list[str],
        task_context: str,
    ) -> StepResult:
        """Run a shell step by delegating to the subagent with an explicit command.

        We instruct the subagent to run the exact command — no improvisation.
        """
        prompt = (
            f"Execute this exact shell command and report the output. "
            f"Do not modify the command. Run it as-is:\n\n"
            f"```\n{step.command}\n```\n\n"
            f"If it fails, report the error output exactly."
        )

        if prior_outputs:
            context_block = "\n".join(prior_outputs)
            prompt = f"## Prior Step Results\n{context_block}\n\n{prompt}"

        result, _imgs = await subagent_runner.run(prompt, context=task_context)

        # Heuristic: if the subagent reports an error, mark as failed
        lower = result.lower()
        failed = any(
            re.search(pattern, lower)
            for pattern in [
                r"(?:^|\n)\s*error:",          # "error:" at line start, not mid-word
                r"command not found",
                r"permission denied",
                r"exit code [1-9]\b",           # exit code 1-9 (not 0, not 144 matching "1")
            ]
        )

        return StepResult(
            step_id=step.id,
            status="failed" if failed else "success",
            output=result,
        )

    async def _run_agent_step(
        self,
        step: SkillStep,
        skill: Skill,
        subagent_runner: "SubagentRunner",
        prior_outputs: list[str],
        task_context: str,
    ) -> StepResult:
        """Run an agent step — give the subagent a prompt with prior context."""
        prompt = step.prompt

        if prior_outputs:
            context_block = "\n".join(prior_outputs)
            prompt = f"## Prior Step Results\n{context_block}\n\n## Your Task\n{prompt}"

        if skill.content:
            prompt = f"## Skill Reference\n{skill.content}\n\n{prompt}"

        result, _imgs = await subagent_runner.run(prompt, context=task_context)

        # Detect when the step ran but its output indicates nothing was accomplished.
        # Uses "no_op" (not "failed") so on_failure:abort doesn't kill the run —
        # downstream steps can still handle the empty result gracefully.
        status = "success"
        result_lower = result.lower()

        no_op_signals = [
            "nothing verified",
            "no code changes",
            "skipping",
            "all_complete",
            "nothing was implemented",
            "no tasks passed",
            "no verified tasks",
            "0 verified",
        ]
        # "all 6 tasks FAIL" / "none of the tasks passed"
        has_total_failure = (
            ("all" in result_lower or "none" in result_lower)
            and any(w in result_lower for w in ["fail", "reject", "not pass"])
        )
        # Empty verified_tasks block
        has_empty_block = "```verified_tasks\n```" in result or "```verified_tasks\n\n```" in result

        if has_total_failure or has_empty_block or any(sig in result_lower for sig in no_op_signals):
            status = "no_op"

        return StepResult(
            step_id=step.id,
            status=status,
            output=result,
        )

    def format_run_result(self, skill: Skill, run: SkillRun) -> str:
        """Format a skill run into a human-readable summary.

        The final step's output is treated as the skill's deliverable and
        included in full. Earlier steps are shown as status lines only.
        """
        status_icon = {
            "success": "✅", "failed": "❌", "aborted": "⛔", "no_op": "⚠️",
        }.get(run.status, "❓")

        lines = [f"{status_icon} Skill **{skill.name}** — {run.status}"]

        if run.error:
            lines.append(f"\n**Error:** {run.error}")

        if run.step_results:
            lines.append("")
            for i, r in enumerate(run.step_results):
                icon = (
                    "✅"
                    if r.status == "success"
                    else "❌" if r.status == "failed" else "⏭"
                )
                duration = f" ({r.duration_ms}ms)" if r.duration_ms else ""
                is_last = i == len(run.step_results) - 1

                if is_last and r.output and r.status == "success":
                    # Final step — include full output as the deliverable
                    lines.append(f"{icon} **{r.step_id}**{duration}")
                    lines.append("")
                    lines.append(r.output)
                else:
                    lines.append(f"{icon} **{r.step_id}**{duration}")
                    if r.error:
                        lines.append(f"```\n{r.error}\n```")

        return "\n".join(lines)
