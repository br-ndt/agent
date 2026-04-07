"""Subagent runner — executes a task against a specific provider/model.

Supports two execution paths, chosen automatically:
  1. Native tools (Claude CLI) — Claude uses its own Bash/Write/Edit tools,
     guided by --allowedTools mapped from the SubagentConfig.tools list.
  2. Tool loop (all other providers) — SubagentRunner prompts the LLM with
     a JSON tool-call format and executes tools in Python.

Both paths write to the same per-subagent workspace, so callers see a
uniform interface regardless of provider.
"""

import asyncio
import json
import os
import time
from pathlib import Path

import structlog

from agent.config import SubagentConfig
from agent.cost_tracker import CostTracker
from agent.providers.base import BaseProvider
from agent.tools.bash import BashTool
from agent.tools.file_ops import FileOpsTool
from agent.tools.web_fetch import WebFetchTool
from agent.tools.web_browser import WebBrowserTool

log = structlog.get_logger()

MAX_TURNS = 15
BASE_DIR = Path(__file__).resolve().parent.parent
WORKSPACES_DIR = BASE_DIR / "workspaces"


async def _agent_git_identity() -> tuple[str, str]:
    """Return (name, email) for the agent's git identity.

    Reads from GIT_AGENT_NAME / GIT_AGENT_EMAIL env vars.
    Falls back to the system git config if unset.
    """
    name = os.environ.get("GIT_AGENT_NAME", "")
    email = os.environ.get("GIT_AGENT_EMAIL", "")

    if not name:
        try:
            proc = await asyncio.create_subprocess_exec(
                "git", "config", "--global", "user.name",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            if proc.returncode == 0:
                name = stdout.decode().strip()
        except Exception:
            pass
        if not name:
            name = "agent"

    if not email:
        try:
            proc = await asyncio.create_subprocess_exec(
                "git", "config", "--global", "user.email",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            if proc.returncode == 0:
                email = stdout.decode().strip()
        except Exception:
            pass
        if not email:
            email = "agent@localhost"

    return name, email

# Maps SubagentConfig tool names → Claude Code native tool names
NATIVE_TOOL_MAP: dict[str, list[str]] = {
    "bash": ["Bash"],
    "file_ops": ["Read", "Write", "Edit", "Glob", "Grep", "ListDir"],
    "web_fetch": ["WebFetch"],
    "web_browser": ["WebFetch", "WebSearch"],
    "git": ["Bash"],  # git/gh commands run via Bash
}


def _has_native_tools(provider: BaseProvider) -> bool:
    """Check whether a provider handles tools natively (e.g. Claude CLI)."""
    return getattr(provider, "has_native_tools", False)


def _resolve_native_allowed(tools: list[str]) -> list[str]:
    """Convert SubagentConfig.tools to Claude Code --allowedTools names."""
    native: list[str] = []
    for t in tools:
        native.extend(NATIVE_TOOL_MAP.get(t, []))
    return sorted(set(native))


class SubagentRunner:
    """Runs a subagent to completion with tools.

    Accepts an optional fallback_provider so that when the primary provider
    fails (rate-limit, timeout, etc.) the runner retries with a different
    model — and automatically picks the right execution path for it.
    """

    def __init__(
        self,
        config: SubagentConfig,
        provider: BaseProvider,
        fallback_provider: BaseProvider | None = None,
        fallback_model: str = "",
        cost_tracker: CostTracker | None = None,
    ):
        self.config = config
        self.provider = provider
        self.fallback_provider = fallback_provider
        self.fallback_model = fallback_model
        self.cost_tracker = cost_tracker

    # ── Public entry point ────────────────────────────────────

    async def run(self, task: str, context: str = "", session_id: str = "") -> str:
        """Execute a task, routing to the right tool path per provider."""
        prompt = task
        if context:
            prompt = f"Context:\n{context}\n\nTask:\n{task}"

        has_tools = bool(self.config.tools)
        workspace = None
        if has_tools:
            if self.config.workspace_dir:
                workspace = (BASE_DIR / self.config.workspace_dir).resolve()
            else:
                workspace = WORKSPACES_DIR / self.config.name
            workspace.mkdir(parents=True, exist_ok=True)
            if "git" in self.config.tools:
                await _ensure_git_identity(workspace)

        log.info(
            "subagent_starting",
            agent=self.config.name,
            model=self.config.model,
            has_tools=has_tools,
            task_len=len(task),
        )

        try:
            return await self._run_with(
                self.provider, self.config.model, prompt, workspace, has_tools
            )
        except Exception as e:
            if not self.fallback_provider:
                raise
            log.warning(
                "subagent_primary_failed",
                agent=self.config.name,
                error=str(e),
                fallback_model=self.fallback_model,
            )
            return await self._run_with(
                self.fallback_provider, self.fallback_model, prompt, workspace, has_tools
            )

    # ── Dispatch ──────────────────────────────────────────────

    async def _run_with(
        self,
        provider: BaseProvider,
        model: str,
        prompt: str,
        workspace: Path | None,
        has_tools: bool,
    ) -> str:
        if has_tools and _has_native_tools(provider):
            return await self._run_native(provider, model, prompt, workspace)
        elif has_tools:
            return await self._run_tool_loop(provider, model, prompt, workspace)
        else:
            return await self._run_text_only(provider, model, prompt, workspace)

    # ── Path 1: Native tools (Claude CLI) ─────────────────────

    async def _run_native(
        self,
        provider: BaseProvider,
        model: str,
        prompt: str,
        workspace: Path,
    ) -> str:
        """Single call — Claude handles tools natively in the workspace."""
        native_allowed = _resolve_native_allowed(self.config.tools)

        # Claude Code sandboxes to cwd, so symlink sibling workspaces
        # into a _ref/ directory so the agent can read them.
        _link_sibling_workspaces(workspace, WORKSPACES_DIR, self.config.name)

        workspace_hint = (
            f"Your workspace is: {workspace}\n"
            f"Write all output files here (not in _ref/).\n"
        )
        if self.config.read_root:
            read_root = (BASE_DIR / self.config.read_root).resolve()
            workspace_hint += f"You can also read files under: {read_root}\n"
        workspace_hint += _ref_listing(workspace)

        log.info(
            "subagent_native_tools",
            agent=self.config.name,
            allowed=native_allowed,
            workspace=str(workspace),
        )

        # Pass git identity env if git tools are enabled
        extra_env = {}
        if "git" in self.config.tools:
            name, email = await _agent_git_identity()
            extra_env = {
                "GIT_AUTHOR_NAME": name,
                "GIT_AUTHOR_EMAIL": email,
                "GIT_COMMITTER_NAME": name,
                "GIT_COMMITTER_EMAIL": email,
                "GIT_CONFIG_GLOBAL": str(await _get_agent_gitconfig()),
                "GIT_SSH_COMMAND": os.environ.get("GIT_SSH_COMMAND", ""),
                "GH_CONFIG_DIR": os.path.join(os.path.expanduser("~"), ".config", "gh"),
            }
            # Claude Code's Bash tool doesn't propagate env vars, so
            # instruct the agent to set repo-local identity after git init.
            workspace_hint += (
                f"\nGit identity: ALWAYS run these commands immediately after "
                f"any `git init` and before any commits:\n"
                f"  git config user.name \"{name}\"\n"
                f"  git config user.email \"{email}\"\n"
                f"Never use the system git identity.\n"
            )

        start = time.monotonic()
        response = await provider.complete(
            messages=[{"role": "user", "content": workspace_hint + "\n" + prompt}],
            system=self.config.personality,
            model=model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            cwd=str(workspace),
            allowed_tools=native_allowed,
            env=extra_env,
        )
        elapsed_ms = int((time.monotonic() - start) * 1000)

        if self.cost_tracker:
            await self.cost_tracker.log_call(
                provider="claude_cli",
                model=model,
                usage=response.usage,
                agent=self.config.name,
                duration_ms=elapsed_ms,
            )

        log.info(
            "subagent_response",
            agent=self.config.name,
            content_len=len(response.content),
            duration_ms=elapsed_ms,
        )
        return response.content

    # ── Path 2: Custom tool loop (Gemini, OpenAI, etc.) ───────

    async def _run_tool_loop(
        self,
        provider: BaseProvider,
        model: str,
        prompt: str,
        workspace: Path,
    ) -> str:
        """Multi-turn loop — LLM emits JSON tool calls, we execute them."""
        bash_tool = None
        file_tool = None
        web_tool = None
        browser_tool = None

        if "bash" in self.config.tools:
            bash_tool = BashTool(
                workspace=workspace,
                allow_network="network" in self.config.tools or "git" in self.config.tools,
                allow_packages="packages" in self.config.tools,
                allow_git="git" in self.config.tools,
            )
        if "file_ops" in self.config.tools:
            if self.config.read_root:
                read_root = (BASE_DIR / self.config.read_root).resolve()
            else:
                read_root = WORKSPACES_DIR
            file_tool = FileOpsTool(workspace=workspace, read_root=read_root)
        if "web_fetch" in self.config.tools:
            web_tool = WebFetchTool()
        if "web_browser" in self.config.tools:
            browser_tool = WebBrowserTool(screenshot_dir=workspace / "_screenshots")

        # Build tool preamble
        available = []
        if browser_tool:
            available.append(
                "web_browser(url) — fetch content via full browser (handles JS, dynamic content, SPA)"
            )
        if web_tool:
            available.append(
                "web_fetch(url) — fast, simple HTML fetch (use for basic articles/blogs)"
            )
        if bash_tool:
            available.append("bash(command) — run shell commands")
        if file_tool:
            available.append(
                "read_file(path), write_file(path, content), edit_file(path, old, new), list_files(path)"
            )

        # Copy sibling files into _ref/ so paths are consistent whether
        # this step runs natively (Claude) or via tool loop (Gemini fallback).
        _link_sibling_workspaces(workspace, WORKSPACES_DIR, self.config.name)
        ref_listing = _ref_listing(workspace)

        tool_preamble = (
            "## Tool Use Instructions\n"
            "You have access to tools in your workspace. To use a tool, you MUST respond with "
            "a JSON block like this and NOTHING ELSE in that message:\n"
            '```tool\n{"tool": "tool_name", "args": {"key": "value"}}\n```\n\n'
            "Available tools:\n"
            + "\n".join(f"- {t}" for t in available)
            + f"\n\nYour workspace is: {workspace}\n"
            + ref_listing
            + "Write all output files to your workspace root (not in _ref/).\n"
            "IMPORTANT: For modern web apps (React, Vue, etc.), ALWAYS prefer `web_browser`. "
            "After each tool call, you will see the result. When you have enough information to "
            "answer the user, respond normally without any tool blocks.\n\n"
            "TASK:\n"
        )

        system = self.config.personality
        messages = [{"role": "user", "content": tool_preamble + prompt}]
        response = None

        for turn in range(MAX_TURNS):
            start = time.monotonic()
            response = await provider.complete(
                messages=messages,
                system=system,
                model=model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                cwd=str(workspace),
            )
            elapsed_ms = int((time.monotonic() - start) * 1000)

            if self.cost_tracker:
                await self.cost_tracker.log_call(
                    provider=self.config.resolved_provider,
                    model=model,
                    usage=response.usage,
                    agent=self.config.name,
                    duration_ms=elapsed_ms,
                )

            log.info("subagent_turn", agent=self.config.name, turn=turn)

            tool_call = _extract_tool_call(response.content)
            if not tool_call:
                log.info(
                    "subagent_response",
                    agent=self.config.name,
                    content_len=len(response.content),
                )
                return response.content

            tool_name = tool_call.get("tool", "")
            args = tool_call.get("args", {})
            log.info(
                "subagent_tool_call",
                agent=self.config.name,
                tool=tool_name,
                args_keys=list(args.keys()),
            )

            result = await _execute_tool(
                tool_name, args, bash_tool, file_tool, web_tool, browser_tool
            )

            messages.append({"role": "assistant", "content": response.content})
            tool_msg = {
                "role": "user",
                "content": f"Tool result for `{tool_name}`:\n```\n{json.dumps(result, indent=2)}\n```",
            }
            # Attach screenshot as image if present (multimodal providers can see it)
            if "screenshot" in result:
                tool_msg["images"] = [result["screenshot"]]
            messages.append(tool_msg)

        log.warning("subagent_max_turns", agent=self.config.name, turns=MAX_TURNS)
        return response.content if response else ""

    # ── Path 3: Text only (no tools) ─────────────────────────

    async def _run_text_only(
        self,
        provider: BaseProvider,
        model: str,
        prompt: str,
        workspace: Path | None,
    ) -> str:
        start = time.monotonic()
        response = await provider.complete(
            messages=[{"role": "user", "content": prompt}],
            system=self.config.personality,
            model=model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            cwd=str(workspace) if workspace else None,
        )
        elapsed_ms = int((time.monotonic() - start) * 1000)

        if self.cost_tracker:
            await self.cost_tracker.log_call(
                provider=self.config.resolved_provider,
                model=model,
                usage=response.usage,
                agent=self.config.name,
                duration_ms=elapsed_ms,
            )

        log.info(
            "subagent_response",
            agent=self.config.name,
            content_len=len(response.content),
            duration_ms=elapsed_ms,
        )
        return response.content


# ── Workspace helpers ─────────────────────────────────────────


async def _get_agent_gitconfig() -> Path:
    """Return path to a shared gitconfig file for agent identity.

    This file is pointed to by GIT_CONFIG_GLOBAL in the subagent env,
    ensuring all git operations (including those spawned by Claude Code's
    internal Bash tool, which doesn't propagate env vars) use the agent
    identity regardless of when repos are cloned or initialized.

    Regenerated each call so env var changes take effect without restart.
    """
    name, email = await _agent_git_identity()
    config_path = BASE_DIR / "state" / "agent-gitconfig"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        f"[user]\n"
        f"    name = {name}\n"
        f"    email = {email}\n"
    )
    return config_path


async def _ensure_git_identity(workspace: Path):
    """Set repo-local git identity in any git repo under the workspace.

    Claude Code doesn't propagate env vars to its Bash tool subprocesses,
    so GIT_AUTHOR_NAME etc. get lost. We set repo-local config as a
    belt-and-suspenders measure alongside GIT_CONFIG_GLOBAL.
    """
    name, email = await _agent_git_identity()

    # Find git repos: workspace itself and any depth below
    for repo in workspace.rglob(".git"):
        repo_dir = repo.parent
        try:
            for key, val in [("user.name", name), ("user.email", email)]:
                proc = await asyncio.create_subprocess_exec(
                    "git", "config", key, val,
                    cwd=str(repo_dir),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await proc.communicate()
            log.debug("git_identity_set", repo=str(repo_dir))
        except Exception as e:
            log.warning("git_identity_failed", repo=str(repo_dir), error=str(e))


def _link_sibling_workspaces(workspace: Path, workspaces_dir: Path, self_name: str):
    """Copy sibling workspace files into workspace/_ref/ so Claude Code can read them.

    Claude Code resolves symlinks and blocks reads outside cwd,
    so we copy instead of symlinking.
    """
    import shutil

    ref_dir = workspace / "_ref"
    # Wipe and recreate to stay in sync
    if ref_dir.exists():
        shutil.rmtree(ref_dir)
    ref_dir.mkdir()

    if not workspaces_dir.exists():
        return

    for sib in workspaces_dir.iterdir():
        if not sib.is_dir() or sib.name == self_name:
            continue
        # Only copy actual files, skip _ref dirs and hidden files
        dest = ref_dir / sib.name
        files = [
            f for f in sib.rglob("*")
            if f.is_file() and not f.name.startswith(".") and "_ref" not in f.parts
        ]
        if not files:
            continue
        for f in files:
            rel = f.relative_to(sib)
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)


def _ref_listing(workspace: Path) -> str:
    """List files in workspace/_ref/ for the agent's context."""
    ref_dir = workspace / "_ref"
    if not ref_dir.exists():
        return ""

    lines = ["Reference files from other subagents (read-only, in _ref/):"]
    for sib in sorted(ref_dir.iterdir()):
        if not sib.is_dir():
            continue
        files = sorted(
            f for f in sib.rglob("*")
            if f.is_file() and not f.name.startswith(".")
        )
        if not files:
            continue
        lines.append(f"  _ref/{sib.name}/")
        for f in files[:20]:
            rel = f.relative_to(sib)
            lines.append(f"    {rel} ({f.stat().st_size} bytes)")
        if len(files) > 20:
            lines.append(f"    ... and {len(files) - 20} more files")

    if len(lines) == 1:
        return ""
    return "\n".join(lines) + "\n"


def _sibling_workspace_listing(workspaces_dir: Path, self_name: str) -> str:
    """Build a concise listing of sibling workspaces and their files."""
    if not workspaces_dir.exists():
        return ""

    siblings = [
        d for d in sorted(workspaces_dir.iterdir())
        if d.is_dir() and d.name != self_name
    ]
    if not siblings:
        return ""

    lines = ["Other subagent workspaces (read-only via ../<name>/ paths):"]
    for sib in siblings:
        files = sorted(
            f for f in sib.rglob("*")
            if f.is_file() and not f.name.startswith(".")
        )
        if not files:
            continue
        lines.append(f"  ../{sib.name}/")
        for f in files[:20]:  # cap to avoid bloating the prompt
            rel = f.relative_to(sib)
            size = f.stat().st_size
            lines.append(f"    {rel} ({size} bytes)")
        if len(files) > 20:
            lines.append(f"    ... and {len(files) - 20} more files")

    if len(lines) == 1:
        return ""  # no siblings with files

    return "\n".join(lines) + "\n"


# ── Tool helpers ──────────────────────────────────────────────


async def _execute_tool(
    tool_name: str,
    args: dict,
    bash_tool: BashTool | None,
    file_tool: FileOpsTool | None,
    web_tool: WebFetchTool | None,
    browser_tool: WebBrowserTool | None,
) -> dict:
    """Execute a tool call and return the result."""
    try:
        if tool_name == "bash" and bash_tool:
            return await bash_tool.execute(args.get("command", ""))
        elif tool_name == "read_file" and file_tool:
            return await file_tool.read(args.get("path", ""))
        elif tool_name == "write_file" and file_tool:
            return await file_tool.write(
                args.get("path", ""), args.get("content", "")
            )
        elif tool_name == "edit_file" and file_tool:
            return await file_tool.edit(
                args.get("path", ""), args.get("old", ""), args.get("new", "")
            )
        elif tool_name == "list_files" and file_tool:
            return await file_tool.list_files(args.get("path", "."))
        elif tool_name == "web_fetch" and web_tool:
            return await web_tool.fetch(args.get("url", ""))
        elif tool_name == "web_browser" and browser_tool:
            return await browser_tool.fetch(args.get("url", ""))
        else:
            return {"error": f"Unknown or unavailable tool: {tool_name}"}
    except Exception as e:
        log.error("tool_execution_error", tool=tool_name, error=str(e))
        return {"error": str(e)}


def _extract_tool_call(text: str) -> dict | None:
    """Extract a tool call JSON block from the response text.

    Looks for ```tool ... ``` or ```json ... ``` blocks containing
    a {"tool": "...", "args": {...}} structure.
    """
    import re

    patterns = [
        r"```tool\s*\n(.*?)\n```",
        r"```json\s*\n(.*?)\n```",
        r'```\s*\n(\{.*?"tool".*?\})\n```',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1).strip())
                if "tool" in parsed:
                    return parsed
            except json.JSONDecodeError:
                continue

    match = re.search(
        r'\{[^{}]*"tool"\s*:\s*"[^"]+"\s*,\s*"args"\s*:\s*\{[^}]*\}[^}]*\}', text
    )
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None
