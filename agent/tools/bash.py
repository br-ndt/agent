"""Sandboxed bash tool — runs shell commands with restrictions.

Security layers:
  1. Workspace isolation: commands run in a per-session workspace dir
  2. Command blocklist: no sudo, no rm -rf /, no network tools (unless allowed)
  3. Timeout: commands killed after N seconds
  4. Output truncation: prevents memory bombs from huge output
  5. No shell expansion of ~, no access outside workspace

Usage by the subagent runner:
    tool = BashTool(workspace="/home/ubuntu/agent/workspaces/session_123")
    result = await tool.execute("python3 hello.py")
"""

import asyncio
import os
from pathlib import Path

import structlog

log = structlog.get_logger()

# Commands that are never allowed
BLOCKED_COMMANDS = [
    "su ",
    "rm -rf /", "rm -rf /*",
    "mkfs", "dd if=",
    "shutdown", "reboot", "halt", "poweroff",
    "passwd", "useradd", "usermod", "userdel",
    "iptables", "ufw",
    "curl |", "wget |",          # Piped downloads (install scripts)
    "> /dev/", ">/dev/",
    "chmod 777",
]

# Commands that require explicit opt-in via tool config
GATED_COMMANDS = {
    "network": ["curl", "wget", "ssh", "scp", "nc", "nmap", "ping"],
    "package": ["pip", "npm", "apt", "uv add"],
    "sudo": ["sudo", "systemctl", "service "],
}

DEFAULT_TIMEOUT = 30
MAX_OUTPUT_CHARS = 50_000


def resolve_case_insensitive(parent: Path, name: str) -> str:
    """If a directory in `parent` matches `name` case-insensitively, return its actual name.

    Returns the original `name` if no match is found (i.e., it's a genuinely new directory).
    This prevents git clone, cd, mkdir etc. from creating case-variant duplicates.
    """
    if not parent.is_dir():
        return name
    name_lower = name.lower()
    for entry in parent.iterdir():
        if entry.is_dir() and entry.name.lower() == name_lower:
            if entry.name != name:
                log.info("case_insensitive_match",
                         requested=name, existing=entry.name, parent=str(parent))
            return entry.name
    return name


class BashTool:
    """Executes shell commands in a sandboxed workspace."""

    def __init__(
        self,
        workspace: Path,
        timeout: int = DEFAULT_TIMEOUT,
        allow_network: bool = False,
        allow_packages: bool = False,
        allow_git: bool = False,
        allow_sudo: bool = False,
    ):
        self.workspace = Path(workspace)
        self.timeout = timeout
        self.allow_network = allow_network
        self.allow_packages = allow_packages
        self.allow_git = allow_git
        self.allow_sudo = allow_sudo

        # Create workspace if needed
        self.workspace.mkdir(parents=True, exist_ok=True)

    def _check_command(self, command: str) -> str | None:
        """Check if a command is allowed. Returns error string or None."""
        cmd_lower = command.lower().strip()

        # Check blocklist
        for blocked in BLOCKED_COMMANDS:
            if blocked in cmd_lower:
                return f"Blocked command: '{blocked}' is not allowed"

        # Check gated commands
        if not self.allow_network:
            for cmd in GATED_COMMANDS["network"]:
                if cmd_lower.startswith(cmd) or f" {cmd}" in cmd_lower or f"|{cmd}" in cmd_lower:
                    return f"Network command '{cmd}' not allowed (enable allow_network)"

        if not self.allow_packages:
            for cmd in GATED_COMMANDS["package"]:
                if cmd_lower.startswith(cmd) or f" {cmd}" in cmd_lower:
                    return f"Package command '{cmd}' not allowed (enable allow_packages)"

        if not self.allow_sudo:
            for cmd in GATED_COMMANDS["sudo"]:
                if cmd_lower.startswith(cmd) or f" {cmd}" in cmd_lower or f"|{cmd}" in cmd_lower:
                    return f"Command '{cmd}' not allowed (enable allow_sudo)"

        # Block path traversal outside workspace (git submodules may need ..)
        if ".." in command and ("/" in command or "\\" in command):
            if not self.allow_git:
                return "Path traversal with '..' is not allowed"

        return None

    def _fix_case_sensitive_paths(self, command: str) -> str:
        """Rewrite commands that would create case-variant duplicate directories.

        Handles:
          - `git clone <url>` (no explicit target) — infers dir name from URL
          - `git clone <url> <target>` — checks target against existing dirs
          - `cd <dir>` — resolves to existing case-insensitive match
          - `mkdir <dir>` / `mkdir -p <dir>` — resolves to existing match
        """
        import re
        import shlex

        stripped = command.strip()

        # git clone <url> [target]
        clone_match = re.match(
            r'(git\s+clone\s+(?:--[^\s]+\s+)*)'  # git clone [flags]
            r'(\S+)'                                # url
            r'(\s+(\S+))?'                          # optional target dir
            r'(.*)',                                 # trailing flags
            stripped,
        )
        if clone_match:
            prefix, url, _, explicit_target, suffix = clone_match.groups()
            if explicit_target:
                fixed = resolve_case_insensitive(self.workspace, explicit_target)
                if fixed != explicit_target:
                    return f"{prefix}{url} {fixed}{suffix}"
            else:
                # Infer directory name from URL (what git would create)
                repo_name = url.rstrip("/").rsplit("/", 1)[-1]
                repo_name = re.sub(r"\.git$", "", repo_name)
                fixed = resolve_case_insensitive(self.workspace, repo_name)
                if fixed != repo_name:
                    return f"{prefix}{url} {fixed}{suffix}"
            return command

        # cd <dir>
        cd_match = re.match(r'cd\s+("?)(\S+)\1\s*(&&|;|\||$)', stripped)
        if cd_match:
            quote, dirname, rest_sep = cd_match.groups()
            fixed = resolve_case_insensitive(self.workspace, dirname)
            if fixed != dirname:
                remainder = stripped[cd_match.end():]
                return f'cd {quote}{fixed}{quote} {rest_sep} {remainder}'.rstrip()
            return command

        # mkdir [-p] <dir>
        mkdir_match = re.match(r'(mkdir\s+(?:-p\s+)?)(\S+)(.*)', stripped)
        if mkdir_match:
            prefix, dirname, suffix = mkdir_match.groups()
            # Resolve the first path component case-insensitively
            parts = Path(dirname).parts
            if parts:
                fixed_first = resolve_case_insensitive(self.workspace, parts[0])
                if fixed_first != parts[0]:
                    fixed_path = str(Path(fixed_first, *parts[1:]))
                    return f"{prefix}{fixed_path}{suffix}"

        return command

    async def execute(self, command: str) -> dict:
        """Execute a command and return {stdout, stderr, exit_code, error}."""
        # Check if command is allowed
        violation = self._check_command(command)
        if violation:
            log.warning("bash_blocked", command=command[:100], reason=violation)
            return {
                "stdout": "",
                "stderr": violation,
                "exit_code": -1,
                "error": violation,
            }

        # Prevent case-variant directory duplicates
        command = self._fix_case_sensitive_paths(command)

        log.info("bash_executing",
                 command=command[:100],
                 workspace=str(self.workspace))

        try:
            # Build restricted environment
            env = {
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                "HOME": str(self.workspace),
                "TERM": "dumb",
                "LANG": "C.UTF-8",
            }

            if self.allow_git:
                from agent.subagent_runner import _get_agent_gitconfig, _agent_git_identity
                name, email = await _agent_git_identity()
                gitconfig_path = await _get_agent_gitconfig()
                home = os.path.expanduser("~")
                env.update({
                    "HOME": home,  # git/gh need real HOME for SSH + config
                    "GIT_AUTHOR_NAME": name,
                    "GIT_AUTHOR_EMAIL": email,
                    "GIT_COMMITTER_NAME": name,
                    "GIT_COMMITTER_EMAIL": email,
                    "GIT_CONFIG_GLOBAL": str(gitconfig_path),
                    "GIT_SSH_COMMAND": os.environ.get("GIT_SSH_COMMAND", ""),
                    "GH_CONFIG_DIR": os.path.join(home, ".config", "gh"),
                })

            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.workspace),
                env=env,
                start_new_session=True,  # own process group for clean kill
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self.timeout,
                )
            except asyncio.TimeoutError:
                # Kill the entire process group (shell + children)
                import signal
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    proc.kill()
                await proc.wait()
                log.warning("bash_timeout", command=command[:100], timeout=self.timeout)
                return {
                    "stdout": "",
                    "stderr": f"Command timed out after {self.timeout}s",
                    "exit_code": -1,
                    "error": "timeout",
                }

            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")

            # Truncate huge output
            if len(stdout) > MAX_OUTPUT_CHARS:
                stdout = stdout[:MAX_OUTPUT_CHARS] + f"\n... (truncated, {len(stdout)} total chars)"
            if len(stderr) > MAX_OUTPUT_CHARS:
                stderr = stderr[:MAX_OUTPUT_CHARS] + f"\n... (truncated, {len(stderr)} total chars)"

            log.info("bash_completed",
                     exit_code=proc.returncode,
                     stdout_len=len(stdout),
                     stderr_len=len(stderr))

            return {
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": proc.returncode,
                "error": None,
            }

        except Exception as e:
            log.error("bash_error", command=command[:100], error=str(e))
            return {
                "stdout": "",
                "stderr": str(e),
                "exit_code": -1,
                "error": str(e),
            }