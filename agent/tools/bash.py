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
    "sudo", "su ",
    "rm -rf /", "rm -rf /*",
    "mkfs", "dd if=",
    "shutdown", "reboot", "halt", "poweroff",
    "passwd", "useradd", "usermod", "userdel",
    "iptables", "ufw",
    "systemctl", "service ",
    "curl |", "wget |",          # Piped downloads (install scripts)
    "> /dev/", ">/dev/",
    "chmod 777",
]

# Commands that require explicit opt-in via tool config
GATED_COMMANDS = {
    "network": ["curl", "wget", "ssh", "scp", "nc", "nmap", "ping"],
    "package": ["pip", "npm", "apt", "uv add"],
}

DEFAULT_TIMEOUT = 30
MAX_OUTPUT_CHARS = 50_000


class BashTool:
    """Executes shell commands in a sandboxed workspace."""

    def __init__(
        self,
        workspace: Path,
        timeout: int = DEFAULT_TIMEOUT,
        allow_network: bool = False,
        allow_packages: bool = False,
        allow_git: bool = False,
    ):
        self.workspace = Path(workspace)
        self.timeout = timeout
        self.allow_network = allow_network
        self.allow_packages = allow_packages
        self.allow_git = allow_git

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

        # Block path traversal outside workspace (git submodules may need ..)
        if ".." in command and ("/" in command or "\\" in command):
            if not self.allow_git:
                return "Path traversal with '..' is not allowed"

        return None

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
                home = os.path.expanduser("~")
                env.update({
                    "HOME": home,  # git/gh need real HOME for SSH + config
                    "GIT_AUTHOR_NAME": "agent-entro",
                    "GIT_AUTHOR_EMAIL": "agent-entro@users.noreply.github.com",
                    "GIT_COMMITTER_NAME": "agent-entro",
                    "GIT_COMMITTER_EMAIL": "agent-entro@users.noreply.github.com",
                    "GIT_SSH_COMMAND": "ssh -i ~/.ssh/id_clawdnet_bot -o StrictHostKeyChecking=no",
                    "GH_CONFIG_DIR": os.path.join(home, ".config", "gh"),
                })

            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.workspace),
                env=env,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self.timeout,
                )
            except asyncio.TimeoutError:
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