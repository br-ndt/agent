"""Subagent runner — executes a task against a specific provider/model.

Now supports a tool loop: the subagent can call bash and file_ops tools,
see the results, and iterate until the task is done.
"""

import json
from pathlib import Path

import structlog

from agent.config import SubagentConfig
from agent.providers.base import BaseProvider
from agent.tools.bash import BashTool
from agent.tools.file_ops import FileOpsTool

log = structlog.get_logger()

MAX_TURNS = 15
WORKSPACES_DIR = Path(__file__).resolve().parent.parent / "workspaces"

# Tool definitions in the format LLMs understand
TOOL_DEFINITIONS = {
    "bash": {
        "name": "bash",
        "description": "Run a shell command in the workspace. Returns stdout, stderr, and exit code. Commands run in an isolated workspace directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute",
                }
            },
            "required": ["command"],
        },
    },
    "read_file": {
        "name": "read_file",
        "description": "Read the contents of a file in the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to the workspace",
                }
            },
            "required": ["path"],
        },
    },
    "write_file": {
        "name": "write_file",
        "description": "Write content to a file in the workspace. Creates parent directories as needed.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to the workspace",
                },
                "content": {
                    "type": "string",
                    "description": "File contents to write",
                },
            },
            "required": ["path", "content"],
        },
    },
    "edit_file": {
        "name": "edit_file",
        "description": "Replace a string in a file (first occurrence).",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"},
                "old": {"type": "string", "description": "String to find"},
                "new": {"type": "string", "description": "Replacement string"},
            },
            "required": ["path", "old", "new"],
        },
    },
    "list_files": {
        "name": "list_files",
        "description": "List files and directories in the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path relative to workspace (default: '.')",
                    "default": ".",
                }
            },
        },
    },
}


class SubagentRunner:
    """Runs a subagent to completion with tools."""

    def __init__(self, config: SubagentConfig, provider: BaseProvider):
        self.config = config
        self.provider = provider

    async def run(self, task: str, context: str = "", session_id: str = "") -> str:
        """Execute a task, with tool loop if tools are configured."""
        prompt = task
        if context:
            prompt = f"Context:\n{context}\n\nTask:\n{task}"

        # Set up tools if this agent has any
        has_tools = bool(self.config.tools)
        bash_tool = None
        file_tool = None
        workspace = None

        if has_tools:
            # Create a workspace for this session
            ws_name = session_id.replace(":", "_") if session_id else "default"
            workspace = WORKSPACES_DIR / self.config.name / ws_name
            workspace.mkdir(parents=True, exist_ok=True)

            if "bash" in self.config.tools:
                bash_tool = BashTool(
                    workspace=workspace,
                    allow_network="network" in self.config.tools,
                    allow_packages="packages" in self.config.tools,
                )
            if "file_ops" in self.config.tools:
                file_tool = FileOpsTool(workspace=workspace)

        # Build tool instruction block for the USER message (not system)
        # This ensures the LLM sees it regardless of provider system prompt behavior
        if has_tools:
            available = []
            if bash_tool:
                available.append("bash(command) — run shell commands")
            if file_tool:
                available.append("read_file(path), write_file(path, content), edit_file(path, old, new), list_files(path)")

            tool_preamble = (
                "You have access to tools in your workspace. To use a tool, respond with ONLY a JSON block like this:\n"
                '```tool\n{"tool": "tool_name", "args": {"key": "value"}}\n```\n\n'
                "Available tools:\n" + "\n".join(f"- {t}" for t in available) +
                f"\n\nYour workspace is: {workspace}\n"
                "Use tools to complete the task. After each tool call, you'll see the result. "
                "When done, respond normally without any tool blocks.\n\n"
                "TASK:\n"
            )
            prompt = tool_preamble + prompt

        system = self.config.personality
        messages = [{"role": "user", "content": prompt}]

        log.info(
            "subagent_starting",
            agent=self.config.name,
            model=self.config.model,
            has_tools=has_tools,
            task_len=len(task),
        )

        for turn in range(MAX_TURNS):
            response = await self.provider.complete(
                messages=messages,
                system=system,
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                cwd=str(workspace) if workspace else None,
            )

            log.info("subagent_turn", agent=self.config.name, turn=turn)

            # If no tools, or response doesn't contain tool calls, we're done
            if not has_tools:
                return response.content

            tool_call = _extract_tool_call(response.content)
            if not tool_call:
                # No tool call in response — agent is done
                return response.content

            # Execute the tool
            tool_name = tool_call.get("tool", "")
            args = tool_call.get("args", {})

            log.info("subagent_tool_call",
                     agent=self.config.name,
                     tool=tool_name,
                     args_keys=list(args.keys()))

            result = await self._execute_tool(
                tool_name, args, bash_tool, file_tool
            )

            # Add the assistant response and tool result to history
            messages.append({"role": "assistant", "content": response.content})
            messages.append({
                "role": "user",
                "content": f"Tool result for `{tool_name}`:\n```\n{json.dumps(result, indent=2)}\n```",
            })

        # Max turns reached
        log.warning("subagent_max_turns", agent=self.config.name, turns=MAX_TURNS)
        return response.content

    async def _execute_tool(
        self,
        tool_name: str,
        args: dict,
        bash_tool: BashTool | None,
        file_tool: FileOpsTool | None,
    ) -> dict:
        """Execute a tool call and return the result."""
        try:
            if tool_name == "bash" and bash_tool:
                return await bash_tool.execute(args.get("command", ""))
            elif tool_name == "read_file" and file_tool:
                return await file_tool.read(args.get("path", ""))
            elif tool_name == "write_file" and file_tool:
                return await file_tool.write(args.get("path", ""), args.get("content", ""))
            elif tool_name == "edit_file" and file_tool:
                return await file_tool.edit(
                    args.get("path", ""), args.get("old", ""), args.get("new", "")
                )
            elif tool_name == "list_files" and file_tool:
                return await file_tool.list_files(args.get("path", "."))
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

    # Try ```tool ... ``` blocks first
    patterns = [
        r'```tool\s*\n(.*?)\n```',
        r'```json\s*\n(.*?)\n```',
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

    # Try bare JSON with "tool" key
    match = re.search(r'\{[^{}]*"tool"\s*:\s*"[^"]+"\s*,\s*"args"\s*:\s*\{[^}]*\}[^}]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None