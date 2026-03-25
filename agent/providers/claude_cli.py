"""Claude CLI provider — uses your Pro/Max subscription via OAuth.

Instead of calling the Anthropic API (which costs per-token), this provider
shells out to the `claude` CLI with `-p` (print/non-interactive mode).
The CLI uses your OAuth login from `claude login`, so usage comes from
your Claude Pro ($20/mo) or Max ($100/mo) subscription.

Prerequisites:
    npm install -g @anthropic-ai/claude-code
    claude login   # Opens browser for OAuth

Usage in config.yaml:
    orchestrator:
      provider: claude_cli
      model: sonnet    # or opus, haiku

    subagents:
      coder:
        provider: claude_cli
        model: opus
"""

import asyncio
import json

import structlog

from .base import BaseProvider, LLMResponse

log = structlog.get_logger()

# Maps long model names to CLI short names
MODEL_ALIASES = {
    "opus": "opus",
    "sonnet": "sonnet",
    "haiku": "haiku",
    "claude-opus-4-6": "opus",
    "claude-sonnet-4-6": "sonnet",
    "claude-haiku-4-5": "haiku",
}


class ClaudeCLIProvider(BaseProvider):
    """Calls Claude via the CLI subprocess using your subscription."""

    def __init__(self, timeout: int = 300, allowed_tools: list[str] | None = None, disallowed_tools: list[str] | None = None, cwd: str | None = None):
        self.timeout = timeout
        self.allowed_tools = allowed_tools or []
        self.disallowed_tools = disallowed_tools or []
        self.cwd = cwd

    async def complete(
        self,
        messages: list[dict],
        system: str = "",
        model: str = "sonnet",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        tools: list[dict] | None = None,
        cwd: str | None = None
    ) -> LLMResponse:
        cli_model = MODEL_ALIASES.get(model, model)

        cmd = [
            "claude",
            "-p",
            "--output-format", "json",
            "--model", cli_model,
        ]

        if self.allowed_tools:
            cmd.extend(["--allowedTools", " ".join(self.allowed_tools)])
        if self.disallowed_tools:
            cmd.extend(["--disallowedTools", " ".join(self.disallowed_tools)])

        if system:
            cmd.extend(["--system-prompt", system])

        prompt = _flatten_messages(messages)

        try:
            # Strip ANTHROPIC_API_KEY from the subprocess environment
            # so the CLI uses OAuth (your subscription) instead of the API
            import os
            env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=cwd or self.cwd,
            )

            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(input=prompt.encode("utf-8")),
                timeout=self.timeout,
            )

            stdout = stdout_bytes.decode("utf-8", errors="replace").strip()
            stderr = stderr_bytes.decode("utf-8", errors="replace").strip()

            if proc.returncode != 0:
                log.error(
                    "claude_cli_error",
                    returncode=proc.returncode,
                    stderr=stderr[:500],
                    stdout=stdout[:500],
                )
                raise RuntimeError(
                    f"claude CLI exited with code {proc.returncode}: "
                    f"{stderr[:200] or stdout[:200]}"
                )

            result = json.loads(stdout)

            # Response format:
            # {
            #   "type": "result",
            #   "result": "Hello!",           <-- the actual text
            #   "is_error": false,
            #   "total_cost_usd": 0.008,
            #   "usage": { "input_tokens": ..., "output_tokens": ... },
            #   "modelUsage": { "claude-sonnet-4-6": { ... } },
            #   ...
            # }
            content = result.get("result", "")

            if result.get("is_error"):
                raise RuntimeError(f"Claude CLI returned error: {content}")

            # Extract token usage from modelUsage (more accurate)
            usage = {}
            model_usage = result.get("modelUsage", {})
            if model_usage:
                # modelUsage is keyed by the full model name
                first_model = next(iter(model_usage.values()), {})
                usage = {
                    "input_tokens": first_model.get("inputTokens", 0),
                    "output_tokens": first_model.get("outputTokens", 0),
                    "cache_read_tokens": first_model.get("cacheReadInputTokens", 0),
                    "cost_usd": first_model.get("costUSD", 0),
                }
            elif "total_cost_usd" in result:
                usage["cost_usd"] = result["total_cost_usd"]

            log.info(
                "claude_cli_response",
                model=cli_model,
                content_len=len(content),
                cost=usage.get("cost_usd", 0),
                duration_ms=result.get("duration_ms", 0),
            )

            return LLMResponse(
                content=content,
                model=f"claude-cli:{cli_model}",
                usage=usage,
            )

        except asyncio.TimeoutError:
            log.error("claude_cli_timeout", timeout=self.timeout)
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            raise RuntimeError(f"claude CLI timed out after {self.timeout}s")

        except json.JSONDecodeError as e:
            log.error("claude_cli_json_error", stdout=stdout[:500], error=str(e))
            if "login" in stdout.lower() or "auth" in stdout.lower():
                raise RuntimeError(
                    "Claude CLI not authenticated. Run: claude login"
                )
            return LLMResponse(content=stdout, model=f"claude-cli:{cli_model}", usage={})


def _flatten_messages(messages: list[dict]) -> str:
    """Convert message list into a single prompt string for the CLI."""
    if len(messages) == 1:
        return messages[0]["content"]

    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
        elif role == "system":
            parts.append(f"[System: {content}]")

    parts.append("Assistant:")
    return "\n\n".join(parts)