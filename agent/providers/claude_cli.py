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

    has_native_tools = True  # SubagentRunner checks this to choose execution path

    def __init__(self, timeout: int = 600, allowed_tools: list[str] | None = None, disallowed_tools: list[str] | None = None, cwd: str | None = None):
        self.timeout = timeout
        self.allowed_tools = allowed_tools or []
        self.disallowed_tools = disallowed_tools or []
        self.cwd = cwd

    async def complete(self, messages, system="", model="sonnet",
                       max_tokens=4096, temperature=0.7, tools=None,
                       cwd=None, **kwargs) -> LLMResponse:
        cli_model = MODEL_ALIASES.get(model, model)

        # Build base command — stream-json gives us live progress events
        base_cmd = [
            "claude", "-p",
            "--verbose",
            "--output-format", "stream-json",
            "--dangerously-skip-permissions",
            "--model", cli_model,
        ]

        # Per-call overrides take priority over constructor defaults
        allowed = kwargs.get("allowed_tools", self.allowed_tools)
        disallowed = kwargs.get("disallowed_tools", self.disallowed_tools)
        if allowed:
            base_cmd.extend(["--allowedTools", " ".join(allowed)])
        if disallowed:
            base_cmd.extend(["--disallowedTools", " ".join(disallowed)])
        if system:
            base_cmd.extend(["--system-prompt", system])

        cmd = base_cmd

        prompt = _flatten_messages(messages)

        try:
            # Strip ANTHROPIC_API_KEY from the subprocess environment
            # so the CLI uses OAuth (your subscription) instead of the API
            import os
            env = {k: v for k, v in os.environ.items()
                   if k != "ANTHROPIC_API_KEY"}
            # Apply per-call env overrides (e.g., git identity)
            env.update(kwargs.get("env", {}))

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=cwd or self.cwd,
            )

            # Send prompt to stdin, then close it
            proc.stdin.write(prompt.encode("utf-8"))
            await proc.stdin.drain()
            proc.stdin.close()

            # Parse stream-json: one JSON object per line.
            # Event types: "system", "assistant", "result"
            # We log tool use and text chunks as they arrive,
            # and collect the final "result" event.
            #
            # Timeout is per-line (inactivity), not wall-clock. A subagent
            # that's actively streaming tool results can run for hours — it
            # only times out if it goes silent for self.timeout seconds.
            result = None

            while True:
                try:
                    raw_line = await asyncio.wait_for(
                        proc.stdout.readline(), timeout=self.timeout
                    )
                except asyncio.TimeoutError:
                    log.error("claude_cli_inactivity_timeout",
                              timeout=self.timeout,
                              msg="No output for timeout period")
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass
                    raise

                if not raw_line:
                    break  # EOF

                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    log.debug("claude_cli_unparseable_line", line=line[:200])
                    continue

                etype = event.get("type", "")

                if etype == "result":
                    result = event
                elif etype == "assistant":
                    # Tool use or text — log for visibility
                    tool = event.get("tool")
                    if tool:
                        log.info("claude_cli_tool_use",
                                 model=cli_model, tool=tool,
                                 input_preview=str(event.get("input", ""))[:150])
                    else:
                        text = event.get("content", "")
                        if text:
                            log.info("claude_cli_text",
                                     model=cli_model, preview=text[:150])
                elif etype == "system":
                    log.info("claude_cli_system", model=cli_model,
                             data=json.dumps(event, default=str)[:300])
                else:
                    log.info("claude_cli_other", model=cli_model,
                             data=json.dumps(event, default=str)[:300])

            # Wait for process to finish
            await proc.wait()

            if result is None:
                stderr_bytes = await proc.stderr.read()
                stderr = stderr_bytes.decode("utf-8", errors="replace").strip()
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"claude CLI exited with code {proc.returncode}: {stderr[:200]}"
                    )
                raise RuntimeError("claude CLI produced no result event")

            content = result.get("result", "")

            if result.get("is_error"):
                raise RuntimeError(f"Claude CLI returned error: {content}")

            # Extract token usage from modelUsage
            usage = {}
            model_usage = result.get("modelUsage", {})
            if model_usage:
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
            raise RuntimeError(f"claude CLI inactive for {self.timeout}s (no output)")

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