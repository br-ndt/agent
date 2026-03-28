"""Provider registry — creates provider instances from config."""

import shutil

import structlog

from agent.config import Config
from .base import BaseProvider
from .resilient import ResilientProvider

log = structlog.get_logger()


def build_providers(cfg: Config) -> dict[str, BaseProvider]:
    """Build a dict of provider_name -> provider_instance from config."""
    providers: dict[str, BaseProvider] = {}

    # Claude CLI provider — uses your Pro/Max subscription (no API key needed).
    # Native tools are enabled; SubagentRunner maps its tool config to
    # Claude Code's native tool names via --allowedTools per call.
    if shutil.which("claude"):
        from .claude_cli import ClaudeCLIProvider

        providers["claude_cli"] = ClaudeCLIProvider()
    else:
        log.debug("claude_cli_not_available",
                  hint="Install: npm install -g @anthropic-ai/claude-code && claude login")

    # Anthropic API provider — pay-per-token
    if cfg.anthropic_api_key:
        from .anthropic import AnthropicProvider
        providers["anthropic"] = AnthropicProvider(cfg.anthropic_api_key)

    if cfg.openai_api_key:
        from .openai import OpenAIProvider
        providers["openai"] = OpenAIProvider(cfg.openai_api_key)

    if cfg.google_api_key:
        from .google import GoogleProvider
        providers["google"] = GoogleProvider(cfg.google_api_key)

    return providers

def build_resilient_providers(cfg: Config) -> dict[str, BaseProvider]:
    """Build providers wrapped with retry/fallback logic."""
    raw = build_providers(cfg)

    if not raw:
        return raw

    resilient: dict[str, BaseProvider] = {}

    # Determine fallback provider (cheapest available)
    # Priority: google > openai > anthropic > claude_cli
    fallback_provider = None
    fallback_model = ""
    for name, model in [("google", "gemini-2.5-flash"), ("openai", "gpt-4o-mini"), ("anthropic", "claude-sonnet-4-6")]:
        if name in raw:
            fallback_provider = raw[name]
            fallback_model = model
            break

    for name, provider in raw.items():
        # Pick a fallback that's different from the primary
        fb = fallback_provider if fallback_provider is not provider else None
        fb_model = fallback_model if fb else ""

        # Claude CLI: long timeout (Opus can take 5+ min), no retries —
        # SubagentRunner handles fallback at the subagent level.
        if name == "claude_cli":
            timeout = 600.0
            retries = 0
        else:
            timeout = 120.0
            retries = 2

        resilient[name] = ResilientProvider(
            primary=provider,
            fallback=fb,
            fallback_model=fb_model,
            max_retries=retries,
            timeout=timeout,
        )

        if fb:
            log.info("resilient_provider",
                     name=name, fallback=fallback_model)
        else:
            log.info("resilient_provider",
                     name=name, fallback="none")

    return resilient