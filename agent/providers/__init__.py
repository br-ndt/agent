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

    # Claude CLI provider — uses your Pro/Max subscription (no API key needed)
    if shutil.which("claude"):
        from .claude_cli import ClaudeCLIProvider

        # Text-only (orchestrator) — no sandbox needed, can't write
        providers["claude_cli"] = ClaudeCLIProvider(
            disallowed_tools=["Bash", "Read", "Write", "Edit", "ListDir",
                              "Grep", "Glob", "WebSearch", "WebFetch"],
        )

        # Tool-enabled claudes — sandboxed, can't escape workspace
        providers["claude_cli_tools"] = ClaudeCLIProvider(
            docker_image="agent-coder",
            disallowed_tools=["WebSearch", "WebFetch"],  # Allow coding tools, disallow internet access
            timeout=300,
        )

        providers["claude_cli_research"] = ClaudeCLIProvider(
            docker_image="agent-coder",
            allowed_tools=["WebSearch", "WebFetch", "Read"],
            timeout=120,
        )
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

        # Docker-sandboxed providers must not fall back to unsandboxed ones
        if hasattr(provider, 'docker_image') and provider.docker_image:
            fb = None
            fb_model = ""

        # Claude CLI gets longer timeout (subprocess overhead)
        timeout = 180.0 if name == "claude_cli" else 120.0

        resilient[name] = ResilientProvider(
            primary=provider,
            fallback=fb,
            fallback_model=fb_model,
            max_retries=2,
            timeout=timeout,
        )

        if fb:
            log.info("resilient_provider",
                     name=name, fallback=fallback_model)
        else:
            log.info("resilient_provider",
                     name=name, fallback="none")

    return resilient