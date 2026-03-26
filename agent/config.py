"""Configuration loader — reads .env and config.yaml."""

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass
class SubagentConfig:
    name: str
    provider: str
    model: str
    personality: str
    tools: list[str] = field(default_factory=list)
    max_tokens: int = 4096
    temperature: float = 0.7


@dataclass
class Config:
    # LLM keys
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    google_api_key: str = ""

    # Chat adapters
    telegram_bot_token: str = ""
    discord_bot_token: str = ""
    other_bots: dict[str, str] = field(default_factory=dict)

    # Access control
    admin_ids: set[str] = field(default_factory=set)
    trusted_ids: set[str] = field(default_factory=set)

    # Tier-based provider+model overrides: tier -> {provider, model}
    tier_routing: dict[str, dict[str, str]] = field(default_factory=dict)

    # Orchestrator
    orchestrator_provider: str = "claude_cli"
    orchestrator_model: str = "sonnet"
    orchestrator_system_prompt: str = ""
    orchestrator_max_tokens: int = 4096

    # Subagents
    subagents: dict[str, SubagentConfig] = field(default_factory=dict)

    # Services
    status_port: int = 8765
    log_level: str = "INFO"


def load_config(config_path: Path | None = None) -> Config:
    """Load config from .env and optional config.yaml."""
    cfg = Config(
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        google_api_key=os.getenv("GOOGLE_API_KEY", ""),
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
        discord_bot_token=os.getenv("DISCORD_BOT_TOKEN", ""),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        status_port=int(os.getenv("STATUS_PORT", "8765")),
    )

    # Admin IDs from env
    admin_id = os.getenv("ADMIN_TELEGRAM_ID", "")
    if admin_id:
        cfg.admin_ids.add(f"telegram:{admin_id}")

    # Load YAML config if it exists
    yaml_path = config_path or BASE_DIR / "config.yaml"
    if yaml_path.exists():
        with open(yaml_path) as f:
            raw = yaml.safe_load(f) or {}
        _apply_yaml(cfg, raw)
    else:
        # Apply sensible defaults when no config.yaml exists
        _apply_defaults(cfg)

    return cfg


def _apply_yaml(cfg: Config, raw: dict):
    """Merge YAML config into the Config object."""
    # Orchestrator
    orch = raw.get("orchestrator", {})
    cfg.orchestrator_provider = orch.get("provider", cfg.orchestrator_provider)
    cfg.orchestrator_model = orch.get("model", cfg.orchestrator_model)
    cfg.orchestrator_system_prompt = orch.get("system_prompt", "")
    cfg.orchestrator_max_tokens = orch.get("max_tokens", cfg.orchestrator_max_tokens)

    # Subagents
    for name, sa_raw in raw.get("subagents", {}).items():
        cfg.subagents[name] = SubagentConfig(
            name=name,
            provider=sa_raw.get("provider", "anthropic"),
            model=sa_raw.get("model", "claude-sonnet-4-6"),
            personality=sa_raw.get("personality", ""),
            tools=sa_raw.get("tools", []),
            max_tokens=sa_raw.get("max_tokens", 4096),
            temperature=sa_raw.get("temperature", 0.7),
        )

    # Access control
    for aid in raw.get("access", {}).get("admin_ids", []) or []:
        cfg.admin_ids.add(aid)
    for tid in raw.get("access", {}).get("trusted_ids", []) or []:
        cfg.trusted_ids.add(tid)
    for tier, routing in raw.get("access", {}).get("tier_routing", {}).items():
        if tier and routing:
            cfg.tier_routing[tier] = {
                "provider": routing.get("provider", cfg.orchestrator_provider),
                "model": routing.get("model", cfg.orchestrator_model),
            }
    cfg.other_bots = raw.get("access", {}).get("other_bots", {}) or {}


def _apply_defaults(cfg: Config):
    """Set up sensible default subagents when no config.yaml."""
    cfg.orchestrator_system_prompt = (
        "You are a helpful multi-agent assistant. When a task is simple, handle it "
        "directly. When a task is complex (writing code, reviewing text, researching), "
        "you can delegate to subagents by emitting:\n"
        '<delegate agent="agent_name">task description</delegate>\n\n'
        "Available subagents: {subagent_list}\n\n"
        "After receiving subagent results, synthesize them into a clear final answer."
    )

    cfg.subagents["coder"] = SubagentConfig(
        name="coder",
        provider="claude_cli",
        model="sonnet",
        personality=(
            "You are a senior software engineer. Write clean, well-documented code. "
            "Prefer simplicity. Always include error handling. Use type hints in Python."
        ),
        tools=["bash", "file_ops"],
        max_tokens=8192,
    )

    cfg.subagents["reviewer"] = SubagentConfig(
        name="reviewer",
        provider="claude_cli",
        model="sonnet",
        personality=(
            "You are a meticulous code reviewer. Focus on bugs, security issues, "
            "and readability. Be constructive but thorough. Suggest changes, don't make them."
        ),
        tools=[],
        max_tokens=4096,
    )
