"""Entry point — start adapters and orchestrator."""

import asyncio
import signal
import sys

import structlog

from agent.adapters import cli as cli_adapter, discord as discord_adapter

from agent.config import load_config
from agent.orchestrator import Orchestrator
from agent.personas import load_personas, apply_persona
from agent.providers import build_providers, build_resilient_providers
from agent.router import Router
from agent.session_store import SessionStore
from agent.subagent_runner import SubagentRunner

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO
)

log = structlog.get_logger()


async def main():
    log.info("agent_starting")

    # ── Load config ───────────────────────────────────────────
    cfg = load_config()

    # ── Build providers ───────────────────────────────────────
    providers = build_resilient_providers(cfg)
    if not providers:
        log.error("no_providers_configured",
                  hint="Set at least ANTHROPIC_API_KEY in .env")
        sys.exit(1)

    log.info("providers_ready", providers=list(providers.keys()))
    personas = load_personas()

    # ── Build subagent runners ────────────────────────────────
    subagent_runners: dict[str, SubagentRunner] = {}
    for name, sa_cfg in cfg.subagents.items():
        if sa_cfg.provider in providers:
            # Inject persona into the subagent's personality
            if name in personas:
                sa_cfg.personality = apply_persona(sa_cfg.personality, personas[name])
            subagent_runners[name] = SubagentRunner(sa_cfg, providers[sa_cfg.provider])
            log.info("subagent_registered",
                    name=name, provider=sa_cfg.provider, model=sa_cfg.model,
                    has_persona=name in personas)
        else:
            log.warning("subagent_skipped_no_provider",
                        name=name, provider=sa_cfg.provider)

    # ── Session store ─────────────────────────────────────────
    session_store = SessionStore()
    await session_store.init()

    # ── Build orchestrator ────────────────────────────────────
    # Inject orchestrator persona
    if "orchestrator" in personas:
        cfg.orchestrator_system_prompt = apply_persona(
            cfg.orchestrator_system_prompt, personas["orchestrator"]
        )
    orch_provider = providers.get(cfg.orchestrator_provider)
    if not orch_provider:
        # Fall back to first available provider
        orch_provider = next(iter(providers.values()))
        log.warning("orchestrator_provider_fallback",
                    wanted=cfg.orchestrator_provider,
                    using=next(iter(providers.keys())))

    orchestrator = Orchestrator(
        provider=orch_provider,
        config=cfg,
        subagent_runners=subagent_runners,
        providers=providers,
        session_store=session_store,
    )

    # ── Build router ──────────────────────────────────────────
    router = Router(cfg, orchestrator)

    # ── Start adapters ────────────────────────────────────────
    adapter_tasks = []

    # Discord adapter (if token provided)
    if cfg.discord_bot_token:
        discord_bot = discord_adapter.DiscordAdapter(
            token=cfg.discord_bot_token,
            allowed_ids=cfg.admin_ids | cfg.trusted_ids or None,
        )
        router.register_adapter("discord", discord_bot)
        adapter_tasks.append(discord_bot.start(router.handle_message))
        log.info("discord_adapter_enabled")

    # Telegram adapter (if token provided)
    if cfg.telegram_bot_token:
        from agent.adapters.telegram import TelegramAdapter
        telegram = TelegramAdapter(
            token=cfg.telegram_bot_token,
            allowed_ids=cfg.admin_ids | cfg.trusted_ids or None,
        )
        router.register_adapter("telegram", telegram)
        adapter_tasks.append(telegram.start(router.handle_message))
        log.info("telegram_adapter_enabled")

    # Check for --no-cli flag (daemon mode)
    daemon_mode = "--no-cli" in sys.argv

    if not daemon_mode:
        cli = cli_adapter.CLIAdapter()
        router.register_adapter("cli", cli)

    # Start background adapters
    for task in adapter_tasks:
        asyncio.create_task(task)

    log.info("agent_ready",
             adapters=list(router.adapters.keys()),
             subagents=list(subagent_runners.keys()),
             daemon_mode=daemon_mode)

    if daemon_mode:
        # Run forever — Discord/Telegram adapters handle messages
        log.info("daemon_mode_running", hint="Ctrl+C or systemctl stop to exit")
        await asyncio.Event().wait()  # Block forever
    else:
        await cli.start(router.handle_message)

    # Graceful shutdown
    log.info("agent_shutting_down")
    await session_store.close()
    for name, adapter in router.adapters.items():
        if name != "cli":
            await adapter.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown by Ctrl+C")
    except SystemExit:
        pass
