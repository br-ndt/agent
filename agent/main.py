"""Entry point — start adapters and orchestrator."""

import asyncio
import os
import signal
import sys

import structlog

from agent.adapters import cli as cli_adapter, discord as discord_adapter

from agent.config import load_config, infer_provider
from agent.cost_tracker import CostTracker
from agent.memory import MemoryStore
from agent.orchestrator import Orchestrator
from agent.personas import load_personas, apply_persona
from agent.skills import load_skills
from agent.persona_enforcement import build_enforced_prompt
from agent.providers import build_providers, build_resilient_providers
from agent.providers.vision import VisionProvider  # NEW
from agent.router import Router
from agent.session_store import SessionStore
from agent.state_ledger import StateLedger
from agent.status_server import configure as configure_status, start_status_server
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
        log.error(
            "no_providers_configured", hint="Set at least ANTHROPIC_API_KEY in .env"
        )
        sys.exit(1)

    log.info("providers_ready", providers=list(providers.keys()))
    personas = load_personas()
    skills = load_skills()

    # ── Vision provider (optional) ────────────────────────────
    vision_provider = None
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if google_api_key:
        try:
            vision_provider = VisionProvider(api_key=google_api_key)
            log.info("vision_provider_enabled")
        except Exception as e:
            log.warning("vision_provider_init_failed", error=str(e))
    else:
        log.info("vision_provider_disabled", hint="Set GOOGLE_API_KEY to enable")

    # ── Session store ─────────────────────────────────────────
    session_store = SessionStore()
    await session_store.init()

    # ── Cost tracker ──────────────────────────────────────────
    cost_tracker = CostTracker()
    await cost_tracker.init()

    # ── State ledger ───────────────────────────────────────────
    memory_store = MemoryStore()
    await memory_store.init()
    state_ledger = StateLedger()
    await state_ledger.init()

    # ── Build subagent runners ────────────────────────────────
    subagent_runners: dict[str, SubagentRunner] = {}
    for name, sa_cfg in cfg.subagents.items():
        provider_name = sa_cfg.resolved_provider
        primary = providers.get(provider_name)
        if not primary:
            log.warning("subagent_skipped_no_provider", name=name, provider=provider_name)
            continue

        # Resolve fallback provider (SubagentRunner handles retry internally)
        fallback = None
        fallback_model = sa_cfg.fallback_model
        if fallback_model:
            fallback_name = infer_provider(fallback_model)
            fallback = providers.get(fallback_name)
            if fallback is primary:
                fallback = None  # same provider, no point

        # Inject persona into the subagent's personality
        if name in personas:
            sa_cfg.personality = build_enforced_prompt(
                agent_name=name,
                system_prompt=sa_cfg.personality,
                persona=personas[name],
            )
        else:
            sa_cfg.personality = build_enforced_prompt(
                agent_name=name,
                system_prompt=sa_cfg.personality,
            )

        # Inject skills relevant to this subagent
        relevant_skills = [s for s in skills if s.subagent == name]
        if relevant_skills:
            skill_blocks = [
                f"### Skill: {s.name}\n{s.description}\n\n{s.content}"
                for s in relevant_skills
            ]
            sa_cfg.personality += "\n\n## Skills\n" + "\n\n---\n\n".join(skill_blocks)
            log.info("skills_injected", agent=name, count=len(relevant_skills))

        subagent_runners[name] = SubagentRunner(
            sa_cfg,
            primary,
            fallback_provider=fallback,
            fallback_model=fallback_model if fallback else "",
            cost_tracker=cost_tracker,
        )
        log.info(
            "subagent_registered",
            name=name,
            provider=provider_name,
            native_tools=getattr(primary, "has_native_tools", False),
            model=sa_cfg.model,
            fallback=fallback_model if fallback else "none",
            has_persona=name in personas,
        )

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
        log.warning(
            "orchestrator_provider_fallback",
            wanted=cfg.orchestrator_provider,
            using=next(iter(providers.keys())),
        )

    orchestrator = Orchestrator(
        provider=orch_provider,
        config=cfg,
        subagent_runners=subagent_runners,
        providers=providers,
        session_store=session_store,
        skills=skills,
        cost_tracker=cost_tracker,
        vision_provider=vision_provider,
        memory_store=memory_store,
        state_ledger=state_ledger,
    )

    # ── Build router ──────────────────────────────────────────
    router = Router(cfg, orchestrator)

    # ── Status server ─────────────────────────────────────────
    configure_status(router, orchestrator, providers, cfg, cost_tracker)
    status_runner = await start_status_server(port=cfg.status_port)

    # ── Start adapters ────────────────────────────────────────
    adapter_tasks = []

    # Discord adapter (if token provided)
    if cfg.discord_bot_token:
        # Use cheapest provider for relevance filtering
        relevance_provider = providers.get("google", next(iter(providers.values())))

        discord_bot = discord_adapter.DiscordAdapter(
            token=cfg.discord_bot_token,
            allowed_ids=cfg.admin_ids | cfg.trusted_ids or None,
            other_bot_ids=cfg.other_bot_ids if hasattr(cfg, "other_bot_ids") else None,
            relevance_provider=relevance_provider,
            relevance_model="gemini-2.5-flash",
            other_bots=cfg.other_bots,
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

    log.info(
        "agent_ready",
        adapters=list(router.adapters.keys()),
        subagents=list(subagent_runners.keys()),
        daemon_mode=daemon_mode,
        vision_enabled=vision_provider is not None,
    )

    if daemon_mode:
        # Run forever — Discord/Telegram adapters handle messages
        log.info("daemon_mode_running", hint="Ctrl+C or systemctl stop to exit")
        await asyncio.Event().wait()  # Block forever
    else:
        await cli.start(router.handle_message)

    # Graceful shutdown
    log.info("agent_shutting_down")
    await cost_tracker.close()
    await session_store.close()
    await memory_store.close()
    await state_ledger.close()
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
