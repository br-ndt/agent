"""Health check and status HTTP endpoint.

Runs a tiny aiohttp server on a configurable port (default 8765).
Provides:
  GET /           — human-readable status page
  GET /health     — JSON health check (for uptime monitors)
  GET /stats      — detailed JSON stats (sessions, providers, costs)

No authentication — keep this behind a firewall or only expose
to your VPN/tailnet. The OCI security list controls who can reach it.
"""

import time
from aiohttp import web

import structlog

from agent.cost_tracker import CostTracker

log = structlog.get_logger()

# Set at startup by main.py
_start_time: float = 0
_router = None
_orchestrator = None
_providers: dict = {}
_config = None
_cost_tracker = None


def configure(router, orchestrator, providers, config, cost_tracker=None):
    """Called once at startup to wire in references."""
    global _start_time, _router, _orchestrator, _providers, _config, _cost_tracker
    _start_time = time.time()
    _router = router
    _orchestrator = orchestrator
    _providers = providers
    _config = config
    _cost_tracker = cost_tracker


def _uptime() -> str:
    """Human-readable uptime string."""
    seconds = int(time.time() - _start_time)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")
    return " ".join(parts)


def _gather_stats() -> dict:
    """Collect current system stats."""
    stats = {
        "status": "ok",
        "uptime": _uptime(),
        "uptime_seconds": int(time.time() - _start_time),
    }

    # Sessions
    if _orchestrator:
        sessions = _orchestrator.sessions
        stats["sessions"] = {
            "active": len(sessions),
            "ids": list(sessions.keys())[:20],  # Cap at 20 for readability
        }

    # Adapters
    if _router:
        stats["adapters"] = list(_router.adapters.keys())

    # Providers
    if _providers:
        provider_stats = {}
        for name, provider in _providers.items():
            p_info = {"type": type(provider).__name__}
            # If resilient, show circuit breaker state
            if hasattr(provider, "primary_circuit"):
                cb = provider.primary_circuit
                p_info["circuit_breaker"] = {
                    "open": cb.is_open,
                    "consecutive_failures": cb.consecutive_failures,
                }
            if hasattr(provider, "primary"):
                p_info["primary"] = type(provider.primary).__name__
            if hasattr(provider, "fallback") and provider.fallback:
                p_info["fallback"] = type(provider.fallback).__name__
            provider_stats[name] = p_info
        stats["providers"] = provider_stats

    # Subagents
    if _orchestrator:
        stats["subagents"] = {
            name: {
                "provider": runner.config.provider,
                "model": runner.config.model,
                "tools": runner.config.tools,
            }
            for name, runner in _orchestrator.subagents.items()
        }

    # Config summary
    if _config:
        stats["config"] = {
            "orchestrator_provider": _config.orchestrator_provider,
            "orchestrator_model": _config.orchestrator_model,
            "admin_count": len(_config.admin_ids),
            "trusted_count": len(_config.trusted_ids),
        }

    return stats


async def handle_index(request):
    """Human-readable status page."""
    stats = _gather_stats()

    lines = [
        "╔══════════════════════════════════════╗",
        "║         Agent Status Dashboard       ║",
        "╚══════════════════════════════════════╝",
        "",
        f"  Status:    {stats['status']}",
        f"  Uptime:    {stats['uptime']}",
        "",
    ]

    if "adapters" in stats:
        lines.append(f"  Adapters:  {', '.join(stats['adapters'])}")

    if "sessions" in stats:
        s = stats["sessions"]
        lines.append(f"  Sessions:  {s['active']} active")

    if "providers" in stats:
        lines.append("")
        lines.append("  Providers:")
        for name, info in stats["providers"].items():
            cb_status = ""
            if "circuit_breaker" in info:
                cb = info["circuit_breaker"]
                if cb["open"]:
                    cb_status = " [CIRCUIT OPEN]"
                elif cb["consecutive_failures"] > 0:
                    cb_status = f" [{cb['consecutive_failures']} failures]"
            primary = info.get("primary", info["type"])
            fallback = info.get("fallback", "none")
            lines.append(f"    {name}: {primary} → {fallback}{cb_status}")

    if "subagents" in stats:
        lines.append("")
        lines.append("  Subagents:")
        for name, info in stats["subagents"].items():
            tools = ", ".join(info["tools"]) if info["tools"] else "none"
            lines.append(
                f"    {name}: {info['provider']}:{info['model']} tools=[{tools}]"
            )

    lines.append("")
    lines.append(
        f"  Last checked: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}"
    )

    return web.Response(text="\n".join(lines), content_type="text/plain")


async def handle_health(request):
    """Minimal health check — returns 200 if alive."""
    return web.json_response(
        {
            "status": "ok",
            "uptime_seconds": int(time.time() - _start_time),
        }
    )


async def handle_stats(request):
    """Detailed JSON stats."""
    return web.json_response(_gather_stats())


async def handle_costs(request):
    """Cost and usage summary."""
    days = int(request.query.get("days", "7"))
    if _cost_tracker:
        summary = await _cost_tracker.get_summary(days=days)
        return web.json_response(summary)
    return web.json_response({"error": "Cost tracker not configured"})


async def start_status_server(port: int = 8765):
    """Start the status HTTP server."""
    app = web.Application()
    app.router.add_get("/", handle_index)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/stats", handle_stats)
    app.router.add_get("/costs", handle_costs)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    log.info("status_server_started", port=port, url=f"http://0.0.0.0:{port}")
    return runner
