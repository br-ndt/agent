"""Health check, status dashboard, and proactive alerting.

Runs an aiohttp server on a configurable port (default 8765).
Provides:
  GET /           — HTML status dashboard
  GET /health     — JSON health check (for uptime monitors)
  GET /stats      — detailed JSON stats (sessions, providers, costs)
  GET /costs      — cost summary (configurable days via ?days=N)
  GET /skills     — skill registry and recent run history

Active health checks run every 60s, pinging each provider.
When a provider fails (especially Claude CLI auth), alerts are sent
to Discord via the registered adapter.

No authentication — keep this behind a firewall or only expose
to your VPN/tailnet. The OCI security list controls who can reach it.
"""

import asyncio
import time
from datetime import datetime, timezone

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

# Health check state
_health_check_results: dict[str, dict] = {}
_health_check_task: asyncio.Task | None = None
_alert_channel_id: str | None = None  # Discord channel for alerts

# Track alert state to avoid spamming
_alerted_providers: set[str] = set()

HEALTH_CHECK_INTERVAL = 60  # seconds


def configure(router, orchestrator, providers, config, cost_tracker=None):
    """Called once at startup to wire in references."""
    global _start_time, _router, _orchestrator, _providers, _config, _cost_tracker
    global _alert_channel_id
    _start_time = time.time()
    _router = router
    _orchestrator = orchestrator
    _providers = providers
    _config = config
    _cost_tracker = cost_tracker

    # Use first admin's DM or a configured channel for alerts
    _alert_channel_id = getattr(config, "alert_channel_id", None)


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


# ── Active Health Checks ────────────────────────────────────────────


async def _check_provider(name: str, provider) -> dict:
    """Lightweight health check — test auth, not completions."""
    start = time.monotonic()
    try:
        actual = getattr(provider, "primary", provider)
        provider_type = type(actual).__name__

        if "ClaudeCLI" in provider_type:
            # Just validate the CLI is authed — no tokens burned
            proc = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    "claude",
                    "--version",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                ),
                timeout=10,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(stderr.decode()[:200])

        elif "Google" in provider_type:
            # List models endpoint — free, validates API key
            import aiohttp

            api_key = getattr(actual, "api_key", None)
            if api_key:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}",
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as resp:
                        if resp.status != 200:
                            raise RuntimeError(f"HTTP {resp.status}")

        elif "Anthropic" in provider_type:
            # Minimal API call — will fail fast on bad auth
            # without burning real tokens
            import aiohttp

            api_key = getattr(actual, "api_key", None)
            if api_key:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        "https://api.anthropic.com/v1/models",
                        headers={
                            "x-api-key": api_key,
                            "anthropic-version": "2023-06-01",
                        },
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as resp:
                        if resp.status == 401:
                            raise RuntimeError("401 Unauthorized")

        else:
            # Unknown provider type — skip
            return {
                "status": "unknown",
                "last_check": datetime.now(timezone.utc).isoformat(),
            }

        elapsed_ms = int((time.monotonic() - start) * 1000)
        return {
            "status": "ok",
            "latency_ms": elapsed_ms,
            "last_check": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        elapsed_ms = int((time.monotonic() - start) * 1000)
        error_str = str(e)

        # Detect specific failure types
        is_auth = any(
            marker in error_str.lower()
            for marker in ["401", "unauthorized", "auth", "login", "oauth"]
        )

        return {
            "status": "auth_failed" if is_auth else "error",
            "error": error_str[:200],
            "latency_ms": elapsed_ms,
            "last_check": datetime.now(timezone.utc).isoformat(),
        }


async def _send_alert(message: str):
    """Send an alert to Discord if an adapter is available."""
    if not _router:
        return

    adapter = _router.adapters.get("discord")
    if not adapter:
        return

    # Try configured alert channel first
    if _alert_channel_id:
        try:
            await adapter.send(_alert_channel_id, message)
            return
        except Exception as e:
            log.warning("alert_channel_send_failed", error=str(e))

    # Fall back to DMing the first admin
    if _config and _config.admin_ids:
        for admin_id in _config.admin_ids:
            if admin_id.startswith("discord:"):
                user_id = admin_id.split(":", 1)[1]
                try:
                    # Get or create DM channel
                    user = await adapter.client.fetch_user(int(user_id))
                    dm = await user.create_dm()
                    await adapter.send(str(dm.id), message)
                    return
                except Exception as e:
                    log.warning("alert_dm_failed", user_id=user_id, error=str(e))


async def _health_check_loop():
    """Periodically check all providers and send alerts on failure."""
    global _health_check_results, _alerted_providers

    # Wait a bit for everything to initialize
    await asyncio.sleep(10)

    while True:
        try:
            for name, provider in _providers.items():
                result = await _check_provider(name, provider)
                _health_check_results[name] = result

                if result["status"] != "ok":
                    if name not in _alerted_providers:
                        # First failure — send alert
                        _alerted_providers.add(name)
                        emoji = "🔑" if result["status"] == "auth_failed" else "🔴"
                        alert_msg = (
                            f"{emoji} **Provider Alert: {name}**\n"
                            f"Status: {result['status']}\n"
                            f"Error: {result.get('error', 'unknown')}"
                        )
                        if (
                            result["status"] == "auth_failed"
                            and "claude" in name.lower()
                        ):
                            alert_msg += (
                                "\n\n**Action needed:** Run `claude login` on the VPS "
                                "and copy credentials to the Docker mount path."
                            )
                        await _send_alert(alert_msg)
                        log.warning(
                            "health_check_failed",
                            provider=name,
                            status=result["status"],
                            error=result.get("error"),
                        )
                else:
                    if name in _alerted_providers:
                        # Was broken, now recovered
                        _alerted_providers.discard(name)
                        await _send_alert(
                            f"✅ **Provider Recovered: {name}** — back online"
                        )
                        log.info("health_check_recovered", provider=name)

        except Exception as e:
            log.error("health_check_loop_error", error=str(e))

        await asyncio.sleep(HEALTH_CHECK_INTERVAL)


# ── Stats Gathering ─────────────────────────────────────────────────


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
            "ids": list(sessions.keys())[:20],
        }

    # Adapters
    if _router:
        stats["adapters"] = list(_router.adapters.keys())
        stats["quiet"] = bool(getattr(_router, "_quiet", False))
        quiet_since = getattr(_router, "_quiet_since", None)
        if stats["quiet"] and quiet_since is not None:
            elapsed = int(time.monotonic() - quiet_since)
            days, rem = divmod(elapsed, 86400)
            hours, rem = divmod(rem, 3600)
            minutes, secs = divmod(rem, 60)
            parts = []
            if days:
                parts.append(f"{days}d")
            if hours:
                parts.append(f"{hours}h")
            if minutes:
                parts.append(f"{minutes}m")
            parts.append(f"{secs}s")
            stats["quiet_duration"] = " ".join(parts)
        else:
            stats["quiet_duration"] = None

    # Providers (with health check results)
    if _providers:
        provider_stats = {}
        for name, provider in _providers.items():
            p_info = {"type": type(provider).__name__}
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
            # Include health check results
            if name in _health_check_results:
                p_info["health"] = _health_check_results[name]
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

    # Skills
    if _orchestrator and hasattr(_orchestrator, "skill_registry"):
        registry = _orchestrator.skill_registry
        active_skills = registry.list_active()
        stats["skills"] = {
            "active_count": len(active_skills),
            "proposed_count": len(registry.list_proposed()),
            "skills": [
                {
                    "name": s.name,
                    "subagent": s.subagent,
                    "triggers": s.triggers,
                    "has_steps": s.has_steps,
                    "step_count": len(s.steps) if s.has_steps else 0,
                }
                for s in active_skills
            ],
        }

    # Session activity
    if _orchestrator:
        session_activity = []
        for sid, session in _orchestrator.sessions.items():
            activity: dict = {
                "session_id": sid,
                "messages": session.message_count,
            }

            # Active skill
            if sid in _orchestrator._skill_tasks:
                skill_status = _orchestrator.skill_executor.get_status(sid)
                activity["skill"] = skill_status or "running"

            # Pending results waiting to be delivered
            pending = _orchestrator._pending_results.get(sid, [])
            if pending:
                activity["pending_results"] = len(pending)

            session_activity.append(activity)

        # Active delegations (keyed by agent, not session)
        busy_agents = {}
        for agent_name, info in _orchestrator._active_delegations.items():
            busy_agents[agent_name] = {
                "session_id": info.get("session_id", "?"),
                "task": info.get("task", "?")[:120],
            }

        stats["session_activity"] = session_activity
        stats["busy_agents"] = busy_agents

    # Config summary
    if _config:
        stats["config"] = {
            "orchestrator_provider": _config.orchestrator_provider,
            "orchestrator_model": _config.orchestrator_model,
            "admin_count": len(_config.admin_ids),
            "trusted_count": len(_config.trusted_ids),
        }

    return stats


def _gather_skill_history() -> list[dict]:
    """Get recent skill runs from the registry."""
    if not _orchestrator or not hasattr(_orchestrator, "skill_registry"):
        return []

    registry = _orchestrator.skill_registry
    all_runs = []
    for skill in registry.list_active():
        runs = registry.get_run_history(skill.name, limit=5)
        for run in runs:
            run["skill_name"] = skill.name
            all_runs.append(run)

    # Sort by started_at descending
    all_runs.sort(key=lambda r: r.get("started_at", ""), reverse=True)
    return all_runs[:20]


# ── HTTP Handlers ───────────────────────────────────────────────────


async def handle_index(request):
    """HTML status dashboard."""
    stats = _gather_stats()
    skill_runs = _gather_skill_history()
    now = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    # Build provider rows
    provider_rows = ""
    for name, info in stats.get("providers", {}).items():
        health = info.get("health", {})
        health_status = health.get("status", "unknown")
        latency = health.get("latency_ms", "—")
        error = health.get("error", "")
        last_check = health.get("last_check", "—")
        if last_check != "—":
            last_check = last_check[11:19]  # Just the time portion

        if health_status == "ok":
            status_badge = '<span class="badge ok">OK</span>'
        elif health_status == "auth_failed":
            status_badge = '<span class="badge auth">AUTH</span>'
        elif health_status == "error":
            status_badge = '<span class="badge error">ERR</span>'
        else:
            status_badge = '<span class="badge unknown">?</span>'

        cb_info = ""
        if "circuit_breaker" in info:
            cb = info["circuit_breaker"]
            if cb["open"]:
                cb_info = ' <span class="badge error">CIRCUIT OPEN</span>'

        primary = info.get("primary", info["type"])
        fallback = info.get("fallback", "—")

        provider_rows += f"""
        <tr>
            <td class="mono">{name}</td>
            <td>{status_badge}{cb_info}</td>
            <td>{primary}</td>
            <td>{fallback}</td>
            <td class="mono">{latency}ms</td>
            <td class="dim">{last_check}</td>
        </tr>"""

    # Build subagent rows
    subagent_rows = ""
    for name, info in stats.get("subagents", {}).items():
        tools = ", ".join(info["tools"]) if info["tools"] else "—"
        subagent_rows += f"""
        <tr>
            <td class="mono">{name}</td>
            <td>{info['provider']}</td>
            <td>{info['model']}</td>
            <td class="dim">{tools}</td>
        </tr>"""

    # Build skill rows
    skill_rows = ""
    for s in stats.get("skills", {}).get("skills", []):
        triggers = ", ".join(s["triggers"][:3]) if s["triggers"] else "—"
        steps = f'{s["step_count"]} steps' if s["has_steps"] else "prompt"
        skill_rows += f"""
        <tr>
            <td class="mono">{s['name']}</td>
            <td>{s['subagent']}</td>
            <td>{steps}</td>
            <td class="dim">{triggers}</td>
        </tr>"""

    # Build session activity rows
    session_rows = ""
    busy_agents = stats.get("busy_agents", {})
    for sa in stats.get("session_activity", []):
        sid = sa["session_id"]
        msg_count = sa.get("messages", 0)

        # Determine status and details
        if "skill" in sa:
            status_badge = '<span class="badge unknown">SKILL</span>'
            # Extract skill name and step from status string
            detail = sa["skill"]
            if len(detail) > 120:
                detail = detail[:120] + "..."
        elif sa.get("pending_results"):
            status_badge = '<span class="badge ok">DONE</span>'
            detail = f"{sa['pending_results']} result(s) waiting"
        else:
            status_badge = '<span class="badge unknown">IDLE</span>'
            detail = "—"

        session_rows += f"""
        <tr>
            <td class="mono">{sid}</td>
            <td class="mono">{msg_count}</td>
            <td>{status_badge}</td>
            <td class="dim">{detail}</td>
        </tr>"""

    # Add busy agent rows
    for agent_name, info in busy_agents.items():
        task_preview = info.get("task", "?")
        if len(task_preview) > 100:
            task_preview = task_preview[:100] + "..."
        session_rows += f"""
        <tr>
            <td class="mono">{info.get('session_id', '?')}</td>
            <td>—</td>
            <td><span class="badge auth">AGENT</span></td>
            <td class="dim"><strong>{agent_name}</strong>: {task_preview}</td>
        </tr>"""

    # Build skill run history rows
    run_rows = ""
    for run in skill_runs:
        status = run.get("status", "unknown")
        if status == "success":
            status_badge = '<span class="badge ok">OK</span>'
        elif status == "failed":
            status_badge = '<span class="badge error">FAIL</span>'
        elif status == "no_op":
            status_badge = '<span class="badge auth">NO-OP</span>'
        else:
            status_badge = f'<span class="badge unknown">{status}</span>'

        started = run.get("started_at", "—")
        if started != "—" and len(started) > 19:
            started = started[:19].replace("T", " ")
        error = run.get("error") or "—"
        if len(error) > 60:
            error = error[:60] + "…"

        run_rows += f"""
        <tr>
            <td class="mono">{run.get('skill_name', '?')}</td>
            <td>{status_badge}</td>
            <td class="dim">{started}</td>
            <td class="dim">{error}</td>
        </tr>"""

    if not run_rows:
        run_rows = '<tr><td colspan="4" class="dim">No runs recorded yet</td></tr>'

    # Session info
    session_count = stats.get("sessions", {}).get("active", 0)

    # Overall status
    any_unhealthy = any(
        info.get("health", {}).get("status", "ok") != "ok"
        for info in stats.get("providers", {}).values()
    )
    overall_class = "error" if any_unhealthy else "ok"
    overall_text = "DEGRADED" if any_unhealthy else "HEALTHY"

    is_quiet = stats.get("quiet", False)
    quiet_duration = stats.get("quiet_duration")
    if is_quiet and quiet_duration:
        quiet_badge = f' <span class="overall error">QUIET ({quiet_duration})</span>'
    elif is_quiet:
        quiet_badge = ' <span class="overall error">QUIET</span>'
    else:
        quiet_badge = ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agent Status</title>
<meta http-equiv="refresh" content="30">
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Outfit:wght@300;400;600;700&display=swap');

  :root {{
    --bg: #0d1117;
    --surface: #161b22;
    --surface-2: #1c2129;
    --border: #30363d;
    --text: #e6edf3;
    --text-dim: #7d8590;
    --green: #3fb950;
    --green-bg: rgba(63, 185, 80, 0.1);
    --red: #f85149;
    --red-bg: rgba(248, 81, 73, 0.1);
    --orange: #d29922;
    --orange-bg: rgba(210, 153, 34, 0.1);
    --blue: #58a6ff;
    --purple: #bc8cff;
  }}

  * {{ margin: 0; padding: 0; box-sizing: border-box; }}

  body {{
    font-family: 'Outfit', sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
    padding: 2rem;
    max-width: 960px;
    margin: 0 auto;
  }}

  .header {{
    display: flex;
    align-items: baseline;
    gap: 1rem;
    margin-bottom: 2rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
  }}

  .header h1 {{
    font-weight: 700;
    font-size: 1.5rem;
    letter-spacing: -0.02em;
  }}

  .header .uptime {{
    color: var(--text-dim);
    font-size: 0.875rem;
    font-weight: 300;
  }}

  .header .overall {{
    margin-left: auto;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    font-weight: 600;
    padding: 0.25rem 0.75rem;
    border-radius: 4px;
    letter-spacing: 0.05em;
  }}

  .header .overall.ok {{
    background: var(--green-bg);
    color: var(--green);
  }}

  .header .overall.error {{
    background: var(--red-bg);
    color: var(--red);
  }}

  .section {{
    margin-bottom: 2rem;
  }}

  .section h2 {{
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-dim);
    margin-bottom: 0.75rem;
  }}

  .card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
  }}

  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.875rem;
  }}

  th {{
    text-align: left;
    padding: 0.625rem 1rem;
    background: var(--surface-2);
    color: var(--text-dim);
    font-weight: 400;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border-bottom: 1px solid var(--border);
  }}

  td {{
    padding: 0.5rem 1rem;
    border-bottom: 1px solid var(--border);
  }}

  tr:last-child td {{
    border-bottom: none;
  }}

  .mono {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8125rem;
  }}

  .dim {{
    color: var(--text-dim);
  }}

  .badge {{
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6875rem;
    font-weight: 600;
    padding: 0.125rem 0.5rem;
    border-radius: 3px;
    letter-spacing: 0.05em;
  }}

  .badge.ok {{
    background: var(--green-bg);
    color: var(--green);
  }}

  .badge.error {{
    background: var(--red-bg);
    color: var(--red);
  }}

  .badge.auth {{
    background: var(--orange-bg);
    color: var(--orange);
  }}

  .badge.unknown {{
    background: rgba(139, 148, 158, 0.1);
    color: var(--text-dim);
  }}

  .stats-row {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 1rem;
    margin-bottom: 2rem;
  }}

  .stat-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
  }}

  .stat-card .label {{
    font-size: 0.6875rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-dim);
    margin-bottom: 0.25rem;
  }}

  .stat-card .value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    font-weight: 600;
  }}

  .stat-card .value.green {{ color: var(--green); }}
  .stat-card .value.blue {{ color: var(--blue); }}
  .stat-card .value.purple {{ color: var(--purple); }}

  .footer {{
    text-align: center;
    color: var(--text-dim);
    font-size: 0.75rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
  }}
</style>
</head>
<body>

<div class="header">
  <h1>Agent Dashboard</h1>
  <span class="uptime">up {stats['uptime']}</span>
  <span class="overall {overall_class}">{overall_text}</span>{quiet_badge}
</div>

<div class="stats-row">
  <div class="stat-card">
    <div class="label">Sessions</div>
    <div class="value blue">{session_count}</div>
  </div>
  <div class="stat-card">
    <div class="label">Providers</div>
    <div class="value green">{len(stats.get('providers', {}))}</div>
  </div>
  <div class="stat-card">
    <div class="label">Subagents</div>
    <div class="value purple">{len(stats.get('subagents', {}))}</div>
  </div>
  <div class="stat-card">
    <div class="label">Skills</div>
    <div class="value green">{stats.get('skills', {}).get('active_count', 0)}</div>
  </div>
</div>

<div class="section">
  <h2>Session Activity</h2>
  <div class="card">
    <table>
      <tr><th>Session</th><th>Messages</th><th>Status</th><th>Details</th></tr>
      {session_rows or '<tr><td colspan="4" class="dim">No active sessions</td></tr>'}
    </table>
  </div>
</div>

<div class="section">
  <h2>Providers</h2>
  <div class="card">
    <table>
      <tr><th>Name</th><th>Health</th><th>Primary</th><th>Fallback</th><th>Latency</th><th>Checked</th></tr>
      {provider_rows or '<tr><td colspan="6" class="dim">No providers configured</td></tr>'}
    </table>
  </div>
</div>

<div class="section">
  <h2>Subagents</h2>
  <div class="card">
    <table>
      <tr><th>Name</th><th>Provider</th><th>Model</th><th>Tools</th></tr>
      {subagent_rows or '<tr><td colspan="4" class="dim">No subagents configured</td></tr>'}
    </table>
  </div>
</div>

<div class="section">
  <h2>Skills</h2>
  <div class="card">
    <table>
      <tr><th>Name</th><th>Agent</th><th>Type</th><th>Triggers</th></tr>
      {skill_rows or '<tr><td colspan="4" class="dim">No skills configured</td></tr>'}
    </table>
  </div>
</div>

<div class="section">
  <h2>Recent Skill Runs</h2>
  <div class="card">
    <table>
      <tr><th>Skill</th><th>Status</th><th>Time</th><th>Error</th></tr>
      {run_rows}
    </table>
  </div>
</div>

<div class="footer">
  Auto-refreshes every 30s · {now}
</div>

</body>
</html>"""

    return web.Response(text=html, content_type="text/html")


async def handle_health(request):
    """Minimal health check — returns 200 if alive, includes provider health."""
    provider_health = {}
    for name, result in _health_check_results.items():
        provider_health[name] = result.get("status", "unknown")

    any_unhealthy = any(s != "ok" for s in provider_health.values())

    is_quiet = bool(_router and getattr(_router, "_quiet", False))

    return web.json_response(
        {
            "status": "degraded" if any_unhealthy else "ok",
            "uptime_seconds": int(time.time() - _start_time),
            "providers": provider_health,
            "quiet": is_quiet,
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


async def handle_skills(request):
    """Skill registry and run history."""
    stats = _gather_stats()
    runs = _gather_skill_history()
    return web.json_response(
        {
            "skills": stats.get("skills", {}),
            "recent_runs": runs,
        }
    )


async def start_status_server(port: int = 8765):
    """Start the status HTTP server and background health checks."""
    global _health_check_task

    app = web.Application()
    app.router.add_get("/", handle_index)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/stats", handle_stats)
    app.router.add_get("/costs", handle_costs)
    app.router.add_get("/skills", handle_skills)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()

    # Start background health check loop
    _health_check_task = asyncio.create_task(_health_check_loop())

    log.info("status_server_started", port=port, url=f"http://0.0.0.0:{port}")
    return runner
