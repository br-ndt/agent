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
import json
import os
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path

from aiohttp import web
import structlog

from agent.cost_tracker import CostTracker
from agent.diagnostics import ErrorJournal

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

# ── OBJ export API key ──────────────────────────────────────────────
_EXPORT_KEY_FILE = Path(__file__).parent / "export_key.txt"
_export_api_key: str = ""

# Cloister HTML files live in the coder workspace
_CLOISTER_HTML_DIR = Path("/home/ubuntu/agent/workspaces/coder/cloister/public")


def _load_or_create_export_key() -> str:
    """Load the export API key.

    Priority:
      1. CLOISTER_API_KEY environment variable — preferred for external access.
      2. export_key.txt on disk — fallback; a new key is generated if missing.

    Logs a warning when the env var is absent so operators know to set it.
    """
    global _export_api_key

    env_key = os.environ.get("CLOISTER_API_KEY", "").strip()
    if env_key:
        _export_api_key = env_key
        print(
            "[status_server] CLOISTER_API_KEY loaded from environment variable.",
            flush=True,
        )
        return env_key

    # Env var not set — warn and fall back to file-based key.
    print(
        "[status_server] WARNING: CLOISTER_API_KEY env var not set. "
        "Falling back to file-based key. Set CLOISTER_API_KEY for external access.",
        flush=True,
    )
    if _EXPORT_KEY_FILE.exists():
        key = _EXPORT_KEY_FILE.read_text().strip()
        if key:
            _export_api_key = key
            return key
    # Generate a fresh 32-char hex key and persist it.
    key = secrets.token_hex(16)  # 16 bytes = 32 hex chars
    _EXPORT_KEY_FILE.write_text(key + "\n")
    _export_api_key = key
    return key


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

            # Active request (orchestrator is processing this session)
            active_req = _orchestrator._active_requests.get(sid)
            if active_req:
                activity["active_request"] = active_req

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
    """HTML status dashboard with tabbed layout."""
    stats = _gather_stats()
    skill_runs = _gather_skill_history()
    now = datetime.now(timezone.utc).isoformat()

    # Pre-fetch journal data so it's embedded in the page (no AJAX needed for initial load)
    try:
        journal = getattr(_orchestrator, "error_journal", None) if _orchestrator else None
        if journal is None:
            journal = ErrorJournal()
        journal_data = journal.query(page=1, per_page=50)
        archive_list = journal.list_archives()
    except Exception:
        journal_data = {"entries": [], "total": 0, "page": 1, "per_page": 50,
                        "pages": 0, "levels": [], "events": []}
        archive_list = []
    journal_json = json.dumps(journal_data)
    archive_list_json = json.dumps(archive_list)

    # Build provider rows
    provider_rows = ""
    for name, info in stats.get("providers", {}).items():
        health = info.get("health", {})
        health_status = health.get("status", "unknown")
        latency = health.get("latency_ms", "—")
        last_check = health.get("last_check", "—")
        if last_check != "—":
            last_check = f'<span class="local-time" data-utc="{last_check}">{last_check[11:19]}</span>'

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

        if "skill" in sa:
            status_badge = '<span class="badge unknown">SKILL</span>'
            detail = sa["skill"]
            if len(detail) > 120:
                detail = detail[:120] + "..."
        elif sa.get("active_request"):
            req = sa["active_request"]
            elapsed = int(time.time() - req.get("started_at", time.time()))
            cls = req.get("classification", "?")
            mdl = req.get("model", "?")
            p = req.get("pass", 0)
            status_badge = f'<span class="badge auth">{cls}</span>'
            detail = f"{mdl} — pass {p + 1}, {elapsed}s"
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

    # Summary stats
    session_count = stats.get("sessions", {}).get("active", 0)

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
    margin-bottom: 1.5rem;
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

  .header .overall.ok {{ background: var(--green-bg); color: var(--green); }}
  .header .overall.error {{ background: var(--red-bg); color: var(--red); }}

  /* ── Tabs ─────────────────────────────────────────────── */
  .tab-bar {{
    display: flex;
    gap: 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
  }}

  .tab-btn {{
    background: none;
    border: none;
    color: var(--text-dim);
    font-family: 'Outfit', sans-serif;
    font-size: 0.8125rem;
    font-weight: 400;
    padding: 0.6rem 1.25rem;
    cursor: pointer;
    border-bottom: 2px solid transparent;
    transition: color 0.15s, border-color 0.15s;
  }}

  .tab-btn:hover {{ color: var(--text); }}

  .tab-btn.active {{
    color: var(--blue);
    border-bottom-color: var(--blue);
    font-weight: 600;
  }}

  .tab-panel {{ display: none; }}
  .tab-panel.active {{ display: block; }}

  /* ── Shared ───────────────────────────────────────────── */
  .section {{ margin-bottom: 2rem; }}

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

  tr:last-child td {{ border-bottom: none; }}

  .mono {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8125rem;
  }}

  .dim {{ color: var(--text-dim); }}

  .badge {{
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6875rem;
    font-weight: 600;
    padding: 0.125rem 0.5rem;
    border-radius: 3px;
    letter-spacing: 0.05em;
  }}

  .badge.ok      {{ background: var(--green-bg);  color: var(--green); }}
  .badge.error   {{ background: var(--red-bg);    color: var(--red); }}
  .badge.auth    {{ background: var(--orange-bg);  color: var(--orange); }}
  .badge.unknown {{ background: rgba(139,148,158,0.1); color: var(--text-dim); }}
  .badge.debug   {{ background: rgba(139,148,158,0.1); color: var(--text-dim); }}
  .badge.info    {{ background: rgba(88,166,255,0.1);  color: var(--blue); }}
  .badge.warning {{ background: var(--orange-bg);  color: var(--orange); }}

  .stats-row {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 1rem;
    margin-bottom: 1.5rem;
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

  .stat-card .value.green  {{ color: var(--green); }}
  .stat-card .value.blue   {{ color: var(--blue); }}
  .stat-card .value.purple {{ color: var(--purple); }}

  .footer {{
    text-align: center;
    color: var(--text-dim);
    font-size: 0.75rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
  }}

  /* ── Journal ──────────────────────────────────────────── */
  .journal-filters {{
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-bottom: 0.75rem;
    align-items: center;
  }}

  .journal-filters select {{
    background: var(--surface-2);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.3rem 0.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
  }}

  .journal-filters select:focus {{
    outline: none;
    border-color: var(--blue);
  }}

  .journal-filters label {{
    font-size: 0.7rem;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}

  .journal-scroll {{
    max-height: 520px;
    overflow-y: auto;
    scrollbar-width: thin;
    scrollbar-color: var(--border) transparent;
  }}

  .journal-scroll::-webkit-scrollbar {{ width: 6px; }}
  .journal-scroll::-webkit-scrollbar-track {{ background: transparent; }}
  .journal-scroll::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }}

  .journal-entry {{
    padding: 0.5rem 1rem;
    border-bottom: 1px solid var(--border);
    font-size: 0.8125rem;
    display: grid;
    grid-template-columns: 6.5rem 10rem 1fr;
    gap: 0.75rem;
    align-items: start;
  }}

  .journal-entry:last-child {{ border-bottom: none; }}

  .journal-entry .je-ts {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: var(--text-dim);
    white-space: nowrap;
  }}

  .journal-entry .je-event {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: var(--blue);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }}

  .journal-entry .je-msg {{
    color: var(--text);
    word-break: break-word;
  }}

  .journal-entry .je-ctx {{
    font-size: 0.7rem;
    color: var(--text-dim);
    margin-top: 0.15rem;
  }}

  .journal-pager {{
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.75rem;
    padding: 0.5rem 1rem;
    border-top: 1px solid var(--border);
    font-size: 0.75rem;
    color: var(--text-dim);
  }}

  .journal-pager button {{
    background: var(--surface-2);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.2rem 0.6rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    cursor: pointer;
  }}

  .journal-pager button:disabled {{
    opacity: 0.3;
    cursor: default;
  }}

  .journal-empty {{
    padding: 1.5rem;
    text-align: center;
    color: var(--text-dim);
    font-size: 0.8125rem;
  }}

  /* ── Mobile ──────────────────────────────────────────── */
  @media (max-width: 640px) {{
    body {{
      padding: 1rem 0.75rem;
    }}

    .header {{
      flex-wrap: wrap;
      gap: 0.5rem;
    }}

    .header h1 {{ font-size: 1.2rem; }}

    .header .overall {{
      margin-left: 0;
    }}

    .tab-bar {{
      overflow-x: auto;
      -webkit-overflow-scrolling: touch;
    }}

    .tab-btn {{
      padding: 0.5rem 0.75rem;
      font-size: 0.75rem;
      white-space: nowrap;
    }}

    .stats-row {{
      grid-template-columns: repeat(2, 1fr);
      gap: 0.5rem;
    }}

    .stat-card {{ padding: 0.75rem; }}
    .stat-card .value {{ font-size: 1.25rem; }}

    /* Make tables horizontally scrollable */
    .card {{
      overflow-x: auto;
      -webkit-overflow-scrolling: touch;
    }}

    table {{ font-size: 0.75rem; }}
    th, td {{ padding: 0.4rem 0.5rem; }}
    .mono {{ font-size: 0.7rem; }}

    /* Journal entries: stack vertically on mobile */
    .journal-entry {{
      grid-template-columns: 1fr;
      gap: 0.25rem;
      padding: 0.5rem 0.75rem;
    }}

    .journal-entry .je-ts {{
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }}

    .journal-entry .je-event {{
      white-space: normal;
    }}

    .journal-filters {{
      gap: 0.35rem;
    }}

    .journal-filters select {{
      font-size: 0.7rem;
      padding: 0.25rem 0.35rem;
    }}

    .journal-pager {{
      gap: 0.5rem;
      font-size: 0.7rem;
    }}
  }}
</style>
</head>
<body>

<div class="header">
  <h1>Agent Dashboard</h1>
  <span class="uptime">up {stats['uptime']}</span>
  <span class="overall {overall_class}">{overall_text}</span>{quiet_badge}
</div>

<!-- Tab bar -->
<div class="tab-bar">
  <button class="tab-btn active" onclick="switchTab('overview')">Overview</button>
  <button class="tab-btn" onclick="switchTab('skills')">Skill Runs</button>
  <button class="tab-btn" onclick="switchTab('journal')">Journal</button>
  <button class="tab-btn" onclick="switchTab('infra')">Infrastructure</button>
</div>

<!-- Tab 1: Overview ──────────────────────────────────── -->
<div class="tab-panel active" id="tab-overview">

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

</div>

<!-- Tab 2: Skill Runs ────────────────────────────────── -->
<div class="tab-panel" id="tab-skills">

<div class="section">
  <h2>Recent Skill Runs</h2>
  <div class="card">
    <table>
      <tr><th>Skill</th><th>Status</th><th>Time</th><th>Error</th></tr>
      {run_rows}
    </table>
  </div>
</div>

</div>

<!-- Tab 3: Journal ───────────────────────────────────── -->
<div class="tab-panel" id="tab-journal">

<div class="journal-filters">
  <label>Source</label>
  <select id="jf-source" onchange="switchJournalSource()">
    <option value="">Live</option>
  </select>
  <label>Level</label>
  <select id="jf-level" onchange="loadJournal(1)">
    <option value="">All</option>
    <option value="error">error</option>
    <option value="warning">warning</option>
    <option value="info">info</option>
    <option value="debug">debug</option>
  </select>
  <label>Event</label>
  <select id="jf-event" onchange="loadJournal(1)">
    <option value="">All</option>
  </select>
  <label>Per page</label>
  <select id="jf-perpage" onchange="loadJournal(1)">
    <option value="25">25</option>
    <option value="50" selected>50</option>
    <option value="100">100</option>
  </select>
</div>
<div class="card">
  <div class="journal-scroll" id="journal-scroll">
    <div class="journal-empty">Select the Journal tab to load entries.</div>
  </div>
  <div class="journal-pager" id="journal-pager" style="display:none">
    <button id="jp-prev" onclick="loadJournal(currentPage-1)">&laquo; Prev</button>
    <span id="jp-info"></span>
    <button id="jp-next" onclick="loadJournal(currentPage+1)">Next &raquo;</button>
  </div>
</div>

</div>

<!-- Tab 4: Infrastructure ────────────────────────────── -->
<div class="tab-panel" id="tab-infra">

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

</div>

<div class="footer">
  <span class="local-time" data-utc="{now}">{now}</span>
</div>

<script>
var _journalBootstrap = {journal_json};
var _archiveList = {archive_list_json};
</script>
<script>
// Convert all UTC timestamps to browser-local time
document.querySelectorAll('.local-time').forEach(function(el) {{
  try {{
    var d = new Date(el.dataset.utc);
    if (!isNaN(d)) el.textContent = d.toLocaleString();
  }} catch(_) {{}}
}});
</script>
<script>
(function() {{
  var currentPage = 1;
  window.currentPage = currentPage;
  var knownEvents = new Set();
  var tabNames = ['overview', 'skills', 'journal', 'infra'];
  var activeTab = 'overview';
  var activeSource = '';  // '' = live, 'filename.jsonl' = archive

  /* ── Populate archive dropdown ─────────────────────── */
  (function() {{
    var sel = document.getElementById('jf-source');
    (_archiveList || []).forEach(function(a) {{
      var opt = document.createElement('option');
      opt.value = a.filename;
      opt.textContent = a.date + ' (' + a.lines + ' entries)';
      sel.appendChild(opt);
    }});
  }})();

  function switchJournalSource() {{
    activeSource = document.getElementById('jf-source').value;
    // Reset event filter since archive may have different events
    var evSel = document.getElementById('jf-event');
    evSel.innerHTML = '<option value="">All</option>';
    knownEvents = new Set();
    loadJournal(1);
  }}
  window.switchJournalSource = switchJournalSource;

  /* ── Tab switching ──────────────────────────────────── */
  function switchTab(id) {{
    activeTab = id;
    location.hash = id;
    document.querySelectorAll('.tab-panel').forEach(function(p) {{
      p.classList.remove('active');
    }});
    document.querySelectorAll('.tab-btn').forEach(function(b) {{
      b.classList.remove('active');
    }});
    var panel = document.getElementById('tab-' + id);
    if (panel) panel.classList.add('active');
    var btns = document.querySelectorAll('.tab-btn');
    for (var i = 0; i < tabNames.length; i++) {{
      if (tabNames[i] === id && btns[i]) btns[i].classList.add('active');
    }}
    if (id === 'journal' && !activeSource) {{
      renderJournal(window._journalBootstrap || {{}});
    }}
  }}
  window.switchTab = switchTab;

  // Restore tab from URL hash on load
  var hash = location.hash.replace('#', '');
  if (hash && tabNames.indexOf(hash) !== -1) {{
    switchTab(hash);
  }}

  // Auto-reload every 60s, skip when on journal tab
  setInterval(function() {{
    if (activeTab !== 'journal') location.reload();
  }}, 60000);

  /* ── Journal rendering ─────────────────────────────── */
  function levelBadge(level) {{
    var cls = {{'error':'error','warning':'warning','info':'info','debug':'debug'}}[level] || 'unknown';
    return '<span class="badge ' + cls + '">' + (level || 'error').toUpperCase() + '</span>';
  }}

  function esc(s) {{
    var d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
  }}

  function renderJournal(data) {{
    // Populate event dropdown
    var evSel = document.getElementById('jf-event');
    (data.events || []).forEach(function(ev) {{
      if (!knownEvents.has(ev)) {{
        knownEvents.add(ev);
        var opt = document.createElement('option');
        opt.value = ev;
        opt.textContent = ev;
        evSel.appendChild(opt);
      }}
    }});

    var scroll = document.getElementById('journal-scroll');
    if (!data.entries || data.entries.length === 0) {{
      scroll.innerHTML = '<div class="journal-empty">No journal entries match the current filters.</div>';
      document.getElementById('journal-pager').style.display = 'none';
      return;
    }}

    var html = '';
    data.entries.forEach(function(e) {{
      var tsRaw = e.ts || '';
      var ts = tsRaw;
      try {{
        var d = new Date(tsRaw);
        ts = d.toLocaleTimeString('en-US', {{hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false}});
      }} catch(_) {{ ts = tsRaw.substring(11,19); }}
      var ev = e.event || 'unknown';
      var msg = e.error || '';
      var ctx = [];
      Object.keys(e).forEach(function(k) {{
        if (k !== 'ts' && k !== 'level' && k !== 'event' && k !== 'error')
          ctx.push(k + '=' + String(e[k]).substring(0, 80));
      }});
      var ctxHtml = ctx.length
        ? '<div class="je-ctx">' + esc(ctx.join('  ')) + '</div>' : '';
      html += '<div class="journal-entry">'
        + '<span class="je-ts">' + levelBadge(e.level) + ' ' + esc(ts) + '</span>'
        + '<span class="je-event" title="' + esc(ev) + '">' + esc(ev) + '</span>'
        + '<span class="je-msg">'
        + esc(msg.length > 300 ? msg.substring(0, 300) + '\u2026' : msg)
        + ctxHtml + '</span></div>';
    }});
    scroll.innerHTML = html;
    scroll.scrollTop = 0;

    var pager = document.getElementById('journal-pager');
    pager.style.display = 'flex';
    document.getElementById('jp-info').textContent =
      'Page ' + data.page + ' / ' + data.pages + '  (' + data.total + ' entries)';
    document.getElementById('jp-prev').disabled = (data.page <= 1);
    document.getElementById('jp-next').disabled = (data.page >= data.pages);
  }}

  function loadJournal(page) {{
    page = page || 1;
    currentPage = page;
    window.currentPage = page;
    var level  = document.getElementById('jf-level').value;
    var event  = document.getElementById('jf-event').value;
    var perPg  = document.getElementById('jf-perpage').value;
    var qs = '?page=' + page + '&per_page=' + perPg;
    if (level) qs += '&level=' + encodeURIComponent(level);
    if (event) qs += '&event=' + encodeURIComponent(event);

    var url = activeSource
      ? '/journal/archive/' + encodeURIComponent(activeSource) + qs
      : '/journal' + qs;

    var scroll = document.getElementById('journal-scroll');
    scroll.innerHTML = '<div class="journal-empty">Loading\u2026</div>';

    fetch(url)
      .then(function(r) {{
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.text();
      }})
      .then(function(text) {{
        if (!text) throw new Error('Empty response from server');
        var data = JSON.parse(text);
        window._journalBootstrap = data;
        renderJournal(data);
      }})
      .catch(function(err) {{
        scroll.innerHTML = '<div class="journal-empty">Failed to load: ' + esc(String(err)) + '</div>';
        document.getElementById('journal-pager').style.display = 'none';
      }});
  }}
  window.loadJournal = loadJournal;
}})();
</script>

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


async def handle_cloister(request):
    """Serve the arcanotech cloister Three.js scene."""
    html_path = _CLOISTER_HTML_DIR / "cloister.html"
    try:
        html = html_path.read_text(encoding="utf-8")
        return web.Response(text=html, content_type="text/html")
    except FileNotFoundError:
        return web.Response(text="<h1>cloister not found</h1>", content_type="text/html", status=404)


async def handle_exports(request):
    """Serve the geometry exporter page (bird + cat OBJ download)."""
    html_path = _CLOISTER_HTML_DIR / "exports.html"
    try:
        html = html_path.read_text(encoding="utf-8")
        return web.Response(text=html, content_type="text/html")
    except FileNotFoundError:
        return web.Response(text="<h1>exports.html not found</h1>", content_type="text/html", status=404)


def _check_export_auth(request) -> bool:
    """Return True if the request carries the correct export API key."""
    # Check ?key= query param
    key_param = request.query.get("key", "")
    if key_param and key_param == _export_api_key:
        return True
    # Check Authorization: Bearer <key> header
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer ") and auth_header[7:] == _export_api_key:
        return True
    return False


async def handle_export_bird(request):
    """Return bird.obj geometry; requires API key."""
    if not _check_export_auth(request):
        return web.Response(
            status=401,
            text="401 Unauthorized — provide ?key=<api_key> or Authorization: Bearer <api_key>",
        )
    from agent.obj_export import generate_bird_obj
    obj_text = generate_bird_obj()
    return web.Response(
        body=obj_text.encode("utf-8"),
        content_type="application/octet-stream",
        headers={"Content-Disposition": 'attachment; filename="bird.obj"'},
    )


async def handle_export_cat(request):
    """Return cat.obj geometry; requires API key."""
    if not _check_export_auth(request):
        return web.Response(
            status=401,
            text="401 Unauthorized — provide ?key=<api_key> or Authorization: Bearer <api_key>",
        )
    from agent.obj_export import generate_cat_obj
    obj_text = generate_cat_obj()
    return web.Response(
        body=obj_text.encode("utf-8"),
        content_type="application/octet-stream",
        headers={"Content-Disposition": 'attachment; filename="cat.obj"'},
    )


async def handle_journal(request):
    """Journal entries with filtering and pagination (JSON API)."""
    try:
        journal = getattr(_orchestrator, "error_journal", None) if _orchestrator else None
        if journal is None:
            journal = ErrorJournal()
        level = request.query.get("level")
        event = request.query.get("event")
        try:
            page = max(1, int(request.query.get("page", "1")))
        except ValueError:
            page = 1
        try:
            per_page = max(1, min(200, int(request.query.get("per_page", "50"))))
        except ValueError:
            per_page = 50

        result = journal.query(level=level, event=event, page=page, per_page=per_page)
        return web.json_response(result)
    except Exception as e:
        log.error("journal_endpoint_error", error=str(e))
        return web.json_response(
            {"entries": [], "total": 0, "page": 1, "per_page": 50,
             "pages": 0, "levels": [], "events": [], "error": str(e)},
            status=200,
        )


async def handle_journal_archives(request):
    """List available archive files (JSON API)."""
    try:
        journal = getattr(_orchestrator, "error_journal", None) if _orchestrator else None
        if journal is None:
            journal = ErrorJournal()
        return web.json_response({"archives": journal.list_archives()})
    except Exception as e:
        return web.json_response({"archives": [], "error": str(e)})


async def handle_journal_archive(request):
    """Query a specific archive file (JSON API)."""
    try:
        filename = request.match_info["filename"]
        journal = getattr(_orchestrator, "error_journal", None) if _orchestrator else None
        if journal is None:
            journal = ErrorJournal()
        level = request.query.get("level")
        event = request.query.get("event")
        try:
            page = max(1, int(request.query.get("page", "1")))
        except ValueError:
            page = 1
        try:
            per_page = max(1, min(200, int(request.query.get("per_page", "50"))))
        except ValueError:
            per_page = 50

        result = journal.query_archive(
            filename, level=level, event=event, page=page, per_page=per_page,
        )
        return web.json_response(result)
    except Exception as e:
        return web.json_response(
            {"entries": [], "total": 0, "page": 1, "per_page": 50,
             "pages": 0, "levels": [], "events": [], "error": str(e)},
        )


async def start_status_server(port: int = 8765):
    """Start the status HTTP server and background health checks."""
    global _health_check_task

    # Load (or create) the OBJ export API key
    api_key = _load_or_create_export_key()
    log.info(
        "export_api_key_loaded",
        key=api_key,
        key_file=str(_EXPORT_KEY_FILE),
    )
    print(f"[status_server] OBJ export API key: {api_key}", flush=True)

    app = web.Application()
    app.router.add_get("/", handle_index)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/stats", handle_stats)
    app.router.add_get("/costs", handle_costs)
    app.router.add_get("/skills", handle_skills)
    app.router.add_get("/journal", handle_journal)
    app.router.add_get("/journal/archives", handle_journal_archives)
    app.router.add_get("/journal/archive/{filename}", handle_journal_archive)
    app.router.add_get("/cloister", handle_cloister)
    app.router.add_get("/exports", handle_exports)
    app.router.add_get("/export/bird.obj", handle_export_bird)
    app.router.add_get("/export/cat.obj", handle_export_cat)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()

    # Start background health check loop
    _health_check_task = asyncio.create_task(_health_check_loop())

    log.info("status_server_started", port=port, url=f"http://0.0.0.0:{port}")
    return runner
