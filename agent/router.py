"""Router — access control tiers and message dispatch."""

import asyncio
import time

import structlog

from agent.adapters.base import IncomingMessage, BaseAdapter
from agent.config import Config
from agent.orchestrator import Orchestrator

log = structlog.get_logger()


class Router:
    def __init__(self, config: Config, orchestrator: Orchestrator):
        self.config = config
        self.orchestrator = orchestrator
        self.adapters: dict[str, BaseAdapter] = {}

        # Global quiet mode — all adapters respect this
        self._quiet = False
        self._quiet_since: float | None = None  # time.monotonic() when enabled
        self._quiet_lock = asyncio.Lock()

    def register_adapter(self, name: str, adapter: BaseAdapter):
        self.adapters[name] = adapter

    def get_tier(self, msg: IncomingMessage) -> str:
        """Determine the user's access tier."""
        composite_id = f"{msg.platform}:{msg.sender_id}"

        if composite_id in self.config.admin_ids:
            return "admin"
        if composite_id in self.config.trusted_ids:
            return "trusted"

        # Bots: trusted_bots get "trusted", remaining other_bots get "bot"
        if composite_id in self.config.trusted_bots:
            return "trusted"
        other_bots = self.config.other_bots
        if other_bots:
            if isinstance(other_bots, dict):
                bot_ids = set(other_bots.keys())
            else:
                bot_ids = {str(b) for b in other_bots}
            if composite_id in bot_ids:
                return "bot"

        if composite_id in self.config.basic_ids:
            return "basic"

        # CLI is always admin (it's local)
        if msg.platform == "cli":
            return "admin"

        return "unknown"

    async def handle_message(self, msg: IncomingMessage):
        """Route an incoming message through access control to the orchestrator."""
        tier = self.get_tier(msg)

        log.info(
            "message_routed",
            platform=msg.platform,
            sender=msg.sender_name,
            tier=tier,
            text_preview=msg.text[:80],
        )

        # Drop unknown users silently
        if tier == "unknown":
            log.debug("unknown_user_dropped", sender_id=msg.sender_id)
            return

        # Admin commands
        raw_text = msg.text.split("[System Context:")[0].strip()
        upper = raw_text.upper()

        if tier == "admin" and (
            upper in ("RESTART", "HEALME", "STATUS", "COST", "COSTS", "SKILLS", "DIAGNOSE", "ERRORS", "QUIET", "WAKE")
            or upper.startswith("SKILL ")
            or upper.startswith("DIAGNOSE ")
            or upper.startswith("CANCEL ")
            or upper == "CANCEL"
            or upper == "RELOAD SKILLS"
        ):
            clean_msg = IncomingMessage(
                platform=msg.platform,
                sender_id=msg.sender_id,
                sender_name=msg.sender_name,
                text=raw_text,
                chat_id=msg.chat_id,
                attachments=msg.attachments,
            )
            await self._handle_admin_command(clean_msg)
            return

        # Quiet mode — block non-admin messages before any LLM work.
        # CLI always bypasses (local admin). Admin commands already returned above.
        if msg.platform != "cli" and tier != "admin":
            async with self._quiet_lock:
                quiet_now = self._quiet
            if quiet_now:
                log.info("quiet_mode_rejected", sender=msg.sender_name, platform=msg.platform)
                return

        # Get the adapter for replies
        adapter = self.adapters.get(msg.platform)
        if not adapter:
            log.error("no_adapter_for_platform", platform=msg.platform)
            return

        # Delegate to orchestrator
        async def reply_fn(text: str):
            await adapter.send(msg.chat_id, text)

        log.info("router_calling_orchestrator", platform=msg.platform)
        try:
            tier_route = self.config.tier_routing.get(tier, {})
            
            # Extract images and audio from attachments
            images = None
            audio = None
            if msg.attachments:
                images = []
                audio = []
                for att in msg.attachments:
                    mime = att.get("mime_type", "")
                    entry = {
                        "data": att.get("data"),
                        "mime_type": mime,
                        "filename": att.get("filename", "file"),
                    }
                    if mime.startswith("image/"):
                        images.append(entry)
                    elif mime.startswith("audio/"):
                        audio.append(entry)
                if not images:
                    images = None
                if not audio:
                    audio = None

            # Call orchestrator (returns dict with text + images)
            result = await self.orchestrator.handle(
                session_id=f"{msg.platform}:{msg.chat_id}",
                user_msg=msg.text,
                reply_fn=reply_fn,
                tier=tier,
                tier_route=tier_route,
                images=images,
                audio=audio,
            )
            
            # Handle response - orchestrator now returns dict
            response_text = result.get('text', '')
            generated_images = result.get('images', [])
            
            log.info(
                "router_got_response", 
                platform=msg.platform, 
                response_len=len(response_text),
                image_count=len(generated_images)
            )
            
            # Send text response (only if non-empty)
            if response_text:
                await adapter.send(msg.chat_id, response_text)
            
            # Send generated images if any
            if generated_images:
                if hasattr(adapter, 'send_images'):
                    await adapter.send_images(msg.chat_id, generated_images)
                else:
                    log.warning(
                        "adapter_missing_send_images", 
                        platform=msg.platform,
                        image_count=len(generated_images)
                    )
                    
        except Exception as e:
            log.error("orchestrator_error", error=str(e), exc_info=True)
            await adapter.send(msg.chat_id, f"Sorry, something went wrong: {e}")

    async def _handle_admin_command(self, msg: IncomingMessage):
        adapter = self.adapters.get(msg.platform)
        if not adapter:
            return

        cmd = msg.text.split("[System Context:")[0].strip().upper()
        if cmd == "STATUS":
            lines = [f"**Status**: running"]
            # Show quiet mode state
            if self._quiet and self._quiet_since is not None:
                elapsed = int(time.monotonic() - self._quiet_since)
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
                lines.append(f"**Mode**: QUIET for {' '.join(parts)} (only WAKE from admins)")
            lines.append(f"**Sessions**: {len(self.orchestrator.sessions)}")

            # Active delegations
            if self.orchestrator._active_delegations:
                lines.append("\n**Active delegations**:")
                for agent, info in self.orchestrator._active_delegations.items():
                    started = info.get("started_at", 0)
                    elapsed = f"{int(time.time() - started)}s" if started else "?"
                    task = info.get("task", "?")[:100]
                    lines.append(f"  `{agent}` ({elapsed}): {task}")
            else:
                lines.append("**Active delegations**: none")

            # Background skill tasks
            if self.orchestrator._skill_tasks:
                lines.append("\n**Background skills**:")
                for sid, (task, _) in self.orchestrator._skill_tasks.items():
                    status = self.orchestrator.skill_executor.get_status(sid)
                    lines.append(f"  session `{sid[:12]}...`: {status or 'running'}")
            else:
                lines.append("**Background skills**: none")

            await adapter.send(msg.chat_id, "\n".join(lines))
        elif cmd == "HEALME":
            for s in self.orchestrator.sessions.values():
                await s.clear()
            self.orchestrator.sessions.clear()
            await adapter.send(msg.chat_id, "All sessions cleared.")
        elif cmd == "RESTART":
            await adapter.send(msg.chat_id, "Restarting...")
            raise SystemExit(0)
        elif cmd == "COST" or cmd == "COSTS":
            if (
                hasattr(self.orchestrator, "cost_tracker")
                and self.orchestrator.cost_tracker
            ):
                summary = await self.orchestrator.cost_tracker.get_today()
                lines = [
                    f"Today's costs: ${summary['total_cost_usd']:.4f} ({summary['total_calls']} calls)"
                ]
                for p in summary.get("by_provider", []):
                    lines.append(
                        f"  {p['provider']}:{p['model']} — ${p['cost_usd']:.4f} ({p['calls']} calls)"
                    )
                for a in summary.get("by_agent", []):
                    lines.append(
                        f"  agent:{a['agent']} — ${a['cost_usd']:.4f} ({a['calls']} calls, avg {a['avg_duration_ms']}ms)"
                    )
                await adapter.send(msg.chat_id, "\n".join(lines))
            else:
                await adapter.send(msg.chat_id, "Cost tracker not available.")
        elif cmd == "SKILLS":
            active = self.orchestrator.skill_registry.list_active()
            proposed = self.orchestrator.skill_registry.list_proposed()
            lines = ["**Active Skills:**"]
            for s in active:
                triggers = (
                    ", ".join(f"`{t}`" for t in s.triggers) if s.triggers else "none"
                )
                steps_info = f" ({len(s.steps)} steps)" if s.has_steps else ""
                lines.append(
                    f"• **{s.name}**{steps_info} → {s.subagent} | Triggers: {triggers}"
                )
            if proposed:
                lines.append("\n**Proposed (awaiting approval):**")
                for s in proposed:
                    lines.append(f"• **{s.name}** → {s.subagent}")
            text = "\n".join(lines) if active or proposed else "No skills configured."
            await adapter.send(msg.chat_id, text)
        elif cmd == "RELOAD SKILLS":
            self.orchestrator.skill_registry.reload()
            self.orchestrator.skills = self.orchestrator.skill_registry.list_active()
            count = len(self.orchestrator.skills)
            await adapter.send(msg.chat_id, f"♻️ Reloaded {count} skill(s).")
        elif cmd == "ERRORS":
            errors = self.orchestrator.error_journal.recent(20)
            if not errors:
                await adapter.send(msg.chat_id, "No recent errors.")
            else:
                lines = [f"**Last {len(errors)} errors:**"]
                for e in errors:
                    ts = e.get("ts", "?")[:16]
                    lines.append(f"`{ts}` **{e.get('event', '?')}**: {e.get('error', '?')[:120]}")
                await adapter.send(msg.chat_id, "\n".join(lines))
        elif cmd == "DIAGNOSE":
            summary = self.orchestrator.diagnostic_store.summary()
            await adapter.send(msg.chat_id, summary)
        elif cmd.startswith("DIAGNOSE "):
            filename = msg.text.split(None, 1)[1].strip()
            report = self.orchestrator.diagnostic_store.read_report(filename)
            if report:
                # Truncate for chat
                if len(report) > 1800:
                    report = report[:1800] + "\n\n... (truncated)"
                await adapter.send(msg.chat_id, report)
            else:
                await adapter.send(msg.chat_id, f"Report not found: `{filename}`")
        elif cmd == "CANCEL" or cmd.startswith("CANCEL "):
            parts = msg.text.strip().split(maxsplit=1)
            if len(parts) < 2:
                # CANCEL with no args — show what can be cancelled
                lines = []
                if self.orchestrator._active_delegations:
                    lines.append("**Active delegations** (use `CANCEL <agent>`):")
                    for agent in self.orchestrator._active_delegations:
                        lines.append(f"  `{agent}`")
                if self.orchestrator._skill_tasks:
                    lines.append("**Background skills** (use `CANCEL SKILL <session>`):")
                    for sid in self.orchestrator._skill_tasks:
                        lines.append(f"  `{sid[:20]}`")
                if not lines:
                    lines.append("Nothing running to cancel.")
                await adapter.send(msg.chat_id, "\n".join(lines))
            else:
                target = parts[1].strip().lower()

                # Cancel a delegation by agent name
                if target in self.orchestrator._active_delegations:
                    self.orchestrator._active_delegations.pop(target)
                    await adapter.send(msg.chat_id, f"Cleared `{target}` from active delegations.")
                # Cancel all delegations
                elif target == "all":
                    count = len(self.orchestrator._active_delegations)
                    self.orchestrator._active_delegations.clear()
                    # Also cancel background skills
                    for sid, (task, _) in list(self.orchestrator._skill_tasks.items()):
                        task.cancel()
                    self.orchestrator._skill_tasks.clear()
                    await adapter.send(msg.chat_id, f"Cancelled {count} delegation(s) and all background skills.")
                # Cancel a background skill by session prefix
                elif target.startswith("skill"):
                    skill_parts = parts[1].strip().split(maxsplit=1)
                    if len(skill_parts) > 1:
                        prefix = skill_parts[1].strip()
                        cancelled = False
                        for sid, (task, _) in list(self.orchestrator._skill_tasks.items()):
                            if sid.startswith(prefix):
                                task.cancel()
                                self.orchestrator._skill_tasks.pop(sid, None)
                                await adapter.send(msg.chat_id, f"Cancelled background skill for session `{sid[:20]}`.")
                                cancelled = True
                                break
                        if not cancelled:
                            await adapter.send(msg.chat_id, f"No background skill matching `{prefix}`.")
                    else:
                        await adapter.send(msg.chat_id, "Usage: `CANCEL SKILL <session_prefix>`")
                else:
                    await adapter.send(msg.chat_id, f"Nothing found for `{target}`. Use `CANCEL` to see options.")
        elif cmd == "QUIET":
            async with self._quiet_lock:
                self._quiet = True
                self._quiet_since = time.monotonic()
            await adapter.send(msg.chat_id, "Going quiet. Only `WAKE` from admins will be heard.")
            log.info("quiet_mode_enabled", by=msg.sender_id)
        elif cmd == "WAKE":
            async with self._quiet_lock:
                self._quiet = False
                self._quiet_since = None
            await adapter.send(msg.chat_id, "Back online. Listening to all messages.")
            log.info("quiet_mode_disabled", by=msg.sender_id)
        elif cmd.startswith("SKILL "):
            await self._handle_skill_subcommand(msg, adapter)

    async def _handle_skill_subcommand(
        self, msg: IncomingMessage, adapter: BaseAdapter
    ):
        parts = msg.text.strip().split(maxsplit=2)
        if len(parts) < 2:
            await adapter.send(
                msg.chat_id,
                "Usage: SKILL STATUS | SKILL INFO|APPROVE|REJECT|DISABLE|ENABLE|HISTORY|RUN <name>",
            )
            return

        subcmd = parts[1].upper()
        registry = self.orchestrator.skill_registry

        # STATUS doesn't require a skill name
        if subcmd == "STATUS":
            executor = self.orchestrator.skill_executor
            if not executor.active_runs:
                await adapter.send(msg.chat_id, "No skills currently running.")
                return
            lines = []
            for sid, run in executor.active_runs.items():
                status = executor.get_status(sid)
                if status:
                    lines.append(status)
            await adapter.send(msg.chat_id, "\n\n".join(lines) or "No active runs.")
            return

        if len(parts) < 3:
            await adapter.send(
                msg.chat_id,
                "Usage: SKILL STATUS | SKILL INFO|APPROVE|REJECT|DISABLE|ENABLE|HISTORY|RUN <name>",
            )
            return

        name = parts[2].strip()

        async def reply_fn(text: str):
            await adapter.send(msg.chat_id, text)

        if subcmd == "INFO":
            skill = registry.get(name)
            if skill:
                info = f"**{skill.name}** v{skill.version}\n"
                info += f"Status: {skill.status} | Agent: {skill.subagent}\n"
                info += f"Description: {skill.description}\n"
                if skill.triggers:
                    info += f"Triggers: {', '.join(skill.triggers)}\n"
                if skill.has_steps:
                    info += f"Steps: {' → '.join(s.id for s in skill.steps)}\n"
                if skill.tags:
                    info += f"Tags: {', '.join(skill.tags)}\n"
                await adapter.send(msg.chat_id, info)
            else:
                await adapter.send(msg.chat_id, f"Skill '{name}' not found.")

        elif subcmd == "APPROVE":
            if registry.approve(name):
                self.orchestrator.skills = registry.list_active()
                await adapter.send(
                    msg.chat_id, f"✅ Skill '{name}' approved and activated."
                )
            else:
                await adapter.send(msg.chat_id, f"Proposed skill '{name}' not found.")

        elif subcmd == "REJECT":
            if registry.reject(name):
                await adapter.send(
                    msg.chat_id, f"❌ Skill '{name}' rejected and deleted."
                )
            else:
                await adapter.send(msg.chat_id, f"Proposed skill '{name}' not found.")

        elif subcmd == "DISABLE":
            if registry.disable(name):
                self.orchestrator.skills = registry.list_active()
                await adapter.send(msg.chat_id, f"Skill '{name}' disabled.")
            else:
                await adapter.send(msg.chat_id, f"Skill '{name}' not found.")

        elif subcmd == "ENABLE":
            if registry.enable(name):
                self.orchestrator.skills = registry.list_active()
                await adapter.send(msg.chat_id, f"Skill '{name}' re-enabled.")
            else:
                await adapter.send(msg.chat_id, f"Skill '{name}' not found.")

        elif subcmd == "HISTORY":
            runs = registry.get_run_history(name)
            if runs:
                lines = [f"**Recent runs for {name}:**"]
                for r in runs[:5]:
                    lines.append(
                        f"• {r['status']} at {r['started_at']} ({r.get('error') or 'ok'})"
                    )
                await adapter.send(msg.chat_id, "\n".join(lines))
            else:
                await adapter.send(msg.chat_id, f"No run history for '{name}'.")

        elif subcmd == "RUN":
            skill = registry.get(name)
            if skill:
                run = await self.orchestrator.skill_executor.execute(
                    skill=skill,
                    task_context="Manual trigger via SKILL RUN command",
                    subagent_runners=self.orchestrator.subagents,
                    session_id="admin-manual",
                )
                result = self.orchestrator.skill_executor.format_run_result(skill, run)
                await adapter.send(msg.chat_id, result)
            else:
                await adapter.send(
                    msg.chat_id, f"Skill '{name}' not found."
                )

        else:
            await adapter.send(
                msg.chat_id,
                f"Unknown: SKILL {subcmd}. Try STATUS|INFO|APPROVE|REJECT|DISABLE|ENABLE|HISTORY|RUN",
            )