"""Router — access control tiers and message dispatch."""

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

    def register_adapter(self, name: str, adapter: BaseAdapter):
        self.adapters[name] = adapter

    def get_tier(self, msg: IncomingMessage) -> str:
        """Determine the user's access tier."""
        composite_id = f"{msg.platform}:{msg.sender_id}"

        if composite_id in self.config.admin_ids:
            return "admin"
        if composite_id in self.config.trusted_ids:
            return "trusted"

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
        upper = msg.text.strip().upper()
        if tier == "admin" and (
            upper in ("RESTART", "HEALME", "STATUS", "COSTS", "SKILLS")
            or upper.startswith("SKILL ")
            or upper == "RELOAD SKILLS"
        ):
            await self._handle_admin_command(msg)
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

            response = await self.orchestrator.handle(
                session_id=f"{msg.platform}:{msg.sender_id}",
                user_msg=msg.text,
                reply_fn=reply_fn,
                tier=tier,
                tier_route=tier_route,
            )
            log.info(
                "router_got_response", platform=msg.platform, response_len=len(response)
            )
            await adapter.send(msg.chat_id, response)
        except Exception as e:
            log.error("orchestrator_error", error=str(e), exc_info=True)
            await adapter.send(msg.chat_id, f"Sorry, something went wrong: {e}")

    async def _handle_admin_command(self, msg: IncomingMessage):
        adapter = self.adapters.get(msg.platform)
        if not adapter:
            return

        cmd = msg.text.upper()
        if cmd == "STATUS":
            session_count = len(self.orchestrator.sessions)
            await adapter.send(
                msg.chat_id,
                f"Status: running\nActive sessions: {session_count}",
            )
        elif cmd == "HEALME":
            for s in self.orchestrator.sessions.values():
                await s.clear()
            self.orchestrator.sessions.clear()
            await adapter.send(msg.chat_id, "All sessions cleared.")
        elif cmd == "RESTART":
            await adapter.send(msg.chat_id, "Restarting...")
            raise SystemExit(0)
        elif cmd == "COSTS":
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
                triggers = ", ".join(f"`{t}`" for t in s.triggers) if s.triggers else "none"
                steps_info = f" ({len(s.steps)} steps)" if s.has_steps else ""
                lines.append(f"• **{s.name}**{steps_info} → {s.subagent} | Triggers: {triggers}")
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
        elif cmd.startswith("SKILL "):
            await self._handle_skill_subcommand(msg, adapter)
    

    async def _handle_skill_subcommand(self, msg: IncomingMessage, adapter: BaseAdapter):
        parts = msg.text.strip().split(maxsplit=2)
        # parts[0] = "SKILL", parts[1] = subcommand, parts[2] = skill name
        if len(parts) < 3:
            await adapter.send(msg.chat_id, "Usage: SKILL INFO|APPROVE|REJECT|DISABLE|ENABLE|HISTORY|RUN <name>")
            return

        subcmd = parts[1].upper()
        name = parts[2].strip()
        registry = self.orchestrator.skill_registry

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
                await adapter.send(msg.chat_id, f"✅ Skill '{name}' approved and activated.")
            else:
                await adapter.send(msg.chat_id, f"Proposed skill '{name}' not found.")

        elif subcmd == "REJECT":
            if registry.reject(name):
                await adapter.send(msg.chat_id, f"❌ Skill '{name}' rejected and deleted.")
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
                    lines.append(f"• {r['status']} at {r['started_at']} ({r.get('error') or 'ok'})")
                await adapter.send(msg.chat_id, "\n".join(lines))
            else:
                await adapter.send(msg.chat_id, f"No run history for '{name}'.")

        elif subcmd == "RUN":
            skill = registry.get(name)
            if skill and skill.subagent in self.orchestrator.subagents:
                run = await self.orchestrator.skill_executor.execute(
                    skill=skill,
                    task_context="Manual trigger via SKILL RUN command",
                    subagent_runner=self.orchestrator.subagents[skill.subagent],
                    session_id="admin-manual",
                    progress_fn=reply_fn,
                )
                result = self.orchestrator.skill_executor.format_run_result(skill, run)
                await adapter.send(msg.chat_id, result)
            else:
                await adapter.send(msg.chat_id, f"Skill '{name}' not found or subagent unavailable.")

        else:
            await adapter.send(msg.chat_id, f"Unknown: SKILL {subcmd}. Try INFO|APPROVE|REJECT|DISABLE|ENABLE|HISTORY|RUN")