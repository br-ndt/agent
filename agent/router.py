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
        if tier == "admin" and msg.text.upper() in ("RESTART", "HEALME", "STATUS"):
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
            log.info("router_got_response", platform=msg.platform, response_len=len(response))
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
