"""Relevance filter — uses a cheap LLM to decide if a message is for this bot.

Instead of brittle regex patterns, we ask Gemini Flash (or whatever the
cheapest provider is) a simple yes/no question. Costs fractions of a penny
per check and handles natural language properly.

Fast path: DMs and direct @mentions skip the LLM check entirely.
The LLM only runs for ambiguous channel messages.
"""

import structlog

from agent.adapters.base import IncomingMessage
from agent.providers.base import BaseProvider

log = structlog.get_logger()

RELEVANCE_PROMPT = """You are a message router for a Discord bot named "{bot_name}".

Your ONLY job: decide if the message is ASKING {bot_name} to do something or DIRECTLY talking to {bot_name}.

Respond with exactly one word: YES, NO, or WAIT.

- YES: The message directly asks {bot_name} to do something or is addressed to {bot_name}.
- NO: The message just mentions {bot_name} in passing, is talking ABOUT {bot_name}, is for another bot, or is general chatter. If in doubt, say NO.
- WAIT: The message tells {bot_name} to hold off, wait for someone else, or stand by.

IMPORTANT: Talking ABOUT {bot_name} is NOT the same as talking TO {bot_name}. "calne responded" is NO. "calne do this" is YES.

Other bots in this channel: {other_bots}

Message from {sender}: {message}"""


class RelevanceFilter:
    def __init__(
        self,
        bot_name: str,
        bot_id: str,
        provider: BaseProvider,
        model: str = "gemini-2.5-flash",
        other_bot_names: dict[str, str] | None = None,
    ):
        """
        Args:
            bot_name: The bot's display name (e.g. "calne")
            bot_id: The bot's Discord user ID
            provider: Cheap LLM provider for classification
            model: Model to use (should be fast and cheap)
            other_bot_names: Mapping of bot_id -> display_name for other bots
        """
        self.bot_name = bot_name
        self.bot_id = bot_id
        self.provider = provider
        self.model = model
        self.other_bot_names = other_bot_names or {}

    async def is_relevant(
        self,
        msg: IncomingMessage,
        is_dm: bool = False,
        is_mentioned: bool = False,
    ) -> tuple[bool, str]:
        """Check if a message is relevant to this bot.

        Returns (should_process, action) where action is one of:
          "process" — run the orchestrator normally
          "skip" — ignore the message entirely
          "wait" — acknowledge and stand by
          "dm" — always process (DM)
          "mentioned" — always process (direct mention, no ambiguity)
        """
        # Fast path: DMs always relevant
        if is_dm:
            return True, "dm"

        # Fast path: direct @mention with no other bots mentioned — skip LLM
        if is_mentioned and not self._mentions_other_entity(msg.text):
            return True, "mentioned"

        # Not mentioned at all and no other ambiguity — skip
        if not is_mentioned and self.bot_name.lower() not in msg.text.lower():
            return False, "skip"

        # Ambiguous: mentioned alongside other bots, or name in text
        # Use LLM to classify
        try:
            other_bots_str = ", ".join(
                self.other_bot_names.values()
            ) if self.other_bot_names else "unknown"

            prompt = RELEVANCE_PROMPT.format(
                bot_name=self.bot_name,
                other_bots=other_bots_str,
                sender=msg.sender_name,
                message=msg.text[:500],
            )

            response = await self.provider.complete(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                max_tokens=5,
                temperature=0.0,
            )

            answer = response.content.strip().upper()

            if answer.startswith("WAIT"):
                log.info("relevance_wait", sender=msg.sender_name,
                         text_preview=msg.text[:80])
                return True, "wait"
            elif answer.startswith("YES"):
                log.debug("relevance_yes", sender=msg.sender_name)
                return True, "process"
            else:
                log.debug("relevance_no", sender=msg.sender_name,
                           text_preview=msg.text[:80])
                return False, "skip"

        except Exception as e:
            log.warning("relevance_filter_error", error=str(e))
            return True, "error_fallback"

    def _mentions_other_entity(self, text: str) -> bool:
        """Check if the message mentions any other bot by ID."""
        for bot_id in self.other_bot_names:
            if f"<@{bot_id}>" in text or f"<@!{bot_id}>" in text:
                return True
        return False