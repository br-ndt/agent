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

Decide if {bot_name} should respond to this message. Respond with exactly one word:

- YES: The message asks {bot_name} to do something, asks {bot_name} a question, or needs {bot_name}'s input.
- REACT: The message mentions {bot_name} by name in ANY way — compliments, jokes, talking about {bot_name}, acknowledging {bot_name}, thanking {bot_name}, or referencing something {bot_name} did. If {bot_name}'s name appears and it's not a direct request, REACT. When in doubt between NO and REACT, choose REACT.
- NO: The message does NOT mention {bot_name} at all, is clearly for another bot with no reference to {bot_name}, or is completely unrelated general chatter where {bot_name} is not referenced.
- WAIT: The message tells {bot_name} to hold off, wait for someone else, or stand by.

Key distinction: if the message mentions {bot_name} by name, it should almost never be NO.
"calne is such a fun fella" → REACT. "calne did a good job" → REACT. "thanks calne" → REACT. "calne can you fix this" → YES. "hey does anyone know about X" → NO.

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
        is_role_mentioned: bool = False,
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

        # Fast path: direct @mention with no other bots mentioned — always process.
        # This guarantees the bot responds when explicitly addressed.
        if is_mentioned and not self._mentions_other_entity(msg.text):
            return True, "mentioned"

        # Fast path: role-mentioned with no other bots — also always process.
        if is_role_mentioned and not self._mentions_other_entity(msg.text):
            return True, "mentioned"

        # Not mentioned at all and name not in text — skip without LLM
        if not is_mentioned and not is_role_mentioned and self.bot_name.lower() not in msg.text.lower():
            return False, "skip"

        # Ambiguous: name in text but not @mentioned, or multiple bots mentioned.
        # Use LLM to decide.
        try:
            if isinstance(self.other_bot_names, dict):
                other_bots_str = ", ".join(self.other_bot_names.values())
            elif isinstance(self.other_bot_names, list):
                other_bots_str = ", ".join(str(b) for b in self.other_bot_names)
            else:
                other_bots_str = "unknown"
            other_bots_str = other_bots_str or "unknown"

            role_hint = ""
            if is_role_mentioned:
                role_hint = (
                    f"\n\nIMPORTANT: {self.bot_name}'s role was pinged in this message. "
                    f"This means the sender intended to reach {self.bot_name} (among others). "
                    f"Lean toward YES unless the message is clearly only for another bot."
                )

            prompt = RELEVANCE_PROMPT.format(
                bot_name=self.bot_name,
                other_bots=other_bots_str,
                sender=msg.sender_name,
                message=msg.text[:500],
            ) + role_hint

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
            elif answer.startswith("REACT"):
                log.debug("relevance_react", sender=msg.sender_name,
                          text_preview=msg.text[:80])
                return True, "react"
            elif answer.startswith("YES"):
                log.debug("relevance_yes", sender=msg.sender_name)
                return True, "process"
            else:
                log.debug("relevance_no", sender=msg.sender_name,
                           text_preview=msg.text[:80])
                return False, "skip"

        except Exception as e:
            log.warning("relevance_filter_error", error=str(e))
            # On error, fall back to processing if mentioned, skip otherwise
            if is_mentioned or is_role_mentioned:
                return True, "error_fallback"
            return False, "skip"

    def _mentions_other_entity(self, text: str) -> bool:
        """Check if the message mentions any other bot by ID."""
        ids = []
        if isinstance(self.other_bot_names, dict):
            ids = list(self.other_bot_names.keys())
        elif isinstance(self.other_bot_names, list):
            ids = self.other_bot_names

        for entry in ids:
            # Strip platform prefix if present (e.g. "discord:123" -> "123")
            bot_id = entry.split(":", 1)[-1] if ":" in str(entry) else str(entry)
            if f"<@{bot_id}>" in text or f"<@!{bot_id}>" in text:
                return True
        return False