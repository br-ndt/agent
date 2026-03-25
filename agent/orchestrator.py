"""Orchestrator — main agent that can delegate to subagents."""

import asyncio
import re
import time
from typing import Callable, Awaitable

import structlog

from agent.config import Config
from agent.providers.base import BaseProvider
from agent.session_store import SessionStore, PersistentSession
from agent.subagent_runner import SubagentRunner

log = structlog.get_logger()


class Orchestrator:
    def __init__(
        self,
        provider: BaseProvider,
        config: Config,
        subagent_runners: dict[str, SubagentRunner],
        providers: dict[str, BaseProvider] | None = None,
        session_store: SessionStore | None = None,
    ):
        self.provider = provider
        self.providers = providers or {}
        self.config = config
        self.subagents = subagent_runners
        self.sessions: dict[str, PersistentSession] = {}
        self.session_store = session_store

        # Build the system prompt with available subagent names
        agent_list = ", ".join(self.subagents.keys()) if self.subagents else "none"
        self.system_prompt = config.orchestrator_system_prompt.replace(
            "{subagent_list}", agent_list
        )
        self.system_prompt = (
            "You are a text-only assistant. NEVER use tools. NEVER attempt to write files, "
            "execute code, or use any tools. When a task requires code, files, or commands, "
            "delegate it to a subagent. Just respond with text.\n\n"
            + self.system_prompt
        )

        # Summarizer: uses cheapest available model
        self._summarizer = self._make_summarizer()

    async def handle(
        self,
        session_id: str,
        user_msg: str,
        reply_fn: Callable[[str], Awaitable[None]],
        tier: str = "admin",
        tier_route: dict[str, str] | None = None,
    ) -> str:
        """Handle a user message. May delegate to subagents."""
        # Resolve provider+model for this tier
        route = tier_route or {}
        provider_name = route.get("provider", self.config.orchestrator_provider)
        model = route.get("model", self.config.orchestrator_model)

        # Look up the provider, fall back to default
        provider = self.providers.get(provider_name, self.provider)
        log.info("orchestrator_routing", 
                 provider_name=provider_name, 
                 model=model, 
                 provider_type=type(provider).__name__,
                 tier=tier,
                 tier_route=tier_route,
                 providers_keys=list(self.providers.keys()))
        
        if session_id not in self.sessions:
            session = PersistentSession(
                session_id=session_id,
                store=self.session_store,
                summarizer=self._summarizer,
            )
            self.sessions[session_id] = session
        else:
            session = self.sessions[session_id]

        await session.ensure_loaded()

        history = self.sessions[session_id]
        await session.add_message("user", user_msg)

        start = time.monotonic()

        # First pass: let the orchestrator decide what to do
        response = await provider.complete(
            messages=session.get_messages_for_llm(),
            system=self.system_prompt,
            model=model,
            max_tokens=self.config.orchestrator_max_tokens,
        )
        log.info("orchestrator_raw_response", content=response.content[:500])

        elapsed = time.monotonic() - start
        log.info("orchestrator_response", elapsed=f"{elapsed:.1f}s", usage=response.usage)

        # Check for delegation blocks
        delegations = _parse_delegations(response.content)

        if not delegations:
            # Direct response — no delegation needed
            final = _strip_thinking(response.content)
            await session.add_message("assistant", final)
            return final

        # Delegation mode: fan out to subagents in parallel
        valid_delegations = [
            d for d in delegations if d["agent"] in self.subagents
        ]

        if not valid_delegations:
            # Orchestrator tried to delegate to unknown agents — just return raw
            final = _strip_thinking(response.content)
            await session.add_message("assistant", final)
            return final

        agent_names = [d["agent"] for d in valid_delegations]
        await reply_fn(f"Working on it — delegating to {', '.join(agent_names)}...")

        # Run subagents concurrently
        tasks = [
            self.subagents[d["agent"]].run(d["task"], context=user_msg)
            for d in valid_delegations
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Format results for synthesis
        result_parts = []
        for d, result in zip(valid_delegations, results):
            if isinstance(result, Exception):
                result_parts.append(
                    f"[{d['agent']}] ERROR: {type(result).__name__}: {result}"
                )
                log.error("subagent_failed", agent=d["agent"], error=str(result))
            else:
                result_parts.append(f"[{d['agent']}] Result:\n{result}")

        synthesis_msg = (
            "Here are the results from the subagents you delegated to:\n\n"
            + "\n\n---\n\n".join(result_parts)
            + "\n\nPlease synthesize these into a clear, final response for the user."
        )

        # Second pass: synthesize
        await session.add_message("assistant", response.content)
        await session.add_message("user", synthesis_msg)

        final_response = await provider.complete(
            messages=session.get_messages_for_llm(),
            system=self.system_prompt,
            model=model,
            max_tokens=self.config.orchestrator_max_tokens,
        )

        final = _strip_thinking(final_response.content)
        await session.add_message("assistant", final)

        return final


    def _make_summarizer(self):
        """Create a cheap summarizer function for session compression."""
        # Prefer Google (cheapest), fall back to default provider
        summarize_provider = self.providers.get("google", self.provider)
        summarize_model = "gemini-2.5-flash" if "google" in self.providers else self.config.orchestrator_model

        async def summarize(text: str) -> str:
            response = await summarize_provider.complete(
                messages=[{"role": "user", "content": text}],
                system=(
                    "Summarize this conversation in 2-3 concise sentences. "
                    "Capture key topics, decisions made, and any pending tasks. "
                    "Do not include greetings or filler."
                ),
                model=summarize_model,
                max_tokens=300,
                temperature=0.3,
            )
            return response.content

        return summarize


def _parse_delegations(text: str) -> list[dict]:
    """Extract <delegate agent="name">task</delegate> blocks."""
    pattern = r'<delegate\s+agent="(\w+)">(.*?)</delegate>'
    matches = re.findall(pattern, text, re.DOTALL)
    return [{"agent": agent, "task": task.strip()} for agent, task in matches]


def _strip_thinking(text: str) -> str:
    """Remove any <thinking> blocks and delegation XML from output."""
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    text = re.sub(r'<delegate\s+agent="\w+">.*?</delegate>', "", text, flags=re.DOTALL)
    return text.strip()

