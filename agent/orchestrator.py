"""Orchestrator — main agent that can delegate to subagents."""

import asyncio
import re
import time
from typing import Callable, Awaitable

import structlog

from agent.config import Config
from agent.cost_tracker import CostTracker
from agent.providers.base import BaseProvider
from agent.session_store import SessionStore, PersistentSession
from agent.subagent_runner import SubagentRunner
from agent.skills import Skill, save_skill, load_skills

log = structlog.get_logger()


class Orchestrator:
    def __init__(
        self,
        provider: BaseProvider,
        config: Config,
        subagent_runners: dict[str, SubagentRunner],
        providers: dict[str, BaseProvider] | None = None,
        session_store: SessionStore | None = None,
        skills: list[Skill] | None = None,
        cost_tracker: CostTracker | None = None,
    ):
        self.provider = provider
        self.providers = providers or {}
        self.config = config
        self.subagents = subagent_runners
        self.cost_tracker = cost_tracker
        self.sessions: dict[str, PersistentSession] = {}
        self.session_store = session_store
        self.skills = skills or []

        # Summarizer: uses cheapest available model
        self._summarizer = self._make_summarizer()

    def get_system_prompt(self) -> str:
        """Dynamically build the system prompt with current subagents and skills."""
        agent_list = ", ".join(self.subagents.keys()) if self.subagents else "none"
        skill_list = (
            ", ".join(f"'{s.name}' ({s.subagent})" for s in self.skills)
            if self.skills
            else "none"
        )

        prompt = self.config.orchestrator_system_prompt.replace(
            "{subagent_list}", agent_list
        )
        prompt = prompt.replace("{skill_list}", skill_list)

        return (
            "You are a text-only assistant. NEVER use tools. NEVER attempt to write files, "
            "execute code, or use any tools. When a task requires code, files, web pages, or commands, "
            "delegate it to a subagent. Just respond with text.\n\n"
            "## Synthesis Rule\n"
            "When a subagent like 'researcher' reports the content of a page, do not summarize "
            "it too heavily if the user's intent was to see what's on the page. "
            "Provide a thorough report of the actual contents found.\n\n"
            "## Skill System\n"
            "You can learn and execute skills. A skill is a set of instructions for a subagent.\n"
            '- To learn a skill, emit: <learn_skill name="name" subagent="subagent_name" description="short desc">markdown instructions</learn_skill>\n'
            '- To execute a skill, emit: <execute_skill name="name">task for the skill</execute_skill>\n\n'
            f"Current Skills: {skill_list}\n\n" + prompt
        )

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
        log.info(
            "orchestrator_routing",
            provider_name=provider_name,
            model=model,
            provider_type=type(provider).__name__,
            tier=tier,
            tier_route=tier_route,
            providers_keys=list(self.providers.keys()),
        )

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
        system_prompt = self.get_system_prompt()
        response = await provider.complete(
            messages=session.get_messages_for_llm(),
            system=system_prompt,
            model=model,
            max_tokens=self.config.orchestrator_max_tokens,
        )
        if self.cost_tracker:
            await self.cost_tracker.log_call(
                provider=provider_name,
                model=model,
                usage=response.usage,
                session_id=session_id,
                agent="orchestrator",
                duration_ms=int((time.monotonic() - start) * 1000),
            )
        log.info("orchestrator_raw_response", content=response.content[:500])

        elapsed = time.monotonic() - start
        log.info(
            "orchestrator_response", elapsed=f"{elapsed:.1f}s", usage=response.usage
        )

        # Check for delegation blocks
        delegations = _parse_delegations(response.content)
        skill_ops = _parse_skill_ops(response.content)

        if not delegations and not skill_ops:
            # Direct response — no delegation needed
            final = _strip_thinking(response.content)
            await session.add_message("assistant", final)
            return final

        # --- Handle Skill Operations ---
        skill_results = []
        for op in skill_ops:
            if op["type"] == "learn":
                try:
                    new_skill = Skill(
                        name=op["name"],
                        subagent=op["subagent"],
                        description=op["description"],
                        content=op["content"],
                    )
                    save_skill(new_skill)
                    self.skills.append(new_skill)

                    # Update relevant subagent runner personality
                    if op["subagent"] in self.subagents:
                        runner = self.subagents[op["subagent"]]
                        skill_block = f"### Skill: {op['name']}\n{op['description']}\n\n{op['content']}"
                        if "## Skills" not in runner.config.personality:
                            runner.config.personality += "\n\n## Skills\n"
                        runner.config.personality += f"\n\n---\n\n{skill_block}"

                    skill_results.append(
                        f"[Skill System] Successfully learned new skill: {op['name']} for subagent {op['subagent']}"
                    )
                    log.info("skill_learned", name=op["name"], subagent=op["subagent"])
                except Exception as e:
                    skill_results.append(
                        f"[Skill System] ERROR: Failed to learn skill {op['name']}: {e}"
                    )
                    log.error("skill_learn_failed", name=op["name"], error=str(e))

            elif op["type"] == "execute":
                skill = next(
                    (s for s in self.skills if s.name.lower() == op["name"].lower()),
                    None,
                )
                if skill:
                    # Treat execution as a delegation to the skill's subagent
                    delegations.append(
                        {
                            "agent": skill.subagent,
                            "task": f"Using your skill '{skill.name}', perform the following task: {op['task']}",
                        }
                    )
                    log.info(
                        "skill_execution_triggered",
                        name=skill.name,
                        agent=skill.subagent,
                    )
                else:
                    skill_results.append(
                        f"[Skill System] ERROR: Skill '{op['name']}' not found."
                    )
                    log.warning("skill_not_found", name=op["name"])

        # Delegation mode: fan out to subagents in parallel
        valid_delegations = [d for d in delegations if d["agent"] in self.subagents]

        if not valid_delegations and not skill_results:
            # Orchestrator tried to delegate to unknown agents — just return raw
            final = _strip_thinking(response.content)
            await session.add_message("assistant", final)
            return final

        agent_names = [d["agent"] for d in valid_delegations]
        all_ops_desc = ", ".join(
            agent_names
            + [f"Skill:{op['name']}" for op in skill_ops if op["type"] == "learn"]
        )
        await reply_fn(f"Working on it — delegating to {all_ops_desc}...")

        # Run subagents concurrently
        tasks = [
            self.subagents[d["agent"]].run(d["task"], context=user_msg)
            for d in valid_delegations
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Format results for synthesis
        result_parts = skill_results  # Include any skill learning successes/errors
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
            system=self.get_system_prompt(),
            model=model,
            max_tokens=self.config.orchestrator_max_tokens,
        )
        if self.cost_tracker:
            await self.cost_tracker.log_call(
                provider=provider_name,
                model=model,
                usage=response.usage,
                session_id=session_id,
                agent="orchestrator",
                duration_ms=int((time.monotonic() - start) * 1000),
            )

        final = _strip_thinking(final_response.content)
        await session.add_message("assistant", final)

        return final

    def _make_summarizer(self):
        """Create a cheap summarizer function for session compression."""
        # Prefer Google (cheapest), fall back to default provider
        summarize_provider = self.providers.get("google", self.provider)
        summarize_model = (
            "gemini-2.5-flash"
            if "google" in self.providers
            else self.config.orchestrator_model
        )

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


def _parse_skill_ops(text: str) -> list[dict]:
    """Extract <learn_skill> and <execute_skill> blocks."""
    ops = []

    # <learn_skill name="..." subagent="..." description="...">markdown</learn_skill>
    learn_pattern = r'<learn_skill\s+name="([^"]+)"\s+subagent="([^"]+)"\s+description="([^"]*)">(.*?)</learn_skill>'
    for name, subagent, desc, content in re.findall(learn_pattern, text, re.DOTALL):
        ops.append(
            {
                "type": "learn",
                "name": name,
                "subagent": subagent,
                "description": desc,
                "content": content.strip(),
            }
        )

    # <execute_skill name="...">task</execute_skill>
    exec_pattern = r'<execute_skill\s+name="([^"]+)">(.*?)</execute_skill>'
    for name, task in re.findall(exec_pattern, text, re.DOTALL):
        ops.append({"type": "execute", "name": name, "task": task.strip()})

    return ops


def _strip_thinking(text: str) -> str:
    """Remove any <thinking> blocks and delegation XML from output."""
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    text = re.sub(r'<delegate\s+agent="\w+">.*?</delegate>', "", text, flags=re.DOTALL)
    text = re.sub(r"<learn_skill\s+.*?>.*?</learn_skill>", "", text, flags=re.DOTALL)
    text = re.sub(
        r"<execute_skill\s+.*?>.*?</execute_skill>", "", text, flags=re.DOTALL
    )
    return text.strip()
