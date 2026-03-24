"""Subagent runner — executes a task against a specific provider/model."""

import structlog

from agent.config import SubagentConfig
from agent.providers.base import BaseProvider

log = structlog.get_logger()

MAX_TURNS = 10


class SubagentRunner:
    """Runs a subagent to completion with its own personality and model."""

    def __init__(self, config: SubagentConfig, provider: BaseProvider):
        self.config = config
        self.provider = provider

    async def run(self, task: str, context: str = "") -> str:
        """Execute a task and return the result string."""
        prompt = task
        if context:
            prompt = f"Context:\n{context}\n\nTask:\n{task}"

        messages = [{"role": "user", "content": prompt}]

        log.info(
            "subagent_starting",
            agent=self.config.name,
            model=self.config.model,
            task_len=len(task),
        )

        for turn in range(MAX_TURNS):
            response = await self.provider.complete(
                messages=messages,
                system=self.config.personality,
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )

            log.info(
                "subagent_turn",
                agent=self.config.name,
                turn=turn,
                usage=response.usage,
            )

            # For now, single-turn (no tool loop). Return the response.
            # Tool execution can be added here later by checking response.tool_calls
            return response.content

        return response.content
