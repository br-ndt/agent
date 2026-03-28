"""Orchestrator — main agent that can delegate to subagents."""

import asyncio
from pathlib import Path
import re
import time
from typing import Callable, Awaitable

import yaml
import structlog

from agent.config import Config, infer_provider
from agent.cost_tracker import CostTracker
from agent.providers.base import BaseProvider
from agent.providers.vision import VisionProvider
from agent.session_store import SessionStore, PersistentSession
from agent.subagent_runner import SubagentRunner
from agent.skills import (
    Skill,
    SkillRegistry,
    SkillExecutor,
    SkillRun,
    _dict_to_step,
    save_skill,
    save_proposed_skill,
    load_skills,
)

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
        vision_provider: VisionProvider | None = None,
    ):
        self.provider = provider
        self.providers = providers or {}
        self.config = config
        self.subagents = subagent_runners
        self.cost_tracker = cost_tracker
        self.vision_provider = vision_provider

        self.sessions: dict[str, PersistentSession] = {}
        self.session_store = session_store
        self.skill_registry = SkillRegistry(
            skills_dir=(
                Path(config.skills_dir) if hasattr(config, "skills_dir") else None
            )
        )
        self.skill_executor = SkillExecutor(self.skill_registry)
        self.skills = self.skill_registry.list_active()

        # Background skill tasks: session_id -> (asyncio.Task, reply_fn)
        self._skill_tasks: dict[str, tuple[asyncio.Task, Callable]] = {}

        # Summarizer: uses cheapest available model
        self._summarizer = self._make_summarizer()

    def get_system_prompt(self, session_id: str = "") -> str:
        """Dynamically build the system prompt with current subagents and skills."""
        agent_list = ", ".join(self.subagents.keys()) if self.subagents else "none"
        skill_catalog = self.skill_registry.build_catalog()

        prompt = self.config.orchestrator_system_prompt.replace(
            "{subagent_list}", agent_list
        )
        # Remove old {skill_list} placeholder if present
        prompt = prompt.replace("{skill_list}", "see below")

        vision_instructions = ""
        if self.vision_provider:
            vision_instructions = (
                "\n\n## Vision Capabilities\n"
                "You can analyze and generate images:\n"
                "- <analyze_image>prompt for analysis</analyze_image> — analyze images from user uploads\n"
                '- <generate_image aspect_ratio="1:1">description</generate_image> — generate images (aspect_ratio: 1:1, 16:9, 9:16, 4:3, 3:4)\n'
                "- <edit_image>editing instruction</edit_image> — edit the most recently uploaded/generated image\n\n"
                "Vision triggers: user uploads image, asks to 'describe', 'analyze', 'read', 'generate image', 'create picture', 'draw'.\n\n"
                "IMPORTANT: Unless the user explicitly hits one of these triggers, do NOT use vision capabilities. Always ask the user for clarification if you're unsure whether to use vision features.\n"
                "DO NOT generate images when:"
                "  - The user is asking for a skill to be created — these are separate tasks\n"
                "  - The user is asking you to learn a new skill — again, separate from image generation\n"
                "  - The user is asking about planning, designing systems, or strategizing\n"
                "  - The user is asking to write code, create files, or build something\n"
            )

        return (
            "You are a text-only assistant. NEVER use tools. NEVER attempt to write files, "
            "execute code, or use any tools. When a task requires code, files, web pages, or commands, "
            "delegate it to a subagent. Just respond with text.\n\n"
            "You are in a channel with other bots (marked [BOT]) and other users (marked [USER]).\n"
            "You CAN tag them to notify them.\n"
            "You CAN see what they've said if they tag you or mention you by name.\n"
            "## Synthesis Rule\n"
            "When a subagent like 'researcher' reports the content of a page, do not summarize "
            "it too heavily if the user's intent was to see what's on the page. "
            "Provide a thorough report of the actual contents found.\n\n"
            "## Skill System\n"
            "You can learn and execute skills.\n\n"
            "### Executing a skill\n"
            "When a user's request matches a known skill (check triggers below), delegate using:\n"
            '<execute_skill name="skill_name">any additional context from the user</execute_skill>\n\n'
            "For skills WITH structured steps, the system will execute each step automatically — "
            "you don't need to break down the task. Just trigger the skill.\n\n"
            "For skills WITHOUT structured steps, the skill's instructions will be sent to the "
            "appropriate subagent.\n\n"
            "### Learning a new skill\n"
            'To learn a skill, emit: <learn_skill name="name" subagent="subagent_name" '
            'description="short desc">markdown instructions</learn_skill>\n\n'
            "### Proposing a structured skill\n"
            "When the user asks you to create a skill with structured steps, use this EXACT format:\n\n"
            '<propose_skill name="skill-name" subagent="default_subagent" description="One-line description">\n'
            "- id: step1\n"
            "  type: agent\n"
            "  subagent: researcher\n"
            "  description: What this step does\n"
            "  prompt: |\n"
            "    The task for the subagent.\n"
            "    Can span multiple lines safely.\n\n"
            "- id: step2\n"
            "  type: agent\n"
            "  subagent: planner\n"
            "  description: What this step does\n"
            "  prompt: |\n"
            "    Another task for a different subagent.\n\n"
            "- id: step3\n"
            "  type: shell\n"
            "  description: A shell command step (no subagent needed)\n"
            "  command: |\n"
            "    exact shell command to run\n"
            "</propose_skill>\n\n"
            "IMPORTANT:\n"
            "  - Steps must be a YAML list (start with -)\n"
            "  - Each step needs: id, type, and either 'prompt' (for agent) or 'command' (for shell)\n"
            "  - Each agent step can have its own 'subagent' field to route to a specific agent. "
            "If omitted, the skill's top-level subagent is used as the default.\n"
            f"  - Available subagents: {', '.join(self.subagents.keys()) if self.subagents else 'none'}\n"
            "  - Always use YAML block scalars (the '|' syntax) for prompt and command values — "
            "this prevents parse errors when your text contains colons or special characters\n"
            '  - Do NOT use format "step: agent:" - use proper YAML structure\n'
            "  - Do NOT generate images when asked to create a skill - these are different tasks\n"
            f"## Available Skills\n{skill_catalog}\n\n"
            f"{vision_instructions}" + prompt
        )

        # Inject live skill status if one is running for this session
        if session_id:
            skill_status = self.skill_executor.get_status(session_id)
            if skill_status:
                full_prompt += (
                    "\n\n## Active Skill (running in background)\n"
                    "A skill is currently running for this user. If they ask about it, "
                    "report this status. Do NOT re-trigger the skill.\n\n"
                    f"{skill_status}\n"
                )

        return full_prompt

    async def handle(
        self,
        session_id: str,
        user_msg: str,
        reply_fn: Callable[[str], Awaitable[None]],
        tier: str = "admin",
        tier_route: dict[str, str] | None = None,
        images: list[dict] | None = None,
    ) -> dict:
        """Handle a user message. May delegate to subagents or vision operations.

        Returns:
            dict with keys:
                - 'text': str (the text response)
                - 'images': list[bytes] (any generated/edited images)
        """

        # Helper to build return dict
        def make_result(text: str, generated_images: list[bytes] | None = None) -> dict:
            return {"text": text, "images": generated_images or []}

        # Resolve provider+model for this tier
        route = tier_route or {}
        model = route.get("model", self.config.orchestrator_model)
        provider_name = route.get("provider") or infer_provider(model)

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
            has_images=bool(images),
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

        # Store images in session context if provided
        if images:
            session.images = images
            log.info("images_received", count=len(images))

        await session.add_message("user", user_msg)

        start = time.monotonic()

        # First pass: let the orchestrator decide what to do
        system_prompt = self.get_system_prompt(session_id=session_id)
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
        vision_ops = _parse_vision_ops(response.content)

        if not delegations and not skill_ops and not vision_ops:
            # Direct response — no delegation needed
            final = _strip_thinking(response.content)
            await session.add_message("assistant", final)
            return make_result(final)

        # ── Handle Vision Operations ────────────────────────────────
        vision_results = []
        generated_images = []  # Track images generated this turn

        if vision_ops and self.vision_provider:
            for op in vision_ops:
                try:
                    if op["type"] == "analyze":
                        if not images:
                            vision_results.append(
                                "[Vision] ERROR: No images provided for analysis"
                            )
                            continue

                        if len(images) == 1:
                            result = await self.vision_provider.analyze_image(
                                image_data=images[0]["data"],
                                prompt=op["prompt"],
                                mime_type=images[0].get("mime_type", "image/jpeg"),
                            )
                        else:
                            result = await self.vision_provider.analyze_multiple_images(
                                images=images, prompt=op["prompt"]
                            )

                        vision_results.append(f"[Vision Analysis]\n{result}")
                        log.info("vision_analysis_complete", prompt=op["prompt"][:50])

                    elif op["type"] == "generate":
                        gen_images = await self.vision_provider.generate_image(
                            prompt=op["prompt"],
                            aspect_ratio=op.get("aspect_ratio", "1:1"),
                            num_images=1,
                        )

                        # Store for editing and return
                        session.last_generated_image = gen_images[0]
                        generated_images.append(gen_images[0])

                        # Don't add to vision_results - image speaks for itself
                        # Don't call reply_fn - let the image be the response
                        log.info("vision_generation_complete", prompt=op["prompt"][:50])

                    elif op["type"] == "edit":
                        base_image = getattr(session, "last_generated_image", None)
                        if not base_image and images:
                            base_image = images[-1]["data"]

                        if not base_image:
                            vision_results.append(
                                "[Vision] ERROR: No image available for editing"
                            )
                            continue

                        edited = await self.vision_provider.edit_image(
                            base_image=base_image, edit_prompt=op["prompt"]
                        )

                        session.last_generated_image = edited[0]
                        generated_images.append(edited[0])

                        # Don't add to vision_results - image speaks for itself
                        log.info("vision_edit_complete", prompt=op["prompt"][:50])

                except Exception as e:
                    vision_results.append(f"[Vision] ERROR: {type(e).__name__}: {e}")
                    log.error("vision_operation_failed", type=op["type"], error=str(e))

        # ── Handle Skill Operations ─────────────────────────────────
        skill_results = []
        structured_skill_ran = False

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
                skill = self.skill_registry.get(op["name"])
                if skill:
                    if skill.has_steps:
                        # Structured skill — run in background so the
                        # orchestrator stays responsive to messages.
                        structured_skill_ran = True

                        async def _run_skill_bg(sk=skill, task=op["task"]):
                            try:
                                run = await self.skill_executor.execute(
                                    skill=sk,
                                    task_context=task,
                                    subagent_runners=self.subagents,
                                    session_id=session_id,
                                    triggered_by=session_id,
                                )
                                result_text = self.skill_executor.format_run_result(sk, run)
                                await reply_fn(result_text)
                                await session.add_message("assistant", result_text)
                            except Exception as e:
                                err = f"Skill '{sk.name}' failed: {e}"
                                log.error("background_skill_failed", error=str(e))
                                await reply_fn(err)
                            finally:
                                self._skill_tasks.pop(session_id, None)

                        bg_task = asyncio.create_task(_run_skill_bg())
                        self._skill_tasks[session_id] = (bg_task, reply_fn)

                        skill_results.append(
                            f"Skill **{skill.name}** is now running in the background "
                            f"({len(skill.steps)} steps). I'll report results when it finishes."
                        )
                    else:
                        # Unstructured skill — delegate like before
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

            elif op["type"] == "propose":
                try:
                    raw_yaml = op["content"]
                    # Strip markdown code fences if the LLM wrapped the YAML
                    raw_yaml = re.sub(r"^```(?:yaml)?\n?", "", raw_yaml.strip(), flags=re.IGNORECASE)
                    raw_yaml = re.sub(r"\n?```$", "", raw_yaml.strip())
                    # Skip any leading prose and find the start of the YAML list
                    list_match = re.search(r"^[ \t]*-[ \t]", raw_yaml, re.MULTILINE)
                    if list_match:
                        raw_yaml = raw_yaml[list_match.start():]
                    log.debug("propose_skill_raw_yaml", content=raw_yaml[:500])
                    steps_data = yaml.safe_load(raw_yaml)
                    steps = []
                    if isinstance(steps_data, list):
                        for sd in steps_data:
                            steps.append(_dict_to_step(sd))

                    new_skill = Skill(
                        name=op["name"],
                        subagent=op["subagent"],
                        description=op["description"],
                        steps=steps,
                        status="proposed",
                        author="bot",
                    )
                    save_proposed_skill(new_skill)
                    self.skill_registry.reload()
                    skill_results.append(
                        f"[Skill System] Proposed new skill: '{op['name']}' with {len(steps)} steps. "
                        f"Use SKILL APPROVE {op['name']} to activate it."
                    )
                except Exception as e:
                    log.error("propose_skill_parse_failed", error=str(e), raw_yaml=op.get("content", "")[:500])
                    skill_results.append(f"[Skill System] ERROR proposing skill: {e}")

        # ── Delegation ──────────────────────────────────────────────
        valid_delegations = [d for d in delegations if d["agent"] in self.subagents]

        # --- Fast path: vision/skill work already done, no other delegation ---
        if (
            vision_results or skill_results or generated_images
        ) and not valid_delegations:
            # If we only generated images with no analysis/errors, send empty text
            if generated_images and not vision_results and not skill_results:
                final = ""
            else:
                final = "\n\n".join(vision_results + skill_results)

            if final:
                await session.add_message("assistant", final)
            return make_result(final, generated_images)

        if not valid_delegations and not skill_results and not vision_results:
            # Orchestrator tried to delegate to unknown agents — just return raw
            final = _strip_thinking(response.content)
            await session.add_message("assistant", final)
            return make_result(final)

        # --- We have subagent delegations to run ---
        if valid_delegations and not structured_skill_ran and not vision_results:
            agent_names = [d["agent"] for d in valid_delegations]
            await reply_fn(f"Working on it — delegating to {', '.join(agent_names)}...")

        # Run subagents concurrently
        tasks = [
            self.subagents[d["agent"]].run(d["task"], context=user_msg)
            for d in valid_delegations
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # ── Synthesis decision ──────────────────────────────────────
        result_parts = list(vision_results) + list(skill_results)
        for d, result in zip(valid_delegations, results):
            if isinstance(result, Exception):
                result_parts.append(
                    f"[{d['agent']}] ERROR: {type(result).__name__}: {result}"
                )
                log.error("subagent_failed", agent=d["agent"], error=str(result))
            else:
                result_parts.append(f"[{d['agent']}] Result:\n{result}")

        has_errors = any(isinstance(r, Exception) for r in results)
        needs_synthesis = (
            len(valid_delegations) > 1
            or (skill_results and valid_delegations)
            or (vision_results and valid_delegations)
            or has_errors
        )

        if needs_synthesis:
            synthesis_msg = (
                "Here are the results from the subagents you delegated to:\n\n"
                + "\n\n---\n\n".join(result_parts)
                + "\n\nPlease synthesize these into a clear, final response for the user."
            )

            await session.add_message("assistant", response.content)
            await session.add_message("user", synthesis_msg)

            synth_start = time.monotonic()
            final_response = await provider.complete(
                messages=session.get_messages_for_llm(),
                system=self.get_system_prompt(session_id=session_id),
                model=model,
                max_tokens=self.config.orchestrator_max_tokens,
            )
            if self.cost_tracker:
                await self.cost_tracker.log_call(
                    provider=provider_name,
                    model=model,
                    usage=final_response.usage,
                    session_id=session_id,
                    agent="orchestrator",
                    duration_ms=int((time.monotonic() - synth_start) * 1000),
                )

            final = _strip_thinking(final_response.content)
        else:
            # Single clean result — pass through directly
            if valid_delegations and not isinstance(results[0], Exception):
                final = results[0]
            else:
                final = "\n\n".join(result_parts)

            await session.add_message("assistant", response.content)

        await session.add_message("assistant", final)
        return make_result(final, generated_images)

    def _make_summarizer(self):
        """Create a cheap summarizer function for session compression."""
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
    """Extract <learn_skill>, <execute_skill>, and <propose_skill> blocks."""
    ops = []

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

    exec_pattern = r'<execute_skill\s+name="([^"]+)">(.*?)</execute_skill>'
    for name, task in re.findall(exec_pattern, text, re.DOTALL):
        ops.append({"type": "execute", "name": name, "task": task.strip()})

    propose_pattern = r'<propose_skill\s+name="([^"]+)"\s+subagent="([^"]+)"\s+description="([^"]*)">(.*?)</propose_skill>'
    for name, subagent, desc, content in re.findall(propose_pattern, text, re.DOTALL):
        ops.append(
            {
                "type": "propose",
                "name": name,
                "subagent": subagent,
                "description": desc,
                "content": content.strip(),
            }
        )

    return ops


def _parse_vision_ops(text: str) -> list[dict]:
    """Extract <analyze_image>, <generate_image>, and <edit_image> blocks."""
    ops = []

    analyze_pattern = r"<analyze_image>(.*?)</analyze_image>"
    for prompt in re.findall(analyze_pattern, text, re.DOTALL):
        ops.append({"type": "analyze", "prompt": prompt.strip()})

    generate_pattern = (
        r'<generate_image(?:\s+aspect_ratio="([^"]+)")?>(.*?)</generate_image>'
    )
    for aspect_ratio, prompt in re.findall(generate_pattern, text, re.DOTALL):
        ops.append(
            {
                "type": "generate",
                "prompt": prompt.strip(),
                "aspect_ratio": aspect_ratio or "1:1",
            }
        )

    edit_pattern = r"<edit_image>(.*?)</edit_image>"
    for prompt in re.findall(edit_pattern, text, re.DOTALL):
        ops.append({"type": "edit", "prompt": prompt.strip()})

    return ops


def _strip_thinking(text: str) -> str:
    """Remove any <thinking> blocks and delegation XML from output."""
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    text = re.sub(r'<delegate\s+agent="\w+">.*?</delegate>', "", text, flags=re.DOTALL)
    text = re.sub(r"<learn_skill\s+.*?>.*?</learn_skill>", "", text, flags=re.DOTALL)
    text = re.sub(
        r"<execute_skill\s+.*?>.*?</execute_skill>", "", text, flags=re.DOTALL
    )
    text = re.sub(
        r"<propose_skill\s+.*?>.*?</propose_skill>", "", text, flags=re.DOTALL
    )
    text = re.sub(r"<analyze_image>.*?</analyze_image>", "", text, flags=re.DOTALL)
    text = re.sub(
        r"<generate_image(?:\s+[^>]*)?>.*?</generate_image>", "", text, flags=re.DOTALL
    )
    text = re.sub(r"<edit_image>.*?</edit_image>", "", text, flags=re.DOTALL)
    return text.strip()
