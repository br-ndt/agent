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
from agent.diagnostics import ErrorJournal, DiagnosticStore, build_diagnosis_prompt
from agent.knowledge import KnowledgeStore, KnowledgeDoc
from agent.memory import MemoryStore, build_memory_index_prompt
from agent.providers.base import BaseProvider
from agent.providers.vision import VisionProvider
from agent.persona_enforcement import check_output_for_violations
from agent.sanitizer import sanitize_delegation_result
from agent.session_store import SessionStore, PersistentSession
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
from agent.state_ledger import StateLedger, EntryType
from agent.subagent_runner import SubagentRunner, WORKSPACES_DIR

log = structlog.get_logger()


def _parse_memory_ops(text: str) -> list[dict]:
    """Extract <recall> and <remember> operations from LLM output."""
    import re

    ops = []

    # <recall topic="name">reason</recall>
    recall_pattern = r'<recall\s+topic="([^"]+)">(.*?)</recall>'
    for topic, reason in re.findall(recall_pattern, text, re.DOTALL):
        ops.append({"type": "recall", "topic": topic.strip(), "reason": reason.strip()})

    # <remember topic="name" tags="t1,t2" global="true">content</remember>
    remember_pattern = r'<remember\s+topic="([^"]+)"(?:\s+tags="([^"]*)")?(?:\s+global="([^"]*)")?\s*>(.*?)</remember>'
    for topic, tags_str, global_str, content in re.findall(
        remember_pattern, text, re.DOTALL
    ):
        tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else []
        is_global = global_str.lower() in ("true", "1", "yes") if global_str else False
        ops.append(
            {
                "type": "remember",
                "topic": topic.strip(),
                "content": content.strip(),
                "tags": tags,
                "global": is_global,
            }
        )

    return ops


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
        memory_store: MemoryStore | None = None,
        state_ledger: StateLedger | None = None,
        knowledge_store: KnowledgeStore | None = None,
    ):
        self.provider = provider
        self.providers = providers or {}
        self.config = config
        self.subagents = subagent_runners
        self.cost_tracker = cost_tracker
        self.vision_provider = vision_provider
        self.memory_store = memory_store
        self.state_ledger = state_ledger
        self.knowledge_store = knowledge_store or KnowledgeStore()
        self.knowledge_store.load()

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

        # Active subagent delegations: agent_name -> {task, session_id, description}
        self._active_delegations: dict[str, dict] = {}

        # Pending background results: session_id -> list of result strings
        # These are results from background tasks that completed while the user
        # was in a different conversation turn. Injected as context, not into
        # session history directly.
        self._pending_results: dict[str, list[str]] = {}

        # Diagnostics
        self.error_journal = ErrorJournal()
        self.diagnostic_store = DiagnosticStore()


    def get_system_prompt(self, session_id: str = "", prompt_tier: str = "full", memory_index: str = "") -> str:
        """Dynamically build the system prompt with current subagents and skills.

        prompt_tier controls how much context is included:
          - "classify": Bare minimum for message classification (~50 tokens)
          - "minimal":  Personality + subagent list + delegation syntax (no skills/vision)
          - "full":     Everything (skills, vision, learning, proposing)
        """
        if prompt_tier == "classify":
            return (
                "Classify this message into one category:\n"
                "- SIMPLE: greetings, small talk, factual questions, math, formatting, "
                "simple opinions, short answers, questions ABOUT skills or capabilities, "
                "status checks, acknowledgements, or anything you can answer conversationally "
                "without running code or fetching external data\n"
                "- DELEGATE: the user is asking you to DO something that requires tools — "
                "write/edit code, execute a skill, fetch a web page, read/write files, "
                "generate images, or run commands. The key word is DO, not ASK ABOUT.\n"
                "- COMPLEX: nuanced analysis, architecture, multi-step reasoning, "
                "debate, detailed explanations, synthesis of multiple topics — but still "
                "answerable conversationally without tools\n\n"
                "IMPORTANT: Questions about how something works, what skills exist, "
                "or explaining a concept are SIMPLE or COMPLEX, not DELEGATE. "
                "DELEGATE is only for requests that require tool execution.\n\n"
                "Reply with ONLY the category name, nothing else."
            )

        agent_list = ", ".join(self.subagents.keys()) if self.subagents else "none"

        # Build a routing guide so the orchestrator knows what each agent does
        agent_descriptions = []
        for name, runner in self.subagents.items():
            # Extract first sentence of personality as a short description
            personality = runner.config.personality.strip()
            first_line = personality.split("\n")[0].strip()
            if len(first_line) > 120:
                first_line = first_line[:117] + "..."
            tools = ", ".join(runner.config.tools) if runner.config.tools else "none"
            agent_descriptions.append(f"- **{name}** [{tools}]: {first_line}")
        agent_guide = "\n".join(agent_descriptions)

        prompt = self.config.orchestrator_system_prompt.replace(
            "{subagent_list}", agent_list
        )
        # Remove old {skill_list} placeholder if present
        prompt = prompt.replace("{skill_list}", "see below")

        base = (
            "You are a text-only assistant. NEVER use tools. NEVER attempt to write files, "
            "execute code, or use any tools. When a task requires code, files, web pages, or commands, "
            "delegate it to a subagent. Just respond with text.\n\n"
            "You are in a channel with other bots (marked [BOT]) and other users (marked [USER]).\n"
            "You CAN tag them to notify them.\n"
            "You CAN see what they've said if they tag you or mention you by name.\n\n"
            "After each operation you request (delegation, vision, etc.), you will see the "
            "results and can decide what to do next — request more operations or respond to the "
            "user. You don't need to plan everything upfront; you can gather information first "
            "and act on it after.\n\n"
            "## Synthesis Rule\n"
            "When a subagent like 'researcher' reports the content of a page, do not summarize "
            "it too heavily if the user's intent was to see what's on the page. "
            "Provide a thorough report of the actual contents found.\n\n"
            "## Web Task Routing\n"
            "- READ a page / research something → delegate to **researcher**\n"
            "- REVIEW a PR or code on the web → delegate to **reviewer**\n"
            "- INTERACT with a website (play a game, fill forms, click buttons, "
            "use a web app) → delegate to **playwright**\n\n"
        )

        if prompt_tier == "minimal":
            # Delegation syntax only — no skills, vision, or learning
            base += (
                "When a task requires delegation, emit:\n"
                '<delegate agent="agent_name">task description</delegate>\n\n'
                f"## Subagent Routing Guide\n{agent_guide}\n\n"
            )
            result = base + prompt
            return result

        # ── Full tier: skills, vision, learning, proposing ──────────
        # Memory index is pre-fetched in the async handle() path and passed in
        memory_block = memory_index
        skill_catalog = self.skill_registry.build_catalog()

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

        result = (
            base + f"## Subagent Routing Guide\n{agent_guide}\n\n"
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
            "  - Do NOT generate images when asked to create a skill - these are different tasks\n\n"
            "### Fixing / improving skills\n"
            "To fix a broken skill or improve an existing one, delegate to the 'sysadmin' agent:\n"
            "<delegate agent=\"sysadmin\">Fix the skill 'skill-name': describe the problem and any error output</delegate>\n\n"
            "The sysadmin can read agent source code and edit skill files directly.\n\n"
            f"## Available Skills\n{skill_catalog}\n\n"
            f"{vision_instructions}"
            + (f"\n{memory_block}\n\n" if memory_block else "")
            + self.knowledge_store.build_index_prompt()
            + "\n\n" + prompt
        )

        # Inject live skill status if one is running for this session
        if session_id:
            skill_status = self.skill_executor.get_status(session_id)
            if skill_status:
                result += (
                    "\n\n## Active Skill (running in background)\n"
                    "A skill is currently running for this user. If they ask about it, "
                    "report this status. Do NOT re-trigger the skill.\n\n"
                    f"{skill_status}\n"
                )

        # Inject busy subagent status
        if self._active_delegations:
            import time as _time
            busy_lines = []
            for agent_name, info in self._active_delegations.items():
                started = info.get("started_at", 0)
                elapsed = int(_time.time() - started) if started else 0
                elapsed_str = f"{elapsed}s ago" if elapsed < 60 else f"{elapsed // 60}m ago"
                busy_lines.append(
                    f"- **{agent_name}** (started {elapsed_str}): {info.get('task', '?')[:100]}"
                )
            result += (
                "\n\n## Busy Subagents\n"
                "These subagents are currently working on tasks. Do NOT delegate "
                "to them — report that they are busy if the user asks for something "
                "that would require them.\n\n" + "\n".join(busy_lines) + "\n"
            )

        return result

    MAX_PASSES = 4

    async def handle(
        self,
        session_id: str,
        user_msg: str,
        reply_fn: Callable[[str], Awaitable[None]],
        tier: str = "admin",
        tier_route: dict[str, str] | None = None,
        images: list[dict] | None = None,
    ) -> dict:
        """Handle a user message via multi-pass orchestration.

        Each pass: LLM decides what to do → operations execute → results
        feed back as context. Loop ends when the LLM responds directly
        (no operations) or max passes reached.

        Returns dict with 'text' and 'images' keys.
        """

        def make_result(text: str, gen_images: list[bytes] | None = None) -> dict:
            return {"text": text, "images": gen_images or []}

        # ── Resolve provider + model for this tier ──────────────────
        route = tier_route or {}
        model = route.get("model", self.config.orchestrator_model)
        provider_name = route.get("provider") or infer_provider(model)
        provider = self.providers.get(provider_name, self.provider)

        log.info(
            "orchestrator_routing",
            provider_name=provider_name,
            model=model,
            provider_type=type(provider).__name__,
            tier=tier,
            has_images=bool(images),
        )

        # ── Session setup ───────────────────────────────────────────
        if session_id not in self.sessions:
            session = PersistentSession(
                session_id=session_id,
                store=self.session_store,
                memory_store=self.memory_store,
            )
            self.sessions[session_id] = session
        else:
            session = self.sessions[session_id]

        await session.ensure_loaded()
        memory_index_prompt = ""
        if self.memory_store:
            pointers = await self.memory_store.get_index(session_id)
            global_pointers = await self.memory_store.get_global_index()
            memory_index_prompt = build_memory_index_prompt(pointers, global_pointers)

        if images:
            session.images = images
            log.info("images_received", count=len(images))
            # Tell the orchestrator LLM that images are attached — it can't
            # see them directly, but it needs to know they're there so it can
            # decide to emit <analyze_image>.
            filenames = [img.get("filename", "image") for img in images]
            user_msg = (
                f"[{len(images)} image(s) attached: {', '.join(filenames)}]\n\n"
                + user_msg
            )

        await session.add_message("user", user_msg)

        # Drain any pending background results into session context
        pending = self._pending_results.pop(session_id, [])
        if pending:
            context_note = (
                "[Background tasks completed since your last message:\n"
                + "\n---\n".join(pending)
                + "\n]"
            )
            await session.add_message("assistant", context_note)
            log.info(
                "pending_results_injected", session_id=session_id, count=len(pending)
            )

        start = time.monotonic()

        # ── Classification ──────────────────────────────────────────
        classify_model = self.config.classify_model or model
        classify_provider_name = infer_provider(classify_model)
        classify_provider = self.providers.get(classify_provider_name, provider)

        if images:
            msg_class = "DELEGATE"
        else:
            try:
                classify_response = await classify_provider.complete(
                    messages=[{"role": "user", "content": user_msg}],
                    system=self.get_system_prompt(prompt_tier="classify"),
                    model=classify_model,
                    max_tokens=10,
                    temperature=0.0,
                )
                msg_class = classify_response.content.strip().upper()
                if self.cost_tracker:
                    await self.cost_tracker.log_call(
                        provider=classify_provider_name,
                        model=classify_model,
                        usage=classify_response.usage,
                        session_id=session_id,
                        agent="classifier",
                        duration_ms=int((time.monotonic() - start) * 1000),
                    )
                if msg_class not in ("SIMPLE", "DELEGATE", "COMPLEX"):
                    msg_class = "DELEGATE"
                log.info("classify_result", classification=msg_class)
            except Exception as e:
                log.warning("classify_failed", error=str(e))
                msg_class = "DELEGATE"

        # ── Select model + prompt tier ──────────────────────────────
        if msg_class == "SIMPLE":
            prompt_tier = "minimal"
            route_model, route_provider = model, provider
            route_provider_name = provider_name
        elif msg_class == "COMPLEX":
            prompt_tier = "full"
            fallback = self.config.orchestrator_fallback_model
            if fallback:
                route_model = fallback
                route_provider_name = infer_provider(fallback)
                route_provider = self.providers.get(route_provider_name, provider)
            else:
                route_model, route_provider = model, provider
                route_provider_name = provider_name
        else:
            prompt_tier = "full"
            route_model, route_provider = model, provider
            route_provider_name = provider_name

        log.info(
            "ingress_tier",
            classification=msg_class,
            prompt_tier=prompt_tier,
            route_model=route_model,
        )

        # ── Multi-pass loop ─────────────────────────────────────────
        generated_images: list[bytes] = []
        structured_skill_ran = False
        notified_delegation = False
        final = ""

        for pass_num in range(self.MAX_PASSES):
            pass_start = time.monotonic()

            # Use smarter model for synthesis passes (pass > 0)
            if pass_num > 0:
                synth_model = (
                    self.config.synthesis_model
                    or self.config.orchestrator_fallback_model
                    or route_model
                )
                synth_provider_name = infer_provider(synth_model)
                synth_provider = self.providers.get(synth_provider_name, route_provider)
                pass_model, pass_provider = synth_model, synth_provider
                pass_provider_name = synth_provider_name
            else:
                pass_model, pass_provider = route_model, route_provider
                pass_provider_name = route_provider_name

            system_prompt = self.get_system_prompt(
                session_id=session_id,
                prompt_tier=prompt_tier,
                memory_index=memory_index_prompt,
            )
            response = await pass_provider.complete(
                messages=session.get_messages_for_llm(),
                system=system_prompt,
                model=pass_model,
                max_tokens=self.config.orchestrator_max_tokens,
            )

            if self.cost_tracker:
                await self.cost_tracker.log_call(
                    provider=pass_provider_name,
                    model=pass_model,
                    usage=response.usage,
                    session_id=session_id,
                    agent="orchestrator",
                    duration_ms=int((time.monotonic() - pass_start) * 1000),
                )

            log.info(
                "orchestrator_pass",
                pass_num=pass_num,
                model=pass_model,
                elapsed=f"{time.monotonic() - pass_start:.1f}s",
                content_preview=response.content[:300],
            )

            # ── Parse operations from this pass ─────────────────────
            delegations = _parse_delegations(response.content)
            skill_ops = _parse_skill_ops(response.content)
            vision_ops = _parse_vision_ops(response.content)
            memory_ops = _parse_memory_ops(response.content)
            knowledge_ops = _parse_knowledge_ops(response.content)

            # Infer skill execution if LLM forgot XML
            if not (session_id in self._skill_tasks) and not any(
                op["type"] == "execute" for op in skill_ops
            ):
                inferred = _infer_skill_execution(response.content, self.skill_registry)
                if inferred:
                    inferred["task"] = user_msg
                    log.warning("skill_execution_inferred", skill=inferred["name"])
                    skill_ops.append(inferred)

            has_ops = bool(delegations or skill_ops or vision_ops or memory_ops or knowledge_ops)

            if not has_ops:
                # No operations — this is the final response
                final = _strip_thinking(response.content)
                await session.add_message("assistant", final)
                break

            # ── Execute operations for this pass ────────────────────
            pass_results: list[str] = []

            # Vision
            v_results, v_images = await self._execute_vision_ops(
                vision_ops,
                images,
                session,
            )
            pass_results.extend(v_results)
            generated_images.extend(v_images)

            # Skills (background skills are fire-and-forget)
            s_results, s_delegations, s_structured = await self._execute_skill_ops(
                skill_ops,
                session_id,
                reply_fn,
                session,
            )
            pass_results.extend(s_results)

            # Memory
            if memory_ops and self.memory_store:
                m_results = await self._execute_memory_ops(memory_ops, session_id)
                pass_results.extend(m_results)

            # Knowledge
            if knowledge_ops:
                k_results = self._execute_knowledge_ops(knowledge_ops)
                pass_results.extend(k_results)

            delegations.extend(s_delegations)
            if s_structured:
                structured_skill_ran = True

            # Delegations
            valid_delegations = [d for d in delegations if d["agent"] in self.subagents]

            if valid_delegations:
                if not notified_delegation and not structured_skill_ran:
                    # Use the LLM's own preamble text as the status message —
                    # it's more natural than rebuilding from the task text.
                    preamble = _extract_preamble(response.content)
                    if preamble:
                        await reply_fn(preamble)
                    else:
                        await reply_fn(_summarize_delegations(valid_delegations))
                    notified_delegation = True

                d_results = await self._execute_delegations(
                    valid_delegations,
                    user_msg,
                    session_id,
                )
                pass_results.extend(d_results)

            # ── If only images generated with no text results, return ──
            if generated_images and not pass_results:
                await session.add_message("assistant", "")
                return make_result("", generated_images)

            # ── Feed results back for next pass ─────────────────────
            # Record this pass's response as assistant, results as user
            await session.add_message("assistant", response.content)

            results_msg = "Results from this step:\n\n" + "\n\n---\n\n".join(
                pass_results
            )
            await session.add_message("user", results_msg)

            log.info(
                "pass_results_fed_back",
                pass_num=pass_num,
                result_count=len(pass_results),
            )

            # After last allowed pass, the next iteration will be the
            # final response (or we'll fall through below)

        else:
            # Exhausted all passes — use last response as-is
            log.warning("max_passes_reached", passes=self.MAX_PASSES)
            final = _strip_thinking(response.content)
            await session.add_message("assistant", final)

        return make_result(final, generated_images)

    # ── Execution helpers (called from the multi-pass loop) ─────

    async def _execute_vision_ops(
        self,
        vision_ops: list[dict],
        images: list[dict] | None,
        session: PersistentSession,
    ) -> tuple[list[str], list[bytes]]:
        """Run vision operations. Returns (text_results, generated_images)."""
        results: list[str] = []
        gen_images: list[bytes] = []

        if not vision_ops or not self.vision_provider:
            return results, gen_images

        for op in vision_ops:
            try:
                if op["type"] == "analyze":
                    if not images:
                        results.append(
                            "[Vision] ERROR: No images provided for analysis"
                        )
                        continue
                    if len(images) == 1:
                        r = await self.vision_provider.analyze_image(
                            image_data=images[0]["data"],
                            prompt=op["prompt"],
                            mime_type=images[0].get("mime_type", "image/jpeg"),
                        )
                    else:
                        r = await self.vision_provider.analyze_multiple_images(
                            images=images,
                            prompt=op["prompt"],
                        )
                    results.append(f"[Vision Analysis]\n{r}")
                    log.info("vision_analysis_complete", prompt=op["prompt"][:50])

                elif op["type"] == "generate":
                    gen = await self.vision_provider.generate_image(
                        prompt=op["prompt"],
                        aspect_ratio=op.get("aspect_ratio", "1:1"),
                        num_images=1,
                    )
                    session.last_generated_image = gen[0]
                    gen_images.append(gen[0])
                    log.info("vision_generation_complete", prompt=op["prompt"][:50])

                elif op["type"] == "edit":
                    base = getattr(session, "last_generated_image", None)
                    if not base and images:
                        base = images[-1]["data"]
                    if not base:
                        results.append("[Vision] ERROR: No image available for editing")
                        continue
                    edited = await self.vision_provider.edit_image(
                        base_image=base,
                        edit_prompt=op["prompt"],
                    )
                    session.last_generated_image = edited[0]
                    gen_images.append(edited[0])
                    log.info("vision_edit_complete", prompt=op["prompt"][:50])

            except Exception as e:
                results.append(f"[Vision] ERROR: {type(e).__name__}: {e}")
                log.error("vision_operation_failed", type=op["type"], error=str(e))

        return results, gen_images

    async def _execute_memory_ops(self, ops, session_id):
        """Execute <recall> and <remember> operations."""
        results = []

        for op in ops:
            if op["type"] == "recall":
                topic = await self.memory_store.get_topic(session_id, op["topic"])
                if topic:
                    content = topic.content[:2000]  # MAX_TOPIC_CONTENT
                    results.append(f"[Memory] Recalled '{op['topic']}':\n{content}")
                else:
                    # Try searching
                    matches = await self.memory_store.search_topics(
                        session_id, op["topic"]
                    )
                    if matches:
                        lines = [
                            f"[Memory] Topic '{op['topic']}' not found exactly. Similar:"
                        ]
                        for m in matches:
                            lines.append(m.to_index_line())
                        results.append("\n".join(lines))
                    else:
                        results.append(
                            f"[Memory] No memory found for topic '{op['topic']}'"
                        )

            elif op["type"] == "remember":
                await self.memory_store.save_topic(
                    session_id=session_id,
                    topic=op["topic"],
                    summary=op["content"][:150],
                    content=op["content"],
                    tags=op.get("tags", []),
                    global_=op.get("global", False),
                )
                scope = "globally" if op.get("global") else "for this session"
                results.append(f"[Memory] Remembered '{op['topic']}' {scope}")

        return results

    async def _execute_skill_ops(
        self,
        skill_ops: list[dict],
        session_id: str,
        reply_fn: Callable[[str], Awaitable[None]],
        session: PersistentSession,
    ) -> tuple[list[str], list[dict], bool]:
        """Run skill operations.

        Returns (results, extra_delegations, structured_skill_ran).
        extra_delegations are added to the delegation list for this pass.
        """
        results: list[str] = []
        extra_delegations: list[dict] = []
        structured_ran = False

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
                    if session.memory_store:
                        try:
                            await session.memory_store.save_topic(
                                session_id=session_id,
                                topic=f"skill-learned-{op['name']}",
                                summary=f"Learned skill '{op['name']}' for {op['subagent']}",
                                content=op["content"][:3000],
                                tags=["skill-definition", op["name"]],
                                global_=True,  # skill definitions should persist across sessions
                            )
                        except Exception:
                            pass
                    if op["subagent"] in self.subagents:
                        runner = self.subagents[op["subagent"]]
                        skill_block = f"### Skill: {op['name']}\n{op['description']}\n\n{op['content']}"
                        if "## Skills" not in runner.config.personality:
                            runner.config.personality += "\n\n## Skills\n"
                        runner.config.personality += f"\n\n---\n\n{skill_block}"
                    results.append(
                        f"[Skill System] Successfully learned new skill: {op['name']} for subagent {op['subagent']}"
                    )
                    log.info("skill_learned", name=op["name"], subagent=op["subagent"])
                except Exception as e:
                    results.append(
                        f"[Skill System] ERROR: Failed to learn skill {op['name']}: {e}"
                    )
                    log.error("skill_learn_failed", name=op["name"], error=str(e))

            elif op["type"] == "execute":
                skill = self.skill_registry.get(op["name"])
                if not skill:
                    results.append(
                        f"[Skill System] ERROR: Skill '{op['name']}' not found."
                    )
                    log.warning("skill_not_found", name=op["name"])
                    continue

                if skill.has_steps:
                    if session_id in self._skill_tasks:
                        status = self.skill_executor.get_status(session_id) or "running"
                        results.append(
                            f"A skill is already running for this session:\n{status}"
                        )
                        continue

                    structured_ran = True

                    async def _run_skill_bg(sk=skill, task=op["task"]):
                        log.info(
                            "background_skill_starting",
                            skill=sk.name,
                            session=session_id,
                        )
                        try:
                            run = await self.skill_executor.execute(
                                skill=sk,
                                task_context=task,
                                subagent_runners=self.subagents,
                                session_id=session_id,
                                triggered_by=session_id,
                            )
                            result_text = self.skill_executor.format_run_result(sk, run)
                            if session.memory_store:
                                try:
                                    await session.memory_store.save_topic(
                                        session_id=session_id,
                                        topic=f"skill-run-{sk.name}",
                                        summary=f"Skill '{sk.name}' {run.status}: {run.error or 'completed successfully'}",
                                        content=result_text[:5000],
                                        tags=["skill-result", sk.name],
                                    )
                                except Exception:
                                    pass  # best-effort, don't fail the skill run

                            log.info("background_skill_done", skill=sk.name)
                            await reply_fn(result_text)
                            # Store as pending result instead of writing to session
                            # directly — prevents context contamination when the user
                            # sends new messages while the skill was running.
                            self._pending_results.setdefault(session_id, []).append(
                                f"[Background skill '{sk.name}' completed]\n{result_text}"
                            )
                            if run.status == "failed":
                                failed_step = next(
                                    (
                                        r
                                        for r in run.step_results
                                        if r.status == "failed"
                                    ),
                                    None,
                                )
                                self._record_error(
                                    "skill_run_failed",
                                    run.error,
                                    skill=sk.name,
                                    step=failed_step.step_id if failed_step else "",
                                )
                                await self._auto_diagnose(
                                    event="skill_run_failed",
                                    error=run.error,
                                    reply_fn=reply_fn,
                                    skill_name=sk.name,
                                    step_id=failed_step.step_id if failed_step else "",
                                    step_output=(
                                        failed_step.output if failed_step else ""
                                    ),
                                    context=f"Task: {task}\nFull result:\n{result_text[:3000]}",
                                )
                        except Exception as e:
                            err = f"Skill '{sk.name}' failed: {e}"
                            log.error(
                                "background_skill_failed", error=str(e), exc_info=True
                            )
                            self._record_error(
                                "background_skill_crashed", str(e), skill=sk.name
                            )
                            await reply_fn(err)
                            self._pending_results.setdefault(session_id, []).append(
                                f"[Background skill '{sk.name}' FAILED: {e}]"
                            )
                            await self._auto_diagnose(
                                event="background_skill_crashed",
                                error=str(e),
                                reply_fn=reply_fn,
                                skill_name=sk.name,
                                context=f"Task: {task}",
                            )
                        finally:
                            self._skill_tasks.pop(session_id, None)

                    bg_task = asyncio.create_task(_run_skill_bg())
                    self._skill_tasks[session_id] = (bg_task, reply_fn)
                    results.append(
                        f"Skill **{skill.name}** is now running in the background "
                        f"({len(skill.steps)} steps). I'll report results when it finishes."
                    )
                else:
                    extra_delegations.append(
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

            elif op["type"] == "propose":
                if "sysadmin" in self.subagents:
                    sysadmin_prompt = (
                        f"Create a new proposed skill file at _proposed/{op['name']}/SKILL.md\n\n"
                        f"Skill name: {op['name']}\n"
                        f"Description: {op['description']}\n"
                        f"Default subagent: {op['subagent']}\n\n"
                        f"The orchestrator drafted these steps (improve them if needed):\n"
                        f"{op['content']}\n\n"
                        "Read agent source files under ../agent/ if you need to understand "
                        "what tools or capabilities are available to each subagent. "
                        "Write the final SKILL.md with proper YAML frontmatter and steps. "
                        "After writing, read the file back and verify the YAML parses correctly."
                    )
                    extra_delegations.append(
                        {"agent": "sysadmin", "task": sysadmin_prompt}
                    )
                    results.append(
                        f"[Skill System] Delegating skill proposal '{op['name']}' to sysadmin..."
                    )
                    log.info("propose_skill_delegated", name=op["name"])
                else:
                    try:
                        raw_yaml = op["content"]
                        raw_yaml = re.sub(
                            r"^```(?:yaml)?\n?",
                            "",
                            raw_yaml.strip(),
                            flags=re.IGNORECASE,
                        )
                        raw_yaml = re.sub(r"\n?```$", "", raw_yaml.strip())
                        list_match = re.search(r"^[ \t]*-[ \t]", raw_yaml, re.MULTILINE)
                        if list_match:
                            raw_yaml = raw_yaml[list_match.start() :]
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
                        results.append(
                            f"[Skill System] Proposed new skill: '{op['name']}' with {len(steps)} steps. "
                            f"Use SKILL APPROVE {op['name']} to activate it."
                        )
                    except Exception as e:
                        log.error("propose_skill_parse_failed", error=str(e))
                        results.append(f"[Skill System] ERROR proposing skill: {e}")

        return results, extra_delegations, structured_ran

    def _execute_knowledge_ops(self, ops: list[dict]) -> list[str]:
        """Execute <learn> operations — save knowledge docs to disk."""
        results = []
        for op in ops:
            try:
                doc = KnowledgeDoc(
                    slug=op["slug"],
                    title=op["title"],
                    tags=op["tags"],
                    agents=op["agents"],
                    url=op.get("url", ""),
                    content=op["content"],
                )
                path = self.knowledge_store.save(doc)
                results.append(
                    f"[Knowledge] Saved '{op['title']}' as {op['slug']}.md "
                    f"(tags: {', '.join(op['tags'])})"
                )
                log.info("knowledge_learned", slug=op["slug"], path=str(path))
            except Exception as e:
                results.append(f"[Knowledge] ERROR saving '{op['slug']}': {e}")
                log.error("knowledge_learn_failed", slug=op["slug"], error=str(e))
        return results

    async def _execute_delegations(
        self,
        delegations: list[dict],
        user_msg: str,
        session_id: str,
    ) -> list[str]:
        """Run subagent delegations concurrently. Returns formatted results.

        Subagents that are already busy are skipped with a status message.
        """
        results: list[str] = []
        to_run: list[dict] = []

        # Guard: skip subagents that are already busy
        for d in delegations:
            agent_name = d["agent"]
            if agent_name in self._active_delegations:
                busy_info = self._active_delegations[agent_name]
                results.append(
                    f"[{agent_name}] BUSY: Currently working on a task "
                    f"for session {busy_info.get('session_id', '?')[:20]}. "
                    f"Try again when it's done."
                )
                log.info(
                    "delegation_skipped_busy",
                    agent=agent_name,
                    busy_session=busy_info.get("session_id"),
                )
            else:
                to_run.append(d)

        if not to_run:
            return results

        # Mark subagents as busy
        for d in to_run:
            self._active_delegations[d["agent"]] = {
                "session_id": session_id,
                "task": d["task"][:200],
                "started_at": time.time(),
            }

        # Inject relevant knowledge into each delegation task
        for d in to_run:
            docs = self.knowledge_store.find_for_task(
                d["task"], agent=d["agent"]
            )
            if docs:
                knowledge_ctx = self.knowledge_store.build_knowledge_context(docs)
                d["task"] = knowledge_ctx + "\n\n" + d["task"]
                log.info(
                    "knowledge_injected",
                    agent=d["agent"],
                    docs=[doc.slug for doc in docs],
                )

        try:
            tasks = [
                self.subagents[d["agent"]].run(d["task"], context=user_msg)
                for d in to_run
            ]
            raw_results = await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            # Clear busy state
            for d in to_run:
                self._active_delegations.pop(d["agent"], None)

        # Reload skills if sysadmin was involved
        if any(d["agent"] == "sysadmin" for d in to_run):
            self.skill_registry.reload()
            self.skills = self.skill_registry.list_active()
            log.info("skills_reloaded_after_sysadmin")

        for d, result in zip(to_run, raw_results):
            if isinstance(result, Exception):
                results.append(
                    f"[{d['agent']}] ERROR: {type(result).__name__}: {result}"
                )
                log.error("subagent_failed", agent=d["agent"], error=str(result))
                self._record_error(
                    "subagent_failed",
                    str(result),
                    agent=d["agent"],
                    task=d["task"][:500],
                    session_id=session_id,
                )
            else:
                sanitized = sanitize_delegation_result(d["agent"], result)
                # validate against ledger
                if self.state_ledger:
                    workspace = str(WORKSPACES_DIR / d["agent"])
                    validation = await self.state_ledger.validate_delegation_result(
                        agent=d["agent"],
                        result=sanitized,
                        workspace=workspace,
                        session_id=session_id,
                    )
                    if validation["issues"]:
                        sanitized += "\n\n⚠️ Validation issues:\n"
                        sanitized += "\n".join(f"- {i}" for i in validation["issues"])
                # Check for persona violations
                violations = check_output_for_violations(sanitized, d["agent"])
                if violations:
                    sanitized += "\n\n⚠️ Persona violations:\n"
                    sanitized += "\n".join(f"- {v}" for v in violations)
                results.append(f"[{d['agent']}] Result:\n{sanitized}")

        # Fire-and-forget auto-diagnosis for failures
        has_errors = any(isinstance(r, Exception) for r in raw_results)
        if has_errors and "sysadmin" in self.subagents:
            error_summary = "; ".join(
                f"{d['agent']}: {r}"
                for d, r in zip(to_run, raw_results)
                if isinstance(r, Exception)
            )
            await self._auto_diagnose(
                event="subagent_delegation_failed",
                error=error_summary,
                context=f"User message: {user_msg[:500]}",
            )

        return results

    def _record_error(
        self,
        event: str,
        error: str,
        **context,
    ) -> None:
        """Record an error to the journal."""
        self.error_journal.record(event, error, context or None)

    async def _auto_diagnose(
        self,
        event: str,
        error: str,
        reply_fn: Callable[[str], Awaitable[None]] | None = None,
        skill_name: str = "",
        step_id: str = "",
        step_output: str = "",
        context: str = "",
    ) -> None:
        """Fire-and-forget: delegate a diagnosis to sysadmin if available."""
        if "sysadmin" not in self.subagents:
            return

        prompt = build_diagnosis_prompt(
            event=event,
            error=error,
            skill_name=skill_name,
            step_id=step_id,
            step_output=step_output,
            context=context,
        )

        async def _run():
            try:
                result = await self.subagents["sysadmin"].run(prompt)
                log.info("auto_diagnose_complete", event=event, result_len=len(result))
                if reply_fn:
                    await reply_fn(
                        f"**Diagnostic report filed** for `{event}`"
                        f"{f' (skill: {skill_name})' if skill_name else ''}. "
                        f"Use `DIAGNOSE` to view reports."
                    )
            except Exception as e:
                log.warning("auto_diagnose_failed", event=event, error=str(e))

        asyncio.create_task(_run())



def _parse_delegations(text: str) -> list[dict]:
    """Extract <delegate agent="name" ...>task</delegate> blocks."""
    pattern = r'<delegate\s+agent="(\w+)"[^>]*>(.*?)</delegate>'
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


_DELEGATION_COUNTER = 0


def _extract_preamble(response_content: str) -> str:
    """Extract the LLM's conversational text before any operation tags.

    When the orchestrator responds with something like:
        "Sure, I'll look into the sync issues and fix them.
         <delegate agent="coder">...</delegate>"

    This extracts "Sure, I'll look into the sync issues and fix them."
    Returns empty string if there's no meaningful preamble.
    """
    # Find the first operation tag
    tag_pattern = r"<(?:delegate|execute_skill|learn_skill|propose_skill|analyze_image|generate_image|edit_image|recall|remember|learn)\s"
    match = re.search(tag_pattern, response_content)
    if not match:
        return ""

    preamble = response_content[: match.start()].strip()

    # Strip thinking tags if present
    preamble = re.sub(r"<thinking>.*?</thinking>", "", preamble, flags=re.DOTALL).strip()

    # Too short to be useful (e.g. just "OK" or empty)
    if len(preamble) < 10:
        return ""

    return preamble


def _summarize_delegations(delegations: list[dict]) -> str:
    """Turn delegation dicts into a short, natural status message.

    Speaks as a single voice — no mention of subagents or delegation.
    Extracts the gist of each task and varies the phrasing so it
    doesn't feel like a template.
    """
    global _DELEGATION_COUNTER

    snippets: list[str] = []
    for d in delegations:
        first_line = ""
        for line in d["task"].strip().splitlines():
            line = line.strip()
            if line and not line.startswith(("Using your skill", "##", "```")):
                first_line = line
                break
        if not first_line:
            first_line = d["task"].strip().splitlines()[0].strip()

        if len(first_line) > 100:
            first_line = first_line[:97] + "..."
        # Lowercase so it reads mid-sentence
        first_line = first_line[0].lower() + first_line[1:] if first_line else "this"
        snippets.append(first_line)

    joined = "; ".join(snippets) if len(snippets) > 1 else snippets[0]

    # Rotate through varied phrasings so consecutive messages don't sound identical
    openers = [
        f"Let me {joined}",
        f"Looking into this — {joined}",
        f"Pulling that together now — {joined}",
        f"Okay, {joined}",
        f"Sure — {joined}",
        f"Working on that — {joined}",
        f"Give me a moment — {joined}",
    ]
    msg = openers[_DELEGATION_COUNTER % len(openers)]
    _DELEGATION_COUNTER += 1
    return msg


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
    text = re.sub(r'<delegate\s+agent="\w+"[^>]*>.*?</delegate>', "", text, flags=re.DOTALL)
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
    text = re.sub(r'<recall\s+topic="[^"]*">.*?</recall>', "", text, flags=re.DOTALL)
    text = re.sub(r'<remember\s+topic="[^"]*"[^>]*>.*?</remember>', "", text, flags=re.DOTALL)
    text = re.sub(r'<learn\s+slug="[^"]*"[^>]*>.*?</learn>', "", text, flags=re.DOTALL)
    return text.strip()


def _parse_knowledge_ops(text: str) -> list[dict]:
    """Extract <learn> operations from LLM output."""
    ops = []
    learn_pattern = (
        r'<learn\s+slug="([^"]+)"\s+title="([^"]+)"\s+tags="([^"]*)"\s+'
        r'agents="([^"]*)"(?:\s+url="([^"]*)")?\s*>(.*?)</learn>'
    )
    for slug, title, tags_str, agents_str, url, content in re.findall(
        learn_pattern, text, re.DOTALL
    ):
        tags = [t.strip() for t in tags_str.split(",") if t.strip()]
        agents = [a.strip() for a in agents_str.split(",") if a.strip()]
        ops.append({
            "type": "learn",
            "slug": slug.strip(),
            "title": title.strip(),
            "tags": tags,
            "agents": agents or ["*"],
            "url": url.strip() if url else "",
            "content": content.strip(),
        })
    return ops


def _infer_skill_execution(text: str, registry: "SkillRegistry") -> dict | None:
    """Detect when the LLM references running a skill without proper XML.

    Returns an execute op dict if a known skill name appears in context
    that suggests the LLM intended to trigger it, or None.
    """
    text_lower = text.lower()

    # Only trigger if the response looks like it's claiming to run something
    action_phrases = [
        "running",
        "executing",
        "starting",
        "triggered",
        "launching",
        "is now running",
        "in the background",
    ]
    if not any(phrase in text_lower for phrase in action_phrases):
        return None

    for skill in registry.list_active():
        if not skill.has_steps:
            continue
        # Check skill name appears in the response
        if skill.name.lower() in text_lower:
            # Extract any surrounding context as the task
            task_context = text.strip()
            return {"type": "execute", "name": skill.name, "task": task_context}

    return None
