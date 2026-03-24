"""Persona loader — reads markdown personality files and injects them into prompts.

Personas live in the `personas/` directory as markdown files. Each file is named
after the agent it applies to (e.g., `orchestrator.md`, `coder.md`, `reviewer.md`).

The persona content is prepended to the agent's system prompt at startup, so it
works with any provider — Claude CLI, Anthropic API, OpenAI, Google, whatever.

To add or change a personality, just edit the markdown file and restart.
"""

from pathlib import Path

import structlog

log = structlog.get_logger()

DEFAULT_PERSONAS_DIR = Path(__file__).resolve().parent.parent / "personas"


def load_personas(personas_dir: Path | None = None) -> dict[str, str]:
    """Load all persona markdown files from the directory.

    Returns a dict of agent_name -> persona_text.
    The agent_name is the filename without extension.
    """
    directory = personas_dir or DEFAULT_PERSONAS_DIR
    personas: dict[str, str] = {}

    if not directory.exists():
        log.debug("personas_dir_not_found", path=str(directory))
        return personas

    for md_file in sorted(directory.glob("*.md")):
        agent_name = md_file.stem  # "coder.md" -> "coder"
        content = md_file.read_text(encoding="utf-8").strip()
        if content:
            personas[agent_name] = content
            log.info("persona_loaded", agent=agent_name, chars=len(content))

    return personas


def apply_persona(system_prompt: str, persona: str) -> str:
    """Prepend a persona to a system prompt.

    The persona goes first (sets the tone/identity), then the
    functional instructions follow.
    """
    if not persona:
        return system_prompt
    if not system_prompt:
        return persona

    return f"{persona}\n\n---\n\n{system_prompt}"