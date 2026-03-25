"""Skill loader — reads markdown skill files from `skills/` directory.

Each skill is a subfolder with a `SKILL.md` containing YAML frontmatter and markdown instructions.
Skills are assigned to specific subagents and describe how to use tools or follow rules.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
import structlog

log = structlog.get_logger()

DEFAULT_SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


@dataclass
class Skill:
    name: str
    description: str
    triggers: list[str] = field(default_factory=list)
    subagent: str = ""
    allowed_tools: list[str] = field(default_factory=list)
    content: str = ""
    path: Path | None = None

    def to_markdown(self) -> str:
        """Serialize skill back to markdown with frontmatter."""
        frontmatter = {
            "name": self.name,
            "description": self.description,
            "triggers": self.triggers,
            "subagent": self.subagent,
            "allowed_tools": self.allowed_tools,
        }
        fm_str = yaml.dump(frontmatter, sort_keys=False).strip()
        return f"---\n{fm_str}\n---\n\n{self.content}"


def load_skills(skills_dir: Path | None = None) -> list[Skill]:
    """Load all skills from the directory.

    Each skill is expected at `skills_dir/<skill_folder>/SKILL.md`.
    """
    directory = skills_dir or DEFAULT_SKILLS_DIR
    skills: list[Skill] = []

    if not directory.exists():
        log.debug("skills_dir_not_found", path=str(directory))
        return skills

    # Look in each subfolder for SKILL.md
    for skill_folder in directory.iterdir():
        if not skill_folder.is_dir():
            continue

        skill_file = skill_folder / "SKILL.md"
        if not skill_file.exists():
            continue

        try:
            raw_text = skill_file.read_text(encoding="utf-8")
            if not raw_text.startswith("---"):
                log.warning("skill_missing_frontmatter", path=str(skill_file))
                continue

            # Split frontmatter and content
            _, fm_raw, content = raw_text.split("---", 2)
            fm = yaml.safe_load(fm_raw)
            content = content.strip()

            skill = Skill(
                name=fm.get("name", skill_folder.name),
                description=fm.get("description", ""),
                triggers=fm.get("triggers", []),
                subagent=fm.get("subagent", ""),
                allowed_tools=fm.get("allowed_tools", []),
                content=content,
                path=skill_file,
            )
            skills.append(skill)
            log.info("skill_loaded", name=skill.name, subagent=skill.subagent)
        except Exception as e:
            log.error("skill_load_failed", path=str(skill_file), error=str(e))

    return skills


def save_skill(skill: Skill, skills_dir: Path | None = None) -> Path:
    """Save a skill to the directory."""
    directory = skills_dir or DEFAULT_SKILLS_DIR
    skill_folder = directory / skill.name.lower().replace(" ", "-")
    skill_folder.mkdir(parents=True, exist_ok=True)

    skill_file = skill_folder / "SKILL.md"
    skill_file.write_text(skill.to_markdown(), encoding="utf-8")
    log.info("skill_saved", name=skill.name, path=str(skill_file))
    return skill_file
