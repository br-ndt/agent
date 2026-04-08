"""Knowledge base — structured reference docs that subagents can learn and recall.

Sits between memory (short text pointers) and skills (action sequences).
Knowledge docs are markdown files in knowledge/ with YAML frontmatter,
indexed by tags and keywords so the orchestrator can inject relevant
context into a subagent's prompt before delegation.

Subagents can also write back knowledge via a <learn> operation,
distilling their workspace output into reusable reference docs.
"""

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import structlog

log = structlog.get_logger()

BASE_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"

# Cap how much knowledge text we inject into a single subagent prompt
MAX_INJECT_CHARS = 6000


@dataclass
class KnowledgeDoc:
    """A single knowledge document parsed from a markdown file."""

    slug: str  # filename without extension, e.g. "crown-game"
    title: str
    tags: list[str] = field(default_factory=list)
    agents: list[str] = field(default_factory=list)  # which agents this is relevant to ("*" = all)
    url: str = ""  # associated URL if any
    content: str = ""  # the markdown body (below frontmatter)
    updated_at: float = 0.0
    file_path: Path = field(default_factory=lambda: Path())

    def matches_query(self, query: str) -> float:
        """Score how well this doc matches a search query. Higher = better."""
        query_lower = query.lower()
        words = set(w for w in query_lower.split() if len(w) > 2)

        score = 0.0

        # Title match
        if query_lower in self.title.lower():
            score += 10.0
        for w in words:
            if w in self.title.lower():
                score += 3.0

        # Tag match
        for tag in self.tags:
            if tag.lower() in query_lower or query_lower in tag.lower():
                score += 5.0
            for w in words:
                if w in tag.lower():
                    score += 2.0

        # URL match
        if self.url and query_lower in self.url.lower():
            score += 4.0
        for w in words:
            if self.url and w in self.url.lower():
                score += 1.5

        # Content keyword match (lighter weight)
        content_lower = self.content.lower()
        for w in words:
            if w in content_lower:
                score += 0.5

        return score

    def matches_agent(self, agent_name: str) -> bool:
        """Check if this doc is relevant to a specific agent."""
        if not self.agents or "*" in self.agents:
            return True
        return agent_name in self.agents


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from a markdown file.

    Returns (metadata_dict, body_content).
    """
    import yaml

    match = re.match(r'^---\s*\n(.*?)\n---\s*\n', text, re.DOTALL)
    if not match:
        return {}, text

    try:
        meta = yaml.safe_load(match.group(1)) or {}
    except Exception:
        meta = {}

    body = text[match.end():]
    return meta, body


def _build_frontmatter(doc: KnowledgeDoc) -> str:
    """Build YAML frontmatter string from a KnowledgeDoc."""
    import yaml

    meta = {
        "title": doc.title,
        "tags": doc.tags,
    }
    if doc.agents:
        meta["agents"] = doc.agents
    if doc.url:
        meta["url"] = doc.url
    if doc.updated_at:
        meta["updated_at"] = doc.updated_at

    return f"---\n{yaml.dump(meta, default_flow_style=False).strip()}\n---\n\n{doc.content}"


class KnowledgeStore:
    """Indexes and retrieves knowledge documents from the knowledge/ directory."""

    def __init__(self, knowledge_dir: Path | None = None):
        self.knowledge_dir = knowledge_dir or KNOWLEDGE_DIR
        self._docs: dict[str, KnowledgeDoc] = {}
        self._loaded = False

    def load(self):
        """Scan knowledge/ directory and index all markdown files."""
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        self._docs.clear()

        for path in sorted(self.knowledge_dir.glob("*.md")):
            try:
                doc = self._parse_file(path)
                self._docs[doc.slug] = doc
            except Exception as e:
                log.warning("knowledge_parse_failed", path=str(path), error=str(e))

        self._loaded = True
        log.info("knowledge_loaded", count=len(self._docs))

    def _parse_file(self, path: Path) -> KnowledgeDoc:
        """Parse a single knowledge markdown file."""
        text = path.read_text()
        meta, body = _parse_frontmatter(text)

        slug = path.stem  # filename without .md

        tags = meta.get("tags", [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",")]

        agents = meta.get("agents", ["*"])
        if isinstance(agents, str):
            agents = [a.strip() for a in agents.split(",")]

        return KnowledgeDoc(
            slug=slug,
            title=meta.get("title", slug.replace("-", " ").title()),
            tags=tags,
            agents=agents,
            url=meta.get("url", ""),
            content=body.strip(),
            updated_at=meta.get("updated_at", path.stat().st_mtime),
            file_path=path,
        )

    # ── Query ────────────────────────────────────────────────

    def search(
        self, query: str, agent: str = "", limit: int = 3, min_score: float = 0.0
    ) -> list[KnowledgeDoc]:
        """Find knowledge docs matching a query, optionally filtered by agent."""
        if not self._loaded:
            self.load()

        scored = []
        for doc in self._docs.values():
            if agent and not doc.matches_agent(agent):
                continue
            score = doc.matches_query(query)
            if score > min_score:
                scored.append((score, doc))

        scored.sort(key=lambda x: -x[0])
        return [doc for _, doc in scored[:limit]]

    def get(self, slug: str) -> KnowledgeDoc | None:
        """Get a specific knowledge doc by slug."""
        if not self._loaded:
            self.load()
        return self._docs.get(slug)

    def list_all(self) -> list[KnowledgeDoc]:
        """List all knowledge docs."""
        if not self._loaded:
            self.load()
        return list(self._docs.values())

    def find_for_task(self, task: str, agent: str = "") -> list[KnowledgeDoc]:
        """Find knowledge docs relevant to a delegation task.

        Uses the task text as the search query. Requires a minimum score
        to avoid injecting loosely-matched knowledge into unrelated tasks.
        """
        return self.search(task, agent=agent, limit=3, min_score=8.0)

    # ── Write ────────────────────────────────────────────────

    def save(self, doc: KnowledgeDoc) -> Path:
        """Save a knowledge doc to disk and update the index."""
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)

        doc.updated_at = time.time()
        if not doc.file_path or doc.file_path == Path():
            doc.file_path = self.knowledge_dir / f"{doc.slug}.md"

        doc.file_path.write_text(_build_frontmatter(doc))
        self._docs[doc.slug] = doc

        log.info(
            "knowledge_saved",
            slug=doc.slug,
            tags=doc.tags,
            content_len=len(doc.content),
        )
        return doc.file_path

    def delete(self, slug: str) -> bool:
        """Delete a knowledge doc."""
        doc = self._docs.pop(slug, None)
        if doc and doc.file_path.exists():
            doc.file_path.unlink()
            log.info("knowledge_deleted", slug=slug)
            return True
        return False

    # ── Prompt building ──────────────────────────────────────

    def build_knowledge_context(self, docs: list[KnowledgeDoc]) -> str:
        """Build a context string from matching knowledge docs for prompt injection."""
        if not docs:
            return ""

        parts = ["## Relevant Knowledge\n"]
        total_chars = 0

        for doc in docs:
            header = f"### {doc.title}"
            if doc.url:
                header += f" ({doc.url})"
            header += f"\nTags: {', '.join(doc.tags)}\n"

            content = doc.content
            remaining = MAX_INJECT_CHARS - total_chars
            if remaining <= 0:
                break
            if len(header) + len(content) > remaining:
                content = content[:remaining - len(header) - 20] + "\n[...truncated]"

            parts.append(header)
            parts.append(content)
            parts.append("")  # blank line separator
            total_chars += len(header) + len(content)

        return "\n".join(parts)

    def build_index_prompt(self) -> str:
        """Build a lightweight index for the orchestrator's system prompt.

        Lists available knowledge topics so the orchestrator knows what's
        available and can reference them in delegations.
        """
        if not self._loaded:
            self.load()

        if not self._docs:
            return ""

        lines = ["\n## Knowledge Base"]
        lines.append(
            "You have access to reference knowledge that can be injected into "
            "subagent prompts. Knowledge is automatically matched to delegation "
            "tasks by keywords and tags. You can also create new knowledge with:\n"
            '<learn slug="topic-name" title="Title" tags="tag1,tag2" '
            'agents="agent1,agent2" url="optional-url">detailed knowledge content</learn>\n'
        )
        lines.append("Available knowledge topics:")
        for doc in sorted(self._docs.values(), key=lambda d: d.slug):
            tags_str = f" [{', '.join(doc.tags[:5])}]" if doc.tags else ""
            agents_str = f" (for: {', '.join(doc.agents)})" if doc.agents and "*" not in doc.agents else ""
            line = f"- **{doc.slug}**: {doc.title}{tags_str}{agents_str}"
            if len(line) > 150:
                line = line[:147] + "..."
            lines.append(line)

        return "\n".join(lines)
