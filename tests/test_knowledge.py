"""Tests for agent.knowledge — knowledge base system."""

import pytest
from pathlib import Path

from agent.knowledge import (
    KnowledgeStore,
    KnowledgeDoc,
    _parse_frontmatter,
    _build_frontmatter,
)


@pytest.fixture
def knowledge_dir(tmp_path):
    """Create a temporary knowledge directory."""
    kdir = tmp_path / "knowledge"
    kdir.mkdir()
    return kdir


@pytest.fixture
def store(knowledge_dir):
    """Create a KnowledgeStore backed by a temp dir."""
    s = KnowledgeStore(knowledge_dir=knowledge_dir)
    return s


@pytest.fixture
def seeded_store(store, knowledge_dir):
    """Store with a couple of docs pre-loaded."""
    (knowledge_dir / "crown-game.md").write_text(
        "---\n"
        "title: Crown Game\n"
        "tags:\n"
        "  - game\n"
        "  - crown\n"
        "  - sebland\n"
        "agents:\n"
        "  - playwright\n"
        "  - coder\n"
        "url: https://sebland.com/crown\n"
        "---\n\n"
        "# Crown Game\n"
        "A territory control game. POST /api/join to join.\n"
    )
    (knowledge_dir / "chess-rules.md").write_text(
        "---\n"
        "title: Chess Rules\n"
        "tags:\n"
        "  - game\n"
        "  - chess\n"
        "agents:\n"
        "  - '*'\n"
        "---\n\n"
        "# Chess\nStandard chess rules.\n"
    )
    store.load()
    return store


class TestFrontmatter:
    def test_parse_valid(self):
        text = "---\ntitle: Test\ntags:\n  - a\n  - b\n---\n\nBody content"
        meta, body = _parse_frontmatter(text)
        assert meta["title"] == "Test"
        assert meta["tags"] == ["a", "b"]
        assert body.strip() == "Body content"

    def test_parse_no_frontmatter(self):
        text = "Just some text without frontmatter"
        meta, body = _parse_frontmatter(text)
        assert meta == {}
        assert body == text

    def test_roundtrip(self):
        doc = KnowledgeDoc(
            slug="test-doc",
            title="Test Document",
            tags=["tag1", "tag2"],
            agents=["coder"],
            url="https://example.com",
            content="Some content here.",
        )
        text = _build_frontmatter(doc)
        meta, body = _parse_frontmatter(text)
        assert meta["title"] == "Test Document"
        assert meta["tags"] == ["tag1", "tag2"]
        assert meta["agents"] == ["coder"]
        assert meta["url"] == "https://example.com"
        assert body.strip() == "Some content here."


class TestKnowledgeDoc:
    def test_matches_query_title(self):
        doc = KnowledgeDoc(slug="crown-game", title="Crown Game", tags=["game"])
        score = doc.matches_query("crown")
        assert score > 0

    def test_matches_query_tag(self):
        doc = KnowledgeDoc(slug="test", title="Test", tags=["chess", "game"])
        score = doc.matches_query("chess")
        assert score > 0

    def test_matches_query_url(self):
        doc = KnowledgeDoc(slug="test", title="Test", url="https://sebland.com/crown")
        score = doc.matches_query("sebland")
        assert score > 0

    def test_no_match(self):
        doc = KnowledgeDoc(slug="test", title="Test", tags=["alpha"])
        score = doc.matches_query("zzzunrelated")
        assert score == 0

    def test_matches_agent_wildcard(self):
        doc = KnowledgeDoc(slug="test", title="Test", agents=["*"])
        assert doc.matches_agent("anything") is True

    def test_matches_agent_specific(self):
        doc = KnowledgeDoc(slug="test", title="Test", agents=["playwright", "coder"])
        assert doc.matches_agent("playwright") is True
        assert doc.matches_agent("researcher") is False

    def test_matches_agent_empty_means_all(self):
        doc = KnowledgeDoc(slug="test", title="Test", agents=[])
        assert doc.matches_agent("anything") is True


class TestKnowledgeStore:
    def test_load_empty(self, store):
        store.load()
        assert store.list_all() == []

    def test_load_from_disk(self, seeded_store):
        docs = seeded_store.list_all()
        assert len(docs) == 2
        slugs = {d.slug for d in docs}
        assert "crown-game" in slugs
        assert "chess-rules" in slugs

    def test_get_by_slug(self, seeded_store):
        doc = seeded_store.get("crown-game")
        assert doc is not None
        assert doc.title == "Crown Game"
        assert "game" in doc.tags
        assert "sebland.com" in doc.url

    def test_get_nonexistent(self, seeded_store):
        assert seeded_store.get("nonexistent") is None

    def test_search_by_keyword(self, seeded_store):
        results = seeded_store.search("crown game")
        assert len(results) >= 1
        assert results[0].slug == "crown-game"

    def test_search_by_tag(self, seeded_store):
        results = seeded_store.search("chess")
        assert len(results) >= 1
        assert results[0].slug == "chess-rules"

    def test_search_filtered_by_agent(self, seeded_store):
        # playwright should see crown-game but chess is for "*" so also visible
        results = seeded_store.search("game", agent="playwright")
        slugs = {d.slug for d in results}
        assert "crown-game" in slugs

    def test_find_for_task(self, seeded_store):
        docs = seeded_store.find_for_task(
            "Play the crown game at sebland.com", agent="playwright"
        )
        assert len(docs) >= 1
        assert docs[0].slug == "crown-game"

    def test_save_new_doc(self, store, knowledge_dir):
        store.load()
        doc = KnowledgeDoc(
            slug="new-topic",
            title="New Topic",
            tags=["test"],
            agents=["*"],
            content="Some new knowledge.",
        )
        path = store.save(doc)
        assert path.exists()
        assert path.name == "new-topic.md"

        # Should be findable now
        found = store.get("new-topic")
        assert found is not None
        assert found.title == "New Topic"

    def test_save_updates_existing(self, seeded_store):
        doc = seeded_store.get("crown-game")
        doc.content = "Updated content about crown."
        seeded_store.save(doc)

        reloaded = seeded_store.get("crown-game")
        assert "Updated content" in reloaded.content

    def test_delete(self, seeded_store, knowledge_dir):
        assert seeded_store.delete("crown-game") is True
        assert seeded_store.get("crown-game") is None
        assert not (knowledge_dir / "crown-game.md").exists()

    def test_delete_nonexistent(self, seeded_store):
        assert seeded_store.delete("nonexistent") is False


class TestPromptBuilding:
    def test_build_knowledge_context_empty(self, store):
        store.load()
        assert store.build_knowledge_context([]) == ""

    def test_build_knowledge_context(self, seeded_store):
        docs = seeded_store.search("crown")
        ctx = seeded_store.build_knowledge_context(docs)
        assert "Relevant Knowledge" in ctx
        assert "Crown Game" in ctx
        assert "sebland.com" in ctx

    def test_build_index_prompt(self, seeded_store):
        prompt = seeded_store.build_index_prompt()
        assert "Knowledge Base" in prompt
        assert "crown-game" in prompt
        assert "chess-rules" in prompt

    def test_build_index_prompt_empty(self, store):
        store.load()
        assert store.build_index_prompt() == ""


class TestParseKnowledgeOps:
    """Test the orchestrator's <learn> tag parsing."""

    def test_parse_learn_tag(self):
        from agent.orchestrator import _parse_knowledge_ops

        text = (
            '<learn slug="tic-tac-toe" title="Tic Tac Toe Rules" '
            'tags="game,ttt" agents="playwright,coder" '
            'url="https://example.com/ttt">'
            "X and O take turns. Three in a row wins."
            "</learn>"
        )
        ops = _parse_knowledge_ops(text)
        assert len(ops) == 1
        assert ops[0]["slug"] == "tic-tac-toe"
        assert ops[0]["title"] == "Tic Tac Toe Rules"
        assert ops[0]["tags"] == ["game", "ttt"]
        assert ops[0]["agents"] == ["playwright", "coder"]
        assert ops[0]["url"] == "https://example.com/ttt"
        assert "Three in a row wins" in ops[0]["content"]

    def test_parse_learn_no_url(self):
        from agent.orchestrator import _parse_knowledge_ops

        text = (
            '<learn slug="test" title="Test" tags="a" agents="*">'
            "content"
            "</learn>"
        )
        ops = _parse_knowledge_ops(text)
        assert len(ops) == 1
        assert ops[0]["url"] == ""

    def test_parse_no_learn_tags(self):
        from agent.orchestrator import _parse_knowledge_ops

        ops = _parse_knowledge_ops("Just some normal text.")
        assert ops == []
