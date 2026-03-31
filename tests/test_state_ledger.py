"""Tests for agent.state_ledger — ground truth state tracking."""

import pytest

from agent.state_ledger import StateLedger, EntryType, LedgerEntry


class TestLedgerRecording:
    """Facts recorded to the ledger must be queryable."""

    @pytest.mark.asyncio
    async def test_record_and_query(self, state_ledger: StateLedger):
        entry_id = await state_ledger.record(
            EntryType.USER_FACT,
            key="user-prefers-dark-mode",
            value="User explicitly said they prefer dark mode",
            session_id="s1",
            agent="orchestrator",
        )
        assert entry_id > 0
        entries = await state_ledger.get_recent(limit=5)
        assert len(entries) == 1
        assert entries[0].key == "user-prefers-dark-mode"
        assert entries[0].entry_type == EntryType.USER_FACT

    @pytest.mark.asyncio
    async def test_filter_by_type(self, state_ledger: StateLedger):
        await state_ledger.record(EntryType.USER_FACT, "k1", "v1")
        await state_ledger.record(EntryType.SYSTEM_EVENT, "k2", "v2")
        await state_ledger.record(EntryType.USER_FACT, "k3", "v3")

        facts = await state_ledger.get_recent(entry_type=EntryType.USER_FACT)
        assert all(e.entry_type == EntryType.USER_FACT for e in facts)
        assert len(facts) == 2

    @pytest.mark.asyncio
    async def test_filter_by_session(self, state_ledger: StateLedger):
        await state_ledger.record(EntryType.USER_FACT, "k1", "v1", session_id="s1")
        await state_ledger.record(EntryType.USER_FACT, "k2", "v2", session_id="s2")
        entries = await state_ledger.get_recent(session_id="s1")
        assert len(entries) == 1
        assert entries[0].key == "k1"

    @pytest.mark.asyncio
    async def test_recent_returns_newest_first(self, state_ledger: StateLedger):
        await state_ledger.record(EntryType.SYSTEM_EVENT, "first", "v")
        await state_ledger.record(EntryType.SYSTEM_EVENT, "second", "v")
        await state_ledger.record(EntryType.SYSTEM_EVENT, "third", "v")
        entries = await state_ledger.get_recent(limit=3)
        assert entries[0].key == "third"
        assert entries[2].key == "first"


class TestFileTracking:
    """File checksums must be tracked and verifiable."""

    @pytest.mark.asyncio
    async def test_record_real_file(self, state_ledger: StateLedger, tmp_path):
        # Create a real file
        test_file = tmp_path / "output.txt"
        test_file.write_text("hello world")

        result = await state_ledger.record_file(str(test_file), agent="coder")
        assert result is True

        # Verify it matches
        verification = await state_ledger.verify_file(str(test_file))
        assert verification["exists"] is True
        assert verification["matches"] is True

    @pytest.mark.asyncio
    async def test_record_missing_file(self, state_ledger: StateLedger, tmp_path):
        fake_path = str(tmp_path / "ghost.txt")
        result = await state_ledger.record_file(fake_path, agent="liar")
        assert result is False

    @pytest.mark.asyncio
    async def test_detect_file_modification(self, state_ledger: StateLedger, tmp_path):
        test_file = tmp_path / "data.txt"
        test_file.write_text("original content")
        await state_ledger.record_file(str(test_file), agent="coder")

        # Modify the file after recording
        test_file.write_text("tampered content")
        verification = await state_ledger.verify_file(str(test_file))
        assert verification["exists"] is True
        assert verification["matches"] is False

    @pytest.mark.asyncio
    async def test_verify_untracked_file(self, state_ledger: StateLedger, tmp_path):
        test_file = tmp_path / "untracked.txt"
        test_file.write_text("I exist but am not tracked")
        verification = await state_ledger.verify_file(str(test_file))
        assert verification["exists"] is True
        assert verification["matches"] is False
        assert "Not in ledger" in verification["details"]

    @pytest.mark.asyncio
    async def test_verify_deleted_after_recording(self, state_ledger: StateLedger, tmp_path):
        test_file = tmp_path / "ephemeral.txt"
        test_file.write_text("now you see me")
        await state_ledger.record_file(str(test_file), agent="coder")
        test_file.unlink()

        verification = await state_ledger.verify_file(str(test_file))
        assert verification["exists"] is False
        assert verification["matches"] is False


class TestDelegationValidation:
    """validate_delegation_result checks subagent claims against disk."""

    @pytest.mark.asyncio
    async def test_valid_file_claim(self, state_ledger: StateLedger, tmp_path):
        # Agent claims to have written a file, and it exists
        test_file = tmp_path / "result.py"
        test_file.write_text("print('hello')")

        result_text = f"I wrote file `{test_file}`"
        validation = await state_ledger.validate_delegation_result(
            agent="coder", result=result_text, session_id="s1"
        )
        assert validation["valid"] is True
        assert str(test_file) in validation["verified_files"]

    @pytest.mark.asyncio
    async def test_false_file_claim(self, state_ledger: StateLedger, tmp_path):
        result_text = "I created file `nonexistent_output.py` with the implementation."
        validation = await state_ledger.validate_delegation_result(
            agent="liar",
            result=result_text,
            workspace=str(tmp_path),
            session_id="s1",
        )
        assert validation["valid"] is False
        assert any("not found" in issue for issue in validation["issues"])

    @pytest.mark.asyncio
    async def test_no_file_claims_is_valid(self, state_ledger: StateLedger):
        result_text = "I analyzed the code and found three potential issues."
        validation = await state_ledger.validate_delegation_result(
            agent="reviewer", result=result_text, session_id="s1"
        )
        assert validation["valid"] is True
        assert validation["issues"] == []

    @pytest.mark.asyncio
    async def test_delegation_recorded_in_ledger(self, state_ledger: StateLedger):
        await state_ledger.validate_delegation_result(
            agent="coder", result="Done with the task.", session_id="s1"
        )
        entries = await state_ledger.get_recent(entry_type=EntryType.DELEGATION_COMPLETED)
        assert len(entries) == 1
        assert entries[0].agent == "coder"


class TestWorkspaceState:
    """get_workspace_state returns tracked files."""

    @pytest.mark.asyncio
    async def test_workspace_state_after_recording(self, state_ledger: StateLedger, tmp_path):
        f1 = tmp_path / "a.py"
        f2 = tmp_path / "b.py"
        f1.write_text("a")
        f2.write_text("b")
        await state_ledger.record_file(str(f1), agent="coder")
        await state_ledger.record_file(str(f2), agent="coder")

        state = await state_ledger.get_workspace_state(str(tmp_path))
        paths = {f["path"] for f in state}
        assert str(f1) in paths
        assert str(f2) in paths


class TestContextSummary:
    """build_context_summary formats entries for injection."""

    def test_empty_entries(self, state_ledger: StateLedger):
        assert state_ledger.build_context_summary([]) == ""

    def test_entries_formatted(self, state_ledger: StateLedger):
        import time
        entries = [
            LedgerEntry(
                entry_type=EntryType.FILE_CREATED,
                key="/workspace/out.py",
                value="size=100, sha256=abc123",
                agent="coder",
                timestamp=time.time(),
            )
        ]
        summary = state_ledger.build_context_summary(entries)
        assert "file_created" in summary
        assert "out.py" in summary
        assert "coder" in summary
