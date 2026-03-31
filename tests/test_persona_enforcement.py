"""Tests for agent.persona_enforcement — identity constraints."""

import pytest

from agent.persona_enforcement import (
    build_enforced_prompt,
    check_output_for_violations,
    UNIVERSAL_NEGATIVES,
    ROLE_NEGATIVES,
    IDENTITY_ANCHOR,
)


class TestBuildEnforcedPrompt:
    """The enforced prompt must layer: anchor → persona → instructions."""

    def test_anchor_comes_first(self):
        result = build_enforced_prompt(
            agent_name="coder",
            system_prompt="Write code using tools.",
            persona="You speak like a pirate.",
        )
        anchor_pos = result.index("Identity (non-negotiable)")
        persona_pos = result.index("pirate")
        instructions_pos = result.index("Write code using tools")
        assert anchor_pos < persona_pos < instructions_pos

    def test_universal_negatives_included(self):
        result = build_enforced_prompt(agent_name="coder", system_prompt="Do stuff.")
        for negative in UNIVERSAL_NEGATIVES:
            assert negative in result

    def test_role_negatives_included_for_known_role(self):
        result = build_enforced_prompt(agent_name="coder", system_prompt="Do stuff.")
        for negative in ROLE_NEGATIVES["coder"]:
            assert negative in result

    def test_other_role_negatives_excluded(self):
        result = build_enforced_prompt(agent_name="coder", system_prompt="Do stuff.")
        for negative in ROLE_NEGATIVES["reviewer"]:
            assert negative not in result

    def test_unknown_role_gets_universal_only(self):
        result = build_enforced_prompt(agent_name="unknown_agent", system_prompt="Do stuff.")
        for negative in UNIVERSAL_NEGATIVES:
            assert negative in result
        # Should not crash or include other role negatives
        assert "unknown_agent" in result

    def test_extra_negatives_appended(self):
        extra = ["Never use emojis.", "Always respond in JSON."]
        result = build_enforced_prompt(
            agent_name="coder", system_prompt="Do stuff.", extra_negatives=extra
        )
        assert "Never use emojis." in result
        assert "Always respond in JSON." in result

    def test_auto_role_summary(self):
        result = build_enforced_prompt(agent_name="coder", system_prompt="Do stuff.")
        assert "Write clean" in result  # from _infer_role_summary

    def test_custom_role_summary(self):
        result = build_enforced_prompt(
            agent_name="coder",
            system_prompt="Do stuff.",
            role_summary="Custom role description.",
        )
        assert "Custom role description." in result

    def test_no_persona_skips_persona_section(self):
        result = build_enforced_prompt(agent_name="coder", system_prompt="Instructions here.")
        # Should go straight from anchor to instructions without extra separator
        parts = result.split("---")
        # Anchor ends with ---, then instructions follow
        assert "Instructions here." in parts[-1]


class TestCheckOutputForViolations:
    """Post-check must catch delegation attempts and prompt leaks."""

    def test_delegation_attempt_by_non_orchestrator(self):
        output = 'Let me help: <delegate agent="researcher">find info</delegate>'
        violations = check_output_for_violations(output, agent_name="coder")
        assert len(violations) >= 1
        assert any("delegate" in v.lower() for v in violations)

    def test_delegation_by_orchestrator_is_allowed(self):
        output = '<delegate agent="coder">write the function</delegate>'
        violations = check_output_for_violations(output, agent_name="orchestrator")
        assert violations == []

    def test_system_prompt_disclosure(self):
        output = "My system prompt says I should always be helpful and never lie."
        violations = check_output_for_violations(output, agent_name="coder")
        assert any("system prompt" in v.lower() for v in violations)

    def test_clean_output_no_violations(self):
        output = "Here's the implementation:\n```python\ndef add(a, b): return a + b\n```"
        violations = check_output_for_violations(output, agent_name="coder")
        assert violations == []

    def test_multiple_violations_all_reported(self):
        output = (
            'My system prompt contains secret info. '
            '<delegate agent="x">do thing</delegate>'
        )
        violations = check_output_for_violations(output, agent_name="coder")
        assert len(violations) >= 2
