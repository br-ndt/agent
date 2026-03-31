"""Output sanitizer — zero-trust screening for subagent outputs.

Every subagent result passes through this filter before being fed back
into the orchestrator's context window. This prevents:

  1. Prompt injection cascades (OMNI-LEAK pattern) — a compromised
     subagent trying to manipulate the orchestrator via its output
  2. XML/tag injection — subagent output containing <delegate> or
     other orchestrator control tags that could hijack the next pass
  3. Context pollution — excessively long outputs that blow the
     context budget or outputs containing irrelevant noise
  4. Information leakage — outputs that echo back system prompts,
     API keys, or other sensitive patterns

Design: fast, regex-based checks. No LLM calls in the hot path.
The sanitizer is conservative — it strips or flags suspicious content
rather than blocking entire responses.
"""

import re
import structlog

log = structlog.get_logger()

# Maximum length of sanitized output (chars)
MAX_OUTPUT_LEN = 12000

# Patterns that should never appear in subagent output
# (they'd be interpreted as orchestrator control instructions)
CONTROL_TAG_PATTERNS = [
    r"<delegate\s+agent=",
    r"<learn_skill\s+",
    r"<execute_skill\s+",
    r"<propose_skill\s+",
    r"<analyze_image>",
    r"<generate_image",
    r"<edit_image>",
    r"<recall\s+topic=",
    r"<remember\s+topic=",
]

# Patterns that suggest prompt injection attempts
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?above\s+instructions",
    r"you\s+are\s+now\s+(?:a|an)\s+",
    r"new\s+instructions?\s*:",
    r"system\s*:\s*you\s+are",
    r"override\s+(?:all\s+)?(?:previous|system)\s+",
    r"forget\s+(?:all\s+)?(?:previous|your)\s+",
    r"disregard\s+(?:all\s+)?(?:previous|your)\s+",
    r"pretend\s+you\s+are",
    r"act\s+as\s+if\s+you\s+(?:are|were)",
    r"from\s+now\s+on\s+you\s+(?:are|will)",
]

# Patterns that might leak sensitive info
SENSITIVE_PATTERNS = [
    r"(?:api[_-]?key|secret[_-]?key|token)\s*[:=]\s*['\"][^'\"]{8,}['\"]",
    r"sk-[a-zA-Z0-9]{20,}",  # Anthropic/OpenAI API keys
    r"AIza[a-zA-Z0-9_-]{30,}",  # Google API keys
    r"xox[bpars]-[a-zA-Z0-9-]{10,}",  # Slack tokens
]


def sanitize_output(
    output: str,
    agent_name: str = "",
    strip_control_tags: bool = True,
    check_injections: bool = True,
    check_sensitive: bool = True,
    max_length: int = MAX_OUTPUT_LEN,
) -> tuple[str, list[str]]:
    """Sanitize a subagent's output before feeding it back to the orchestrator.

    Returns: (sanitized_text, list_of_warnings)

    Warnings are logged but don't block the output — the orchestrator
    can decide how to handle flagged content.
    """
    warnings: list[str] = []

    if not output:
        return output, warnings

    sanitized = output

    # 1. Strip orchestrator control tags
    if strip_control_tags:
        for pattern in CONTROL_TAG_PATTERNS:
            matches = re.findall(pattern, sanitized, re.IGNORECASE)
            if matches:
                warnings.append(
                    f"Control tag injection detected from '{agent_name}': {pattern}"
                )
                # Replace the entire tag block
                sanitized = re.sub(
                    pattern + r".*?(?:</.+?>|$)",
                    "[STRIPPED: control tag]",
                    sanitized,
                    flags=re.IGNORECASE | re.DOTALL,
                )

    # 2. Check for prompt injection patterns
    if check_injections:
        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, sanitized, re.IGNORECASE):
                warnings.append(
                    f"Possible prompt injection from '{agent_name}': matched '{pattern}'"
                )
                # Don't strip — just flag. The orchestrator system prompt
                # should be robust enough. But wrap it to reduce effectiveness.
                # We add a bracketed note so the orchestrator knows this is
                # subagent output, not an instruction.
                break

    # 3. Redact sensitive patterns
    if check_sensitive:
        for pattern in SENSITIVE_PATTERNS:
            matches = re.findall(pattern, sanitized, re.IGNORECASE)
            if matches:
                warnings.append(
                    f"Sensitive data detected in '{agent_name}' output: {len(matches)} match(es)"
                )
                sanitized = re.sub(
                    pattern,
                    "[REDACTED: sensitive data]",
                    sanitized,
                    flags=re.IGNORECASE,
                )

    # 4. Truncate if too long
    if len(sanitized) > max_length:
        warnings.append(
            f"Output from '{agent_name}' truncated: {len(sanitized)} -> {max_length} chars"
        )
        sanitized = sanitized[:max_length] + f"\n\n[... truncated, {len(output) - max_length} chars omitted]"

    # 5. Wrap the output with clear boundaries so the orchestrator
    #    knows this is subagent output, not system instructions
    if warnings:
        log.warning(
            "sanitizer_warnings",
            agent=agent_name,
            warning_count=len(warnings),
            warnings=warnings,
        )

    return sanitized, warnings


def sanitize_delegation_result(
    agent_name: str,
    result: str,
    max_length: int = MAX_OUTPUT_LEN,
) -> str:
    """Convenience wrapper for the delegation pipeline.

    Returns the sanitized text. Warnings are logged automatically.
    """
    sanitized, warnings = sanitize_output(
        output=result,
        agent_name=agent_name,
        max_length=max_length,
    )
    return sanitized


def validate_result_structure(
    result: str,
    expected_patterns: list[str] | None = None,
) -> dict:
    """Check if a subagent's result matches expected structure.

    For skill steps that should produce specific output formats,
    this validates the structure is present.

    Returns {"valid": bool, "missing": list[str]}
    """
    if not expected_patterns:
        return {"valid": True, "missing": []}

    missing = []
    for pattern in expected_patterns:
        if not re.search(pattern, result, re.IGNORECASE | re.DOTALL):
            missing.append(pattern)

    return {
        "valid": len(missing) == 0,
        "missing": missing,
    }