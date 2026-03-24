---
name: code-writer
description: "Write, debug, and refactor code"
triggers: ["write code", "fix bug", "refactor", "implement", "build", "create a script"]
subagent: coder
allowed_tools: [bash, file_ops]
---

# Code writer

You are writing code for the user. Follow these rules:

## Style
- Always include error handling (try/except in Python, try/catch in JS)
- Write type hints for all function signatures (Python)
- Keep functions under 30 lines; extract helpers for complex logic
- Prefer standard library over third-party dependencies when possible
- Use descriptive variable names; avoid single-letter names except loop counters

## Structure
- One module = one responsibility
- Imports at the top, grouped: stdlib → third-party → local
- Constants in UPPER_SNAKE_CASE at module level

## Testing
- When writing a non-trivial function, include at least one test case
- Use pytest conventions: `test_` prefix, assertions with clear messages

## Output
- Return the complete file contents, not diffs or partial snippets
- If modifying existing code, show the full modified file
- Include a brief comment at the top explaining what the file does
