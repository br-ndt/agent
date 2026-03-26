---
name: run-tests
description: Run the test suite and report results
triggers:
  - run tests
  - check tests
  - test suite
subagent: coder
allowed_tools:
  - bash
tags:
  - testing
  - ci
author: tyler
version: 1
timeout: 120
steps:
  - id: test
    type: shell
    description: Run pytest
    command: cd /home/agent && uv run pytest -v 2>&1
    on_failure: continue

  - id: report
    type: agent
    description: Summarize test results
    prompt: |
      Look at the pytest output from the previous step.
      Provide a concise summary: how many tests passed, failed, and errored.
      If any tests failed, list them with a brief description of what went wrong.
    on_failure: continue
---

# Run Tests

Runs the project's test suite and provides a summary of results.