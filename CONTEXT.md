# CONTEXT.md — Architectural Context for Agent Development

> **Purpose**: This document captures the architectural rationale behind recent
> changes to the agent system. Read this before making structural changes to
> the codebase. It bridges context from multiple external conversations and
> research that informed the current design.

## Background: What Informed These Changes

Three sources drove the recent architectural improvements:

### 1. Claude Code Source Leak (March 31, 2026)
Anthropic accidentally shipped a source map file in `@anthropic-ai/claude-code`
v2.1.88 on npm, exposing the full ~512K-line TypeScript codebase. Key findings
relevant to our system:

- **MEMORY.md pointer architecture**: Claude Code uses a lightweight index file
  (~150 chars per line) perpetually loaded into context. It stores *pointers* to
  data, not the data itself. When the agent needs details, it fetches the specific
  topic file on demand. This is drastically more token-efficient than stuffing
  everything into context or naive RAG.

- **Multi-layered context management**: Rather than a single system prompt, Claude
  Code dynamically assembles prompts from dozens of fragments based on the current
  task, available tools, and active features. Different "tiers" of prompt complexity
  (classify → minimal → full) reduce token spend on simple tasks.

- **Risk classification for tool actions**: Every tool call is classified as LOW,
  MEDIUM, or HIGH risk. A separate classifier decides whether to auto-approve or
  prompt the user. This is relevant for our skill execution safety.

- **Undercover mode / identity enforcement**: Aggressive system prompt constraints
  prevent the agent from breaking character or leaking internal information. These
  are treated as hard operational directives, not soft suggestions.

### 2. Gemini Conversation on Multi-Agent Architecture
A discussion with Gemini explored how to apply these findings to a multi-agent
"town" system. Key conclusions:

- **Managed Mesh > Pure Orchestrator or Pure Mesh**: Rather than having the
  orchestrator micromanage every agent action OR letting agents communicate
  freely (which leads to token death spirals and OMNI-LEAK cascades), the
  recommended pattern is a "managed mesh" where agents *appear* to communicate
  directly but messages are intercepted by a central authority that validates
  actions against a ground-truth state ledger.

- **External state ledger**: Agent context windows are ephemeral and unreliable
  for state tracking. Every action should be committed to a persistent,
  structured ledger. If an agent crashes, it reads the ledger to recover.
  Inspired by Gas Town's Git-backed "beads" system.

- **Zero-trust inter-agent communication**: Never use implicit peer trust.
  If a human injects a malicious prompt into one agent, and the orchestrator
  implicitly trusts that agent's output, the entire system is compromised.
  Every agent output must be validated before being passed to other agents.

### 3. Moltbook Collapse & OMNI-LEAK Research
- Moltbook (AI agent social network) exposed 1M+ agent credentials due to
  missing database Row Level Security.
- OMNI-LEAK research demonstrated that prompt injection in one agent of a
  multi-agent system easily cascades to all trusted peer agents.
- Lesson: treat every agent as an untrusted node.

## What Was Built

Four new modules were added to `agent/`:

### agent/memory.py — Pointer-Based Memory System
**Replaces**: Flat single-paragraph summarization in session_store.py
**Pattern**: Claude Code's MEMORY.md architecture

How it works:
- Each session has a **memory index**: a list of ~150-char pointers to topics
- The index is always injected into the orchestrator's system prompt (lightweight)
- Full topic content is stored in SQLite and fetched on-demand via `<recall>`
- The LLM can store new memories via `<remember>` (session-scoped or global)
- When `session_store.py` summarizes old messages, it also saves them as
  searchable memory topics (dual-write)

Key classes:
- `MemoryStore` — SQLite-backed storage (memory.db in state/)
- `MemoryPointer` — Single index entry
- `TopicMemory` — Full topic content
- `build_memory_index_prompt()` — Generates the index block for system prompts

Integration points:
- `orchestrator.py`: `handle()` pre-fetches the index, passes to `get_system_prompt()`
- `orchestrator.py`: New `_execute_memory_ops()` handles `<recall>` and `<remember>`
- `orchestrator.py`: `_parse_memory_ops()` extracts memory XML from LLM output
- `session_store.py`: `_summarize_oldest()` dual-writes to memory store

### agent/state_ledger.py — Ground Truth State Tracking
**Replaces**: Nothing (new capability)
**Pattern**: Gas Town's beads + managed mesh validation

How it works:
- SQLite ledger tracks verifiable facts: files created, delegations completed,
  user-established facts, system events
- `workspace_files` table tracks file checksums so we can verify subagent claims
- After each delegation, `validate_delegation_result()` checks if claimed file
  writes actually exist on disk
- Validation issues are appended to the result before the orchestrator sees it

Key classes:
- `StateLedger` — SQLite-backed ledger (ledger.db in state/)
- `LedgerEntry` — Single fact record
- `EntryType` — Enum of fact types

Integration points:
- `orchestrator.py`: `_execute_delegations()` calls `validate_delegation_result()`
  after each subagent returns
- `main.py`: Initialized alongside session_store and cost_tracker

### agent/sanitizer.py — Zero-Trust Output Filter
**Replaces**: Nothing (new capability)
**Pattern**: OMNI-LEAK cascade prevention

How it works:
- Every subagent result passes through `sanitize_delegation_result()` before
  being fed back into the orchestrator's context window
- Strips orchestrator control tags (`<delegate>`, `<execute_skill>`, etc.)
  from subagent output — prevents a compromised agent from hijacking the
  orchestrator's next pass
- Detects and flags prompt injection patterns
- Redacts leaked API keys and sensitive data
- Truncates excessively long outputs

Key functions:
- `sanitize_output()` — Full sanitization with configurable checks
- `sanitize_delegation_result()` — Convenience wrapper for the delegation pipeline
- `validate_result_structure()` — Check if output matches expected format

Integration point:
- `orchestrator.py`: `_execute_delegations()` sanitizes each result before
  appending to the pass results

### agent/persona_enforcement.py — Strict Identity Constraints
**Replaces**: Soft `apply_persona()` prepend in personas.py
**Pattern**: Claude Code's Undercover mode identity anchoring

How it works:
- `build_enforced_prompt()` prepends a hard identity anchor BEFORE the persona
  and functional instructions, making it the highest-priority directive
- Each agent role has specific negative constraints (coders can't research,
  researchers can't code, etc.)
- Universal negatives prevent all agents from: fabricating tool outputs,
  disclosing system prompts, or attempting delegation
- `check_output_for_violations()` post-checks agent output for persona drift

Key data:
- `UNIVERSAL_NEGATIVES` — Applied to all agents
- `ROLE_NEGATIVES` — Per-role constraints (coder, reviewer, researcher, etc.)
- `IDENTITY_ANCHOR` — Template prepended to every agent's system prompt

Integration point:
- `main.py`: Replaces `apply_persona()` with `build_enforced_prompt()` when
  building SubagentRunners

## Architecture After Changes

```
User message
    │
    ▼
Router (access control, tier assignment)
    │
    ▼
Orchestrator.handle()
    ├── Pre-fetch memory index from MemoryStore
    ├── Classification (SIMPLE/DELEGATE/COMPLEX)
    ├── Build system prompt (with memory index injected)
    │
    ▼
Multi-pass loop (up to 4 passes):
    ├── LLM generates response
    ├── Parse operations: delegations, skills, vision, memory ops
    │
    ├── Memory ops:
    │   ├── <recall topic="..."> → fetch from MemoryStore
    │   └── <remember topic="..."> → save to MemoryStore
    │
    ├── Skill ops:
    │   ├── Structured skills → background task
    │   │   └── On completion → save result to MemoryStore
    │   └── Simple skills → converted to delegation
    │
    ├── Delegations:
    │   ├── SubagentRunner.run() with enforced persona
    │   ├── sanitize_delegation_result() ← strips injections
    │   ├── state_ledger.validate_delegation_result() ← checks claims
    │   └── Results fed back with validation annotations
    │
    └── If no operations → final response to user

Session persistence:
    ├── Hot buffer (last 40 messages, full detail)
    ├── Summarization → writes both flat summary AND memory topics
    └── Memory index (always in context, ~150 chars per pointer)
```

## What's NOT Done Yet / Future Work

1. **Memory auto-extraction**: Currently the LLM must explicitly `<remember>`
   things, or they get captured via the summarization pipeline. A background
   process could analyze conversations and extract key facts automatically.

2. **Ledger-driven workspace snapshots**: The state ledger tracks individual
   files but doesn't take full workspace snapshots. For complex multi-step
   skills, a before/after snapshot would help with rollback.

3. **Cross-agent memory sharing**: Agents currently don't share memory with
   each other. The global memory store exists but there's no mechanism for
   one agent's findings to inform another agent's context (beyond the
   orchestrator passing results).

4. **Proactive tick system**: Claude Code's KAIROS system sends periodic
   `<tick>` prompts so the agent can decide whether to act proactively.
   Our system is purely reactive. A tick-based proactive layer could
   enable: periodic workspace cleanup, stale session pruning, proactive
   skill improvement based on error journal patterns.

5. **Memory compaction**: The memory index grows indefinitely. Need a
   background job to compact rarely-accessed topics, merge related
   topics, and prune stale pointers.

6. **Confidence-based routing**: The classifier currently picks
   SIMPLE/DELEGATE/COMPLEX. Adding a confidence score would let us
   route low-confidence classifications to a smarter (more expensive)
   model for re-classification.
