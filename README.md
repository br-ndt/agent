# agent

Multi-agent AI daemon. Chat via Discord (or CLI), with multi-model subagent delegation, persona system, tier-based routing, and resilient provider fallback. Modeled after [seb](https://github.com/sammcgrail/seb) [sven](https://github.com/svenflow/dispatch)

## What this does

A central orchestrator agent receives your messages and either handles them directly or delegates subtasks to specialized subagents — each with its own personality, model, and provider. Claude handles heavy lifting on your Pro/Max subscription (no API cost), while cheaper models like Gemini Flash handle trusted-tier users or fallback scenarios.

## Quick start

```bash
# 1. Run bootstrap (firewall, system deps, Claude CLI)
chmod +x bootstrap.sh && ./bootstrap.sh

# 2. Install Python dependencies
uv sync

# 3. Configure
cp .env.example .env
cp config.example.yaml config.yaml
nano .env                          # Set API keys (see Provider setup below)

# 4. Authenticate Claude CLI
claude login                       # Opens browser for OAuth

# 5. Test provider connectivity
uv run python test_connection.py

# 6. Run interactively
uv run python -m agent.main
```

## Provider setup

The system supports a hybrid provider model — Claude uses your subscription, other models use API keys.

| Provider | Auth method | Cost | .env variable |
|----------|------------|------|---------------|
| Claude (via CLI) | OAuth subscription | $0 (Pro/Max plan) | None needed — run `claude login` |
| Google Gemini | API key | ~$0.075/M tokens | `GOOGLE_API_KEY` |
| OpenAI GPT | API key | ~$2.50/M tokens | `OPENAI_API_KEY` |
| Anthropic API | API key | ~$3/M tokens | `ANTHROPIC_API_KEY` |

At minimum you need Claude CLI authenticated. Add other providers as needed for subagents or fallback.

## Discord setup

1. Go to https://discord.com/developers/applications → New Application
2. **Bot** tab → Reset Token → copy to `.env` as `DISCORD_BOT_TOKEN`
3. **Bot** tab → enable **Message Content Intent** under Privileged Gateway Intents
4. **OAuth2 > URL Generator** → Scopes: `bot` → Permissions: `Send Messages`, `Read Message History`, `View Channel` → copy URL and invite bot to your server
5. Add your Discord user ID to `config.yaml` under `access.admin_ids`
6. The bot responds to DMs and @mentions in channels

## Personas

Each agent has a personality defined in a markdown file in `personas/`:

```
personas/
├── orchestrator.md    # Main agent personality
├── coder.md           # Coding subagent
└── reviewer.md        # Code review subagent
```

Edit these to change voice, tone, and behavioral rules. Personas are injected into system prompts at startup — they work with any provider.

## Tier-based routing

Different users can be routed to different providers and models:

```yaml
# config.yaml
access:
  admin_ids:
    - "discord:123456789"       # You — gets Claude Sonnet
  trusted_ids:
    - "discord:987654321"       # Friends — gets Gemini Flash

  tier_routing:
    admin:
      provider: claude_cli
      model: sonnet
    trusted:
      provider: google
      model: gemini-2.5-flash
```

Unknown users are silently dropped — no response, no information leak.

## Subagent delegation

The orchestrator can delegate complex tasks to specialized subagents. Ask it something like "Write a Python function to check primes, then review it for bugs" and it will fan out to the coder and reviewer subagents in parallel, then synthesize the results.

Subagents are configured in `config.yaml`:

```yaml
subagents:
  coder:
    provider: claude_cli
    model: opus
    personality: "Senior software engineer. Clean code, error handling, type hints."
    tools: [bash, file_ops]

  reviewer:
    provider: claude_cli
    model: sonnet
    personality: "Code reviewer. Bugs, security, readability."
    tools: []
```

## Resilient providers

Every provider call goes through retry logic with exponential backoff, automatic fallback to alternate providers, and a circuit breaker that skips broken providers for 60 seconds. If Claude's auth expires overnight, your Discord bot keeps working on Gemini Flash.

## Session persistence and memory

Conversations persist to SQLite. The last 40 messages stay in the hot buffer as full detail; when the buffer overflows, old messages are evicted into a **pointer-based memory system** (`agent/memory.py`). A lightweight index of topic pointers is always loaded into the orchestrator's system prompt. Full topic content is stored separately in SQLite and fetched on demand via `<recall>` / `<remember>` XML ops. No flat summary is injected into the message list — the memory index handles long-term context without unbounded token growth.

## Security and trust model

The delegation pipeline treats every subagent as an untrusted node:

- **Output sanitizer** (`agent/sanitizer.py`) — Strips orchestrator control tags, flags prompt injection patterns, and redacts leaked API keys from subagent output before the orchestrator sees it.
- **State ledger** (`agent/state_ledger.py`) — Tracks ground-truth file checksums and validates subagent claims ("I wrote foo.py") against disk. False claims are flagged in the result.
- **Persona enforcement** (`agent/persona_enforcement.py`) — Injects hard identity anchors and role-specific negative constraints into subagent prompts. Post-checks output for violations like unauthorized delegation attempts or system prompt disclosure.

See `CONTEXT.md` for architectural rationale.

## Running as a daemon

```bash
# Install the systemd service
sudo cp systemd/agent.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable agent

# Start / stop / restart
sudo systemctl start agent
sudo systemctl stop agent
sudo systemctl restart agent      # After code or config changes

# Logs
sudo journalctl -u agent -f      # Live tail
sudo journalctl -u agent -n 100  # Last 100 lines
```

In daemon mode (`--no-cli`), the agent runs headless with only Discord (and/or Telegram) adapters. Systemd auto-restarts on crash.

## Admin commands (via chat)

| Command   | Effect                          |
|-----------|---------------------------------|
| `STATUS`  | Show active session count       |
| `HEALME`  | Clear all sessions              |
| `RESTART` | Graceful restart (systemd restarts the process) |

## Skills

Modular capability definitions in `skills/`. Each skill is a directory with a `SKILL.md` that gets injected into the relevant subagent's prompt:

```
skills/
├── code-writer/SKILL.md    # Coding rules and patterns
└── reviewer/SKILL.md       # Review checklist and format
```

## Project structure

```
agent/
├── agent/
│   ├── main.py                # Entry point
│   ├── config.py              # Config loader (.env + config.yaml)
│   ├── router.py              # Access control + message dispatch
│   ├── orchestrator.py        # Main agent + subagent delegation
│   ├── subagent_runner.py     # Runs tasks against any provider/model
│   ├── personas.py            # Loads personality markdown files
│   ├── session_store.py       # Persistent sessions with summarization
│   ├── memory.py              # Pointer-based long-term memory (SQLite)
│   ├── state_ledger.py        # Ground-truth state tracking and validation
│   ├── sanitizer.py           # Zero-trust output filter for subagents
│   ├── persona_enforcement.py # Strict identity constraints per role
│   ├── providers/
│   │   ├── base.py            # Abstract provider interface
│   │   ├── anthropic.py       # Claude via API (pay-per-token)
│   │   ├── claude_cli.py      # Claude via CLI (subscription)
│   │   ├── openai.py          # GPT-4o, etc.
│   │   ├── google.py          # Gemini Flash / Pro
│   │   └── resilient.py       # Retry, backoff, fallback wrapper
│   └── adapters/
│       ├── base.py            # Abstract adapter interface
│       ├── cli.py             # Local REPL for development
│       ├── discord.py         # Discord bot
│       └── telegram.py        # Telegram bot
├── personas/                  # Agent personality markdown files
├── skills/                    # Modular skill definitions
├── scripts/                   # Helper scripts
├── state/                     # Runtime state — sessions, logs (gitignored)
├── systemd/                   # Systemd service file
├── tests/                     # Unit and integration tests (pytest)
├── config.yaml                # Agent and subagent configuration
├── .env.example               # API key template
├── bootstrap.sh               # Server setup script
├── test_connection.py         # Provider connectivity test
├── CONTEXT.md                 # Architectural rationale
└── pyproject.toml             # Python dependencies
```

## Tests

```bash
uv run pytest tests/ -v
```

Tests cover the four security/memory modules and their integration through the delegation pipeline. The integration tests in `test_delegation_pipeline.py` mock subagent runners and exercise the full sanitizer -> ledger -> persona chain against real SQLite databases.
