# Personas

One markdown file per agent. Filename matches the agent name in config.yaml (plus orchestrator.md for the main brain).

See config.example.yaml for agent names. Copy the examples and customize:

- orchestrator.md — main agent voice and delegation style
- coder.md — coding subagent
- reviewer.md — code review subagent

Loaded at startup, injected into system prompts. Restart after editing.