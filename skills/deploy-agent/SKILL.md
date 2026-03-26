---
name: deploy-agent
description: Pull latest code, rebuild, and restart the agent daemon on the VPS
triggers:
  - deploy
  - push to prod
  - ship it
  - redeploy
subagent: coder
allowed_tools:
  - bash
  - file_ops
tags:
  - deployment
  - devops
author: tyler
version: 1
timeout: 180
steps:
  - id: pull
    type: shell
    description: Pull latest code from main
    command: cd /home/agent && git pull origin main
    on_failure: abort

  - id: sync-deps
    type: shell
    description: Sync Python dependencies
    command: cd /home/agent && uv sync
    on_failure: abort

  - id: check-config
    type: agent
    description: Verify config.yaml is valid
    prompt: |
      Read the file /home/agent/config.yaml and check that it's valid YAML.
      Report any syntax errors. If it looks good, say "Config OK".
    on_failure: abort

  - id: restart
    type: shell
    description: Restart the agent systemd service
    command: sudo systemctl restart agent
    on_failure: abort

  - id: verify
    type: agent
    description: Check that the agent came back up healthy
    prompt: |
      Run "systemctl status agent" and "journalctl -u agent -n 20 --no-pager".
      Report whether the service is running and if there are any errors in the
      recent logs. If everything looks good, confirm the deploy succeeded.
    on_failure: continue
---

# Deploy Agent

Deploys the latest version of the multi-agent daemon to the Oracle ARM VPS.

## Safety Rules
- NEVER proceed past the pull step if there are merge conflicts
- If `uv sync` fails, it's usually a dependency issue — report it, don't retry blindly
- The verify step checks systemd status AND recent journal logs

## Rollback
If something goes wrong after restart:
```
sudo systemctl stop agent
cd /home/agent && git checkout HEAD~1
uv sync
sudo systemctl start agent
```