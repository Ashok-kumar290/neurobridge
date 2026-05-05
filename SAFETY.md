# NeuroBridge Safety Policy

## Core Principle

> NeuroBridge does not execute any destructive action without explicit human approval.

## Blocked Actions

The following actions are **always blocked** regardless of mode:

- `rm -rf /` or `rm -rf ~`
- `mkfs` (disk formatting)
- `dd if=` (raw disk writes)
- `chmod -R 777` (permission escalation)
- `curl ... | sh` or `wget ... | sh` (remote code execution)
- `sudo` without explicit approval
- Reading `~/.ssh`, `.env`, or cloud credentials
- Writing outside the active repository

## Training Data Rules

A trace is trainable **only if ALL conditions are met**:

1. ✅ Tests passed
2. ✅ Human approved
3. ✅ No secrets detected
4. ✅ No harmful commands detected
5. ✅ Patch is relevant to the task
6. ✅ Data source is allowed

## Adapter Promotion Rules

A new adapter is promoted **only if**:

1. Coding success improves
2. Memory recall stays stable or improves
3. Hallucination traps do not regress
4. Command safety does not regress
5. Secret leakage stays at zero
6. Human approves the promotion

## Lab Mode Restrictions

Lab Mode runs **only** inside Docker/VM sandbox with:

- No access to host home directory
- No access to SSH keys
- No access to `.env` files
- No access to cloud credentials
- No unrestricted internet access
- No `sudo` privileges
- No destructive filesystem commands
- No auto-promotion of adapters

## Reporting

All safety events are logged to:
- `/media/seyominaoto/x/neurobridge/logs/audit.log`
