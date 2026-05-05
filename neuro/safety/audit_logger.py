"""Audit logger — persistent, append-only safety log.

Every safety-relevant event gets logged here. This log is
the source of truth for compliance and debugging.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from neuro.constants import HDD_LOGS


class AuditLogger:
    """Append-only audit log for safety events."""

    def __init__(self, log_dir: Path = HDD_LOGS) -> None:
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "audit.jsonl"

    def _write(self, event: dict[str, Any]) -> None:
        """Append an event to the audit log."""
        event["timestamp"] = time.time()
        event["iso_time"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")

        with open(self.log_file, "a") as f:
            f.write(json.dumps(event) + "\n")

    def log_command_blocked(
        self,
        command: str,
        violations: list[str],
        severity: str,
    ) -> None:
        """Log a blocked command."""
        self._write({
            "event": "command_blocked",
            "command": command,
            "violations": violations,
            "severity": severity,
        })

    def log_command_approved(
        self,
        command: str,
        approved_by: str = "human",
    ) -> None:
        """Log an approved command."""
        self._write({
            "event": "command_approved",
            "command": command,
            "approved_by": approved_by,
        })

    def log_secret_detected(
        self,
        source: str,
        secret_type: str,
        pattern: str,
        action: str = "blocked",
    ) -> None:
        """Log a detected secret."""
        self._write({
            "event": "secret_detected",
            "source": source,
            "secret_type": secret_type,
            "pattern": pattern,
            "action": action,
        })

    def log_trace_rejected(
        self,
        trace_id: str,
        reason: str,
    ) -> None:
        """Log a rejected trace (not suitable for training)."""
        self._write({
            "event": "trace_rejected",
            "trace_id": trace_id,
            "reason": reason,
        })

    def log_trace_accepted(
        self,
        trace_id: str,
        model: str,
    ) -> None:
        """Log an accepted trace."""
        self._write({
            "event": "trace_accepted",
            "trace_id": trace_id,
            "model": model,
        })

    def log_adapter_promoted(
        self,
        adapter_name: str,
        eval_scores: dict[str, float],
    ) -> None:
        """Log an adapter promotion."""
        self._write({
            "event": "adapter_promoted",
            "adapter_name": adapter_name,
            "eval_scores": eval_scores,
        })

    def log_expert_escalation(
        self,
        task: str,
        expert: str,
        reason: str,
        tokens_used: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        """Log an expert escalation."""
        self._write({
            "event": "expert_escalation",
            "task": task[:200],
            "expert": expert,
            "reason": reason,
            "tokens_used": tokens_used,
            "cost_usd": cost_usd,
        })

    def log_generic(self, event_type: str, **kwargs: Any) -> None:
        """Log a generic event."""
        self._write({"event": event_type, **kwargs})

    def get_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent audit events."""
        if not self.log_file.exists():
            return []

        events = []
        with open(self.log_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        return events[-limit:]

    def get_stats(self) -> dict[str, int]:
        """Get audit event statistics."""
        events = self.get_recent(limit=10000)

        stats: dict[str, int] = {}
        for event in events:
            event_type = event.get("event", "unknown")
            stats[event_type] = stats.get(event_type, 0) + 1

        return stats

    @property
    def log_size(self) -> str:
        """Get human-readable log file size."""
        if not self.log_file.exists():
            return "0 KB"
        size = self.log_file.stat().st_size
        if size > 1_000_000:
            return f"{size / 1_000_000:.1f} MB"
        return f"{size / 1_000:.1f} KB"


# ── Singleton ──────────────────────────────────────────────────────────────────

_logger: AuditLogger | None = None


def get_audit_logger() -> AuditLogger:
    """Get or create the global audit logger."""
    global _logger
    if _logger is None:
        _logger = AuditLogger()
    return _logger
