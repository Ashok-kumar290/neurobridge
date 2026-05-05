"""Claude Code bridge — invokes Claude Code as a terminal coding agent.

Claude Code runs as a subprocess, receives expert packets as prompts,
and returns patches/plans. NeuroBridge captures the output.
"""

from __future__ import annotations

import json
import subprocess
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()


@dataclass
class ClaudeResponse:
    """Response from Claude Code."""

    content: str
    exit_code: int
    success: bool
    model: str = "claude-code"


class ClaudeBridge:
    """Bridge to Claude Code terminal agent."""

    def __init__(self) -> None:
        self.executable = shutil.which("claude")
        self.available = self.executable is not None

    def is_available(self) -> bool:
        """Check if Claude Code is installed."""
        return self.available

    def invoke(
        self,
        prompt: str,
        cwd: Path | None = None,
        max_turns: int = 5,
    ) -> ClaudeResponse:
        """Invoke Claude Code with a prompt.

        Uses --print flag for non-interactive single-response mode.
        """
        if not self.available:
            return ClaudeResponse(
                content="Claude Code is not installed. Run: npm install -g @anthropic-ai/claude-code",
                exit_code=1,
                success=False,
            )

        cmd = [
            self.executable,
            "--print",  # non-interactive, single response
            prompt,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(cwd) if cwd else None,
                timeout=1200,  # 20 min timeout
            )

            return ClaudeResponse(
                content=result.stdout.strip() or result.stderr.strip(),
                exit_code=result.returncode,
                success=result.returncode == 0,
            )
        except subprocess.TimeoutExpired:
            return ClaudeResponse(
                content="Claude Code timed out after 5 minutes.",
                exit_code=-1,
                success=False,
            )
        except Exception as e:
            return ClaudeResponse(
                content=f"Error invoking Claude Code: {e}",
                exit_code=-1,
                success=False,
            )

    def code(
        self,
        task: str,
        context: str = "",
        cwd: Path | None = None,
    ) -> ClaudeResponse:
        """Send a coding task to Claude Code with optional context."""
        if context:
            prompt = f"{context}\n\n---\n\nTask: {task}"
        else:
            prompt = task

        return self.invoke(prompt, cwd=cwd)
