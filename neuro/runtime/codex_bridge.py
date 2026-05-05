"""Codex CLI bridge — invokes OpenAI Codex as a terminal coding agent.

Codex CLI runs as a subprocess in suggest mode, receives expert packets,
and returns patches. NeuroBridge captures the output.
"""

from __future__ import annotations

import subprocess
import shutil
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

console = Console()


@dataclass
class CodexResponse:
    """Response from Codex CLI."""

    content: str
    exit_code: int
    success: bool
    model: str = "codex-cli"


class CodexBridge:
    """Bridge to OpenAI Codex CLI agent."""

    def __init__(self) -> None:
        self.executable = shutil.which("codex")
        self.available = self.executable is not None

    def is_available(self) -> bool:
        """Check if Codex CLI is installed."""
        return self.available

    def invoke(
        self,
        prompt: str,
        cwd: Path | None = None,
        approval_mode: str = "suggest",
    ) -> CodexResponse:
        """Invoke Codex CLI with a prompt.

        Uses --quiet flag for non-interactive output.
        approval_mode: "suggest" (default), "auto-edit", or "full-auto"
        """
        if not self.available:
            return CodexResponse(
                content="Codex CLI is not installed. Run: npm install -g @openai/codex",
                exit_code=1,
                success=False,
            )

        cmd = [
            self.executable,
            "--quiet",
            "--approval-mode", approval_mode,
            prompt,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(cwd) if cwd else None,
                timeout=300,
            )

            return CodexResponse(
                content=result.stdout.strip() or result.stderr.strip(),
                exit_code=result.returncode,
                success=result.returncode == 0,
            )
        except subprocess.TimeoutExpired:
            return CodexResponse(
                content="Codex CLI timed out after 5 minutes.",
                exit_code=-1,
                success=False,
            )
        except Exception as e:
            return CodexResponse(
                content=f"Error invoking Codex CLI: {e}",
                exit_code=-1,
                success=False,
            )

    def code(
        self,
        task: str,
        context: str = "",
        cwd: Path | None = None,
    ) -> CodexResponse:
        """Send a coding task to Codex with optional context."""
        if context:
            prompt = f"{context}\n\n---\n\nTask: {task}"
        else:
            prompt = task

        return self.invoke(prompt, cwd=cwd)
