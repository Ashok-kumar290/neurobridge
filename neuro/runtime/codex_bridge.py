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
        self._session_id: str | None = None  # set after first call, used for resume

    def is_available(self) -> bool:
        """Check if Codex CLI is installed."""
        return self.available

    def invoke(
        self,
        prompt: str,
        cwd: Path | None = None,
        bypass_sandbox: bool = True,
        sandbox_mode: str = "danger-full-access",
    ) -> CodexResponse:
        """Invoke Codex CLI with a prompt.

        Uses `codex exec` for non-interactive mode and writes only the
        final message to a temp file (bypasses the verbose transcript).
        """
        if not self.available:
            return CodexResponse(
                content="Codex CLI is not installed. Run: npm install -g @openai/codex",
                exit_code=1,
                success=False,
            )

        import tempfile

        # Capture only the final agent message via -o
        with tempfile.NamedTemporaryFile(
            mode="r", suffix=".txt", delete=False, encoding="utf-8"
        ) as tmp:
            output_file = tmp.name

        # Support resuming an existing session for persistent context
        cmd = [
            self.executable,
            "exec",
        ]
        if self._session_id:
            cmd += ["resume", self._session_id]
        cmd += [
            "--skip-git-repo-check",
            "--color", "never",
            "-o", output_file,
            "-s", sandbox_mode,
        ]
        if bypass_sandbox:
            cmd.append("--dangerously-bypass-approvals-and-sandbox")
        if cwd:
            cmd.extend(["-C", str(cwd)])
        cmd.append(prompt)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(cwd) if cwd else None,
                timeout=300,
                stdin=subprocess.DEVNULL,
            )

            # Read final message from output file
            content = ""
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()
            except Exception:
                pass
            finally:
                try:
                    Path(output_file).unlink(missing_ok=True)
                except Exception:
                    pass

            # Parse session id from stdout for future resume calls
            stdout = result.stdout or ""
            if not self._session_id:
                import re as _re
                m = _re.search(r"session id:\s*([0-9a-f-]+)", stdout, _re.IGNORECASE)
                if m:
                    self._session_id = m.group(1)

            # Fallback: parse stdout (strip wrapper noise)
            if not content:
                lines = [
                    ln for ln in stdout.splitlines()
                    if "[NeuroBridge Proxy]" not in ln
                ]
                content = "\n".join(lines).strip() or result.stderr.strip()

            return CodexResponse(
                content=content,
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
