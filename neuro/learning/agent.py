"""ReAct-style tool-use agent — the local model can iterate using real tools.

This is the capability lever that lets a 7B model punch above its weight:
instead of generating one shot and hoping, the model thinks → calls a tool →
observes the real result → continues. This grounds reasoning in actual
computation (run_python), filesystem (read/write), and memory (recall).

Protocol per turn (the model produces ONE of these blocks):

    THOUGHT: <reasoning>
    ACTION: <tool_name>
    INPUT:
    ```
    <tool input>
    ```

After each ACTION the agent appends:

    OBSERVATION:
    ```
    <tool output>
    ```

The model emits a final answer with:

    FINAL: <answer>

Stop conditions: FINAL emitted, step budget, or repeated identical actions.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()


SYSTEM_PROMPT = """You are an expert problem-solving agent with access to real tools.

You think step by step and use tools to ground your reasoning in actual computation.
Never guess when you can compute. Never assume when you can verify.

## Available Tools

- `run_python` — Execute Python code. Use this for math, data manipulation, simulations,
  searching graphs, brute-force verification. You have numpy, sympy, networkx, math, itertools.
- `read_file` — View a file's first 200 lines. Input: file path.
- `write_file` — Create/overwrite a file. Input format: first line is path, then `---`, then content.
- `list_files` — List directory entries. Input: directory path (default ".").
- `search_memory` — Search past expert solutions for similar problems. Input: query string.
- `run_tests` — Execute pytest in the current directory. Input: optional path filter.

## Output Protocol

Every turn produce exactly ONE of these two formats:

### Format A (use a tool):
THOUGHT: <one-paragraph reasoning about what to do next>
ACTION: <one of: run_python, read_file, write_file, list_files, search_memory, run_tests>
INPUT:
```
<tool input here>
```

### Format B (final answer):
THOUGHT: <reasoning about why you are confident>
FINAL: <your final answer to the user>

## Rules

1. NEVER skip the THOUGHT line. Always think before acting.
2. **MANDATORY**: You MUST use `run_python` AT LEAST ONCE before emitting FINAL on any problem with a numerical answer. Pure analytical reasoning is FORBIDDEN — you must verify by actual computation.
3. For math problems: write Python code that COMPUTES the answer (BFS, brute-force, sympy, simulation). Do NOT derive answers by hand and skip computation. Your hand-derivations are often wrong; the code is the source of truth.
4. **Always print() the answer** in your Python code. Code that just evaluates an expression without print() produces no observation.
5. For coding tasks: write code with `write_file`, then run it with `run_python` or `run_tests`.
6. If a tool gives an error, read the error carefully and try a different approach.
7. Be concise. Each THOUGHT should be 1-3 sentences.
8. Only emit FINAL after you have an OBSERVATION confirming your answer. Never emit FINAL without supporting evidence from a tool call.
"""


@dataclass
class AgentStep:
    """One step in the agent's transcript."""
    thought: str = ""
    action: str = ""  # tool name or "FINAL"
    input: str = ""   # tool input or final answer
    observation: str = ""
    duration_ms: float = 0.0


@dataclass
class AgentResult:
    problem: str
    final_answer: str = ""
    steps: list[AgentStep] = field(default_factory=list)
    success: bool = False
    total_time_ms: float = 0.0
    stop_reason: str = ""  # "FINAL" | "budget" | "repeat" | "error"


# ─── Tools ────────────────────────────────────────────────────────────────────


def tool_run_python(code: str, cwd: Path | None = None, timeout: int = 20) -> str:
    """Execute Python code in a subprocess. Returns stdout+stderr."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        script_path = f.name
    try:
        result = subprocess.run(
            ["python3", script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
            cwd=str(cwd) if cwd else None,
        )
        out = result.stdout.strip()
        err = result.stderr.strip()
        parts = []
        if out:
            parts.append(f"STDOUT:\n{out}")
        if err:
            parts.append(f"STDERR:\n{err}")
        if not parts:
            parts.append(f"(no output, exit code {result.returncode})")
        full = "\n\n".join(parts)
        # Truncate to keep context manageable
        if len(full) > 4000:
            full = full[:4000] + f"\n... [truncated, total {len(full)} chars]"
        return full
    except subprocess.TimeoutExpired:
        return f"ERROR: execution timed out after {timeout}s"
    except Exception as e:
        return f"ERROR: {e}"
    finally:
        try:
            Path(script_path).unlink(missing_ok=True)
        except Exception:
            pass


def tool_read_file(path: str, cwd: Path | None = None, max_lines: int = 200) -> str:
    """Read the first max_lines of a file."""
    p = (cwd / path) if cwd else Path(path)
    if not p.exists():
        return f"ERROR: file not found: {p}"
    if not p.is_file():
        return f"ERROR: not a file: {p}"
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        truncated = len(lines) > max_lines
        body = "\n".join(f"{i+1:4d}\t{ln}" for i, ln in enumerate(lines[:max_lines]))
        if truncated:
            body += f"\n... [{len(lines) - max_lines} more lines]"
        return body
    except Exception as e:
        return f"ERROR reading {p}: {e}"


def tool_write_file(input_block: str, cwd: Path | None = None) -> str:
    """Write content to a file. Input format:
        <path>
        ---
        <content>
    """
    if "---" not in input_block:
        return "ERROR: input must have format `path\\n---\\n<content>`"
    head, _, content = input_block.partition("---")
    path = head.strip().splitlines()[0].strip()
    if not path:
        return "ERROR: missing path"
    content = content.lstrip("\n")
    p = (cwd / path) if cwd else Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"OK: wrote {len(content)} bytes to {p}"
    except Exception as e:
        return f"ERROR writing {p}: {e}"


def tool_list_files(path: str = ".", cwd: Path | None = None, max_entries: int = 50) -> str:
    """List directory entries."""
    p = (cwd / path) if cwd else Path(path or ".")
    if not p.exists():
        return f"ERROR: not found: {p}"
    if not p.is_dir():
        return f"ERROR: not a directory: {p}"
    try:
        entries = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        lines = []
        for e in entries[:max_entries]:
            kind = "d" if e.is_dir() else "f"
            try:
                size = e.stat().st_size if e.is_file() else "-"
            except Exception:
                size = "?"
            lines.append(f"{kind}  {size:>10}  {e.name}")
        result = "\n".join(lines) if lines else "(empty)"
        if len(entries) > max_entries:
            result += f"\n... [{len(entries) - max_entries} more entries]"
        return result
    except Exception as e:
        return f"ERROR listing {p}: {e}"


def tool_search_memory(query: str, top_k: int = 3) -> str:
    """Search experience memory for similar past solutions."""
    try:
        from neuro.learning.adaptive_mind import AdaptiveMind
        mind = AdaptiveMind(model="super-qwen:3b", use_steering=False, auto_learn=False)
        experiences = mind.memory.recall(query, top_k=top_k)
        if not experiences:
            return "(no relevant experiences in memory)"
        parts = []
        for i, exp in enumerate(experiences, 1):
            parts.append(
                f"### Experience {i} (source={exp.source}, quality={exp.combined_score():.2f})\n"
                f"Q: {exp.query[:200]}\n"
                f"A: {exp.response[:600]}"
            )
        return "\n\n".join(parts)
    except Exception as e:
        return f"ERROR searching memory: {e}"


def tool_run_tests(path: str = "", cwd: Path | None = None, timeout: int = 60) -> str:
    """Run pytest. Optionally restricted to a path."""
    cmd = ["python3", "-m", "pytest", "-x", "--tb=short", "-q"]
    if path.strip():
        cmd.append(path.strip())
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
            cwd=str(cwd) if cwd else None,
        )
        out = (result.stdout + "\n" + result.stderr).strip()
        if len(out) > 4000:
            out = out[:4000] + f"\n... [truncated]"
        return out + f"\n\n(exit code {result.returncode})"
    except subprocess.TimeoutExpired:
        return f"ERROR: pytest timed out after {timeout}s"
    except Exception as e:
        return f"ERROR running pytest: {e}"


# ─── Agent loop ────────────────────────────────────────────────────────────────


# Regex to parse the model's structured output
ACTION_BLOCK_RE = re.compile(
    r"ACTION:\s*(\w+)\s*\n"
    r"INPUT:\s*\n"
    r"```(?:\w+)?\n?(.*?)```",
    re.DOTALL,
)
FINAL_RE = re.compile(r"FINAL:\s*(.+?)(?:\n\n|\Z)", re.DOTALL)
THOUGHT_RE = re.compile(r"THOUGHT:\s*(.+?)(?=\n(?:ACTION:|FINAL:)|\Z)", re.DOTALL)


class Agent:
    """ReAct-style tool-use agent."""

    TOOLS: dict[str, Callable] = {
        "run_python": lambda inp, cwd: tool_run_python(inp, cwd=cwd),
        "read_file": lambda inp, cwd: tool_read_file(inp.strip(), cwd=cwd),
        "write_file": lambda inp, cwd: tool_write_file(inp, cwd=cwd),
        "list_files": lambda inp, cwd: tool_list_files(inp.strip() or ".", cwd=cwd),
        "search_memory": lambda inp, cwd: tool_search_memory(inp.strip()),
        "run_tests": lambda inp, cwd: tool_run_tests(inp.strip(), cwd=cwd),
    }

    def __init__(
        self,
        model: str = "super-qwen:7b",
        max_steps: int = 12,
        cwd: Path | None = None,
        verbose: bool = True,
        temperature: float = 0.2,
    ):
        self.model = model
        self.max_steps = max_steps
        self.cwd = cwd
        self.verbose = verbose
        self.temperature = temperature
        self._ollama = None

    def _get_ollama(self):
        if self._ollama is None:
            from neuro.runtime.ollama_client import get_ollama_client
            self._ollama = get_ollama_client()
        return self._ollama

    def _call_model(self, transcript: str) -> str:
        """Send the full transcript to the model and get the next turn."""
        client = self._get_ollama()
        resp = client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": transcript},
            ],
            temperature=self.temperature,
        )
        return resp.content

    def _parse_turn(self, text: str) -> tuple[str, str, str]:
        """Parse model output. Returns (thought, action, action_input).
        action is "" if no action found, "FINAL" if final answer.
        """
        thought_m = THOUGHT_RE.search(text)
        thought = thought_m.group(1).strip() if thought_m else ""

        final_m = FINAL_RE.search(text)
        if final_m:
            return thought, "FINAL", final_m.group(1).strip()

        action_m = ACTION_BLOCK_RE.search(text)
        if action_m:
            return thought, action_m.group(1).strip(), action_m.group(2)

        # Fallback: no structured output
        return thought, "", text.strip()

    def _print_step(self, step: AgentStep, num: int) -> None:
        if not self.verbose:
            return
        if step.action == "FINAL":
            console.print(Panel(
                f"[bold]Thought:[/bold] {step.thought}\n\n"
                f"[bold green]FINAL:[/bold green] {step.input}",
                title=f"[bold cyan]Step {num} — Final[/bold cyan]",
                border_style="green",
            ))
        else:
            preview = step.input[:200] + ("..." if len(step.input) > 200 else "")
            obs_preview = step.observation[:300] + ("..." if len(step.observation) > 300 else "")
            console.print(Panel(
                f"[bold]Thought:[/bold] {step.thought}\n\n"
                f"[bold yellow]Action:[/bold yellow] {step.action}\n"
                f"[dim]Input:[/dim] {preview}\n\n"
                f"[bold blue]Observation:[/bold blue]\n{obs_preview}",
                title=f"[bold cyan]Step {num}[/bold cyan]",
                border_style="cyan",
            ))

    def run(self, problem: str) -> AgentResult:
        """Execute the ReAct loop on a problem."""
        start = time.time()
        result = AgentResult(problem=problem)
        transcript = f"# Problem\n{problem}\n\n# Begin\n"

        last_action_sig = None
        repeat_count = 0
        tools_used = 0  # count of successful tool calls

        for step_num in range(1, self.max_steps + 1):
            t0 = time.time()
            try:
                response = self._call_model(transcript)
            except Exception as e:
                result.stop_reason = f"model_error: {e}"
                break

            thought, action, action_input = self._parse_turn(response)
            step = AgentStep(
                thought=thought,
                action=action,
                input=action_input,
                duration_ms=(time.time() - t0) * 1000,
            )

            if action == "FINAL":
                # Guard: don't accept FINAL without any tool usage
                if tools_used == 0:
                    step.observation = (
                        "REJECTED: You attempted to give a FINAL answer without using any tools. "
                        "You MUST use `run_python` to actually compute and verify your answer first. "
                        "Hand-derivations are often wrong. Use code as the source of truth."
                    )
                    result.steps.append(step)
                    transcript += (
                        f"\nTHOUGHT: {thought}\nFINAL: {action_input}\n\n"
                        f"REJECTED:\n{step.observation}\n\n"
                    )
                    self._print_step(step, step_num)
                    continue

                result.final_answer = action_input
                result.success = True
                result.stop_reason = "FINAL"
                result.steps.append(step)
                self._print_step(step, step_num)
                break

            if not action:
                # Model produced unstructured output — coax it back
                step.observation = (
                    "ERROR: Your last response did not follow the required protocol. "
                    "Use exactly THOUGHT: ... ACTION: ... INPUT: ... or FINAL: ..."
                )
                result.steps.append(step)
                transcript += f"\n{response}\n\n{step.observation}\n\n"
                self._print_step(step, step_num)
                continue

            if action not in self.TOOLS:
                step.observation = (
                    f"ERROR: unknown tool '{action}'. "
                    f"Available: {', '.join(self.TOOLS.keys())}"
                )
                result.steps.append(step)
                transcript += (
                    f"\nTHOUGHT: {thought}\nACTION: {action}\nINPUT:\n```\n{action_input}\n```\n"
                    f"OBSERVATION:\n```\n{step.observation}\n```\n\n"
                )
                self._print_step(step, step_num)
                continue

            # Detect repeated identical action
            sig = (action, action_input.strip())
            if sig == last_action_sig:
                repeat_count += 1
                if repeat_count >= 2:
                    step.observation = (
                        "WARNING: You repeated the exact same action. "
                        "Reflect on why this approach isn't working and try something different, "
                        "or emit FINAL if you have enough information."
                    )
                    result.steps.append(step)
                    transcript += (
                        f"\nTHOUGHT: {thought}\nACTION: {action}\nINPUT:\n```\n{action_input}\n```\n"
                        f"OBSERVATION:\n```\n{step.observation}\n```\n\n"
                    )
                    self._print_step(step, step_num)
                    continue
            else:
                repeat_count = 0
            last_action_sig = sig

            # Execute tool
            tool_fn = self.TOOLS[action]
            try:
                observation = tool_fn(action_input, self.cwd)
                tools_used += 1
            except Exception as e:
                observation = f"ERROR executing {action}: {e}"
            step.observation = observation
            result.steps.append(step)

            transcript += (
                f"\nTHOUGHT: {thought}\nACTION: {action}\nINPUT:\n```\n{action_input}\n```\n"
                f"OBSERVATION:\n```\n{observation}\n```\n\n"
            )
            self._print_step(step, step_num)

        else:
            # Loop exhausted
            result.stop_reason = "budget"
            # Use last thought as best-effort answer
            if result.steps:
                last = result.steps[-1]
                result.final_answer = last.thought or "(no answer produced within step budget)"

        result.total_time_ms = (time.time() - start) * 1000

        # Learn from successful runs
        if result.success and result.final_answer:
            try:
                from neuro.learning.adaptive_mind import AdaptiveMind
                mind = AdaptiveMind(model=self.model, use_steering=False, auto_learn=False)
                mind.memory.learn(
                    query=problem,
                    response=result.final_answer,
                    source="agent",
                    quality_score=0.85,
                )
            except Exception:
                pass

        return result

    def display_summary(self, result: AgentResult) -> None:
        status = "[green]SUCCESS[/green]" if result.success else f"[yellow]{result.stop_reason}[/yellow]"
        console.print(Panel(
            f"[bold]Problem:[/bold] {result.problem}\n"
            f"[bold]Status:[/bold] {status} | "
            f"[bold]Steps:[/bold] {len(result.steps)} | "
            f"[bold]Time:[/bold] {result.total_time_ms/1000:.1f}s\n\n"
            f"[bold green]Answer:[/bold green]\n{result.final_answer}",
            title="[bold cyan]Agent Result[/bold cyan]",
            border_style="green" if result.success else "yellow",
        ))
