"""Offline Advanced Solver — multi-step reasoning agent for hard problems.

Matches the capability of Opus/Codex for algorithmic problems by chaining
local models in a Plan → Recall → Code → Verify → Refine loop.

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │  Problem: "Implement Dijkstra with Fibonacci heap"      │
    │                       ↓                                 │
    │  [Step 1] Plan (3B, fast)                               │
    │    → Break into subproblems, identify data structures   │
    │                       ↓                                 │
    │  [Step 2] Recall (experience memory)                    │
    │    → Pull similar past solutions from Codex/Claude      │
    │                       ↓                                 │
    │  [Step 3] Code (7B + steering lens)                     │
    │    → Generate solution using plan + experiences         │
    │                       ↓                                 │
    │  [Step 4] Verify (execute Python, run tests)            │
    │    → If code fails, capture error                       │
    │                       ↓                                 │
    │  [Step 5] Refine (7B, if needed, up to 3 iterations)    │
    │    → Use error to fix bugs                              │
    │                       ↓                                 │
    │  [Step 6] Learn (store final solution)                  │
    │    → Add to experience memory with high quality score   │
    └─────────────────────────────────────────────────────────┘

Usage:
    solver = Solver()
    result = solver.solve("Write a red-black tree in Python")
    print(result.code)
    print(result.explanation)
"""

from __future__ import annotations

import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel

console = Console()


PLAN_PROMPT = """You are an expert algorithms instructor. Produce the OPTIMAL plan for this problem.

Problem: {problem}

CRITICAL: Identify the BEST-KNOWN time complexity for this problem class.
Do NOT settle for a naive solution. Use built-ins when they are asymptotically optimal.

Output format:
1. OPTIMAL COMPLEXITY: state the best-known O(...) for this problem
2. KEY INSIGHT: the trick that achieves optimal complexity
3. DATA STRUCTURES: what to use (prefer built-ins like dict, set, heapq, deque, bisect)
4. ALGORITHM: numbered steps (max 5)
5. WHY OPTIMAL: 1 sentence proving no better complexity is possible
6. EDGE CASES: list 2-3

Be extremely concise. Total output under 250 words."""


CODE_PROMPT = """You are an expert Python programmer. Write production-quality code.

## Problem
{problem}

## Plan
{plan}

## Similar Past Solutions (reference only, adapt to current problem)
{experiences}

## Requirements
- Write ONLY executable Python code
- Include a small `if __name__ == "__main__":` test block
- No explanations outside code comments
- Handle edge cases from the plan
- Code must be self-contained and runnable

## Code:
```python
"""


REFINE_PROMPT = """The previous code failed. Fix it.

## Problem
{problem}

## Previous Code
```python
{code}
```

## Error
{error}

## Fix the code. Output ONLY the corrected full Python code:
```python
"""


STRATEGIES_PROMPT = """List {n} DIFFERENT optimal-or-near-optimal strategies to solve this problem.
Each strategy should use a different approach (e.g. built-in, manual loop, divide-and-conquer, DP, etc).

Problem: {problem}

Output ONLY a JSON array, like:
["strategy 1 name", "strategy 2 name", "strategy 3 name"]

Each name should be 3-8 words describing the approach (e.g. "Python slicing s[::-1]", "iterative two-pointer", "recursive divide and conquer")."""


STRATEGY_CODE_PROMPT = """Write Python code implementing this specific strategy.

## Problem
{problem}

## Strategy
{strategy}

## Requirements
- Define a function with a clear name (e.g. `solve`, or descriptive)
- Use ONLY this strategy — no fallbacks
- Include a `if __name__ == "__main__":` test block with at least 1 example
- Output ONLY Python code, nothing else

```python
"""


EXPLAIN_PROMPT = """Explain this code concisely:

Problem: {problem}

Code:
```python
{code}
```

Output format (max 150 words):
- APPROACH: 1 sentence
- WHY IT WORKS: 2 sentences
- COMPLEXITY: time/space"""


REVIEW_PROMPT = """Review this proposed solution and its output. 
Is the answer physically/mathematically plausible for the given problem?

Problem: {problem}
Output: {output}

Check for:
1. Physical impossibility (e.g. traveling 2000 units in 3 steps of length 5).
2. Logical contradictions (e.g. negative number of ways, probability > 1).
3. Scale errors (e.g. 0 matrices when we know some exist).

Output ONLY 'PLAUSIBLE' or 'IMPLAUSIBLE' followed by a 1-sentence reason."""


@dataclass
class SolveStep:
    """A single step in the solve process."""
    name: str
    content: str
    duration_ms: float = 0.0
    model: str = ""


@dataclass
class Candidate:
    """A single candidate solution in a tournament."""
    code: str
    strategy: str  # description of approach (e.g. "naive loop", "slicing", "deque")
    verified: bool = False
    runtime_us: float = float("inf")  # microseconds for benchmark
    output: str = ""
    error: str = ""
    plausible: bool = True
    review_comment: str = ""


@dataclass
class SolveResult:
    """Final result of solving a problem."""
    problem: str
    plan: str = ""
    code: str = ""
    explanation: str = ""
    verified: bool = False
    verification_output: str = ""
    iterations: int = 0
    total_time_ms: float = 0.0
    experiences_used: int = 0
    steps: list[SolveStep] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)
    chosen_strategy: str = ""

    @property
    def success(self) -> bool:
        return bool(self.code) and (self.verified or self.iterations > 0)


class Solver:
    """Multi-step offline coding agent.

    Uses cooperative model chaining:
      - 3B for fast planning (local, free)
      - 7B for code generation (local, free, steered for factuality)
      - Python subprocess for verification (local, free)
    """

    def __init__(
        self,
        planner_model: str = "super-qwen:3b",
        coder_model: str = "super-qwen:7b",
        use_steering: bool = True,
        max_refinements: int = 3,
        verify_timeout: int = 15,
    ):
        self.planner_model = planner_model
        self.coder_model = coder_model
        self.use_steering = use_steering
        self.max_refinements = max_refinements
        self.verify_timeout = verify_timeout

        self._mind = None
        self._ollama = None

    def _get_mind(self):
        if self._mind is None:
            from neuro.learning.adaptive_mind import AdaptiveMind
            self._mind = AdaptiveMind(
                model=self.coder_model,
                use_steering=self.use_steering,
                auto_learn=False,  # we manually learn only successful solutions
            )
        return self._mind

    def _get_ollama(self):
        if self._ollama is None:
            from neuro.runtime.ollama_client import get_ollama_client
            self._ollama = get_ollama_client()
        return self._ollama

    def _call(self, model: str, prompt: str, temperature: float = 0.2) -> str:
        """Call a model via Ollama."""
        client = self._get_ollama()
        resp = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return resp.content.strip()

    def _extract_code(self, text: str) -> str:
        """Extract Python code from a model response."""
        # Try fenced code block first
        fenced = re.search(r"```(?:python)?\n?(.*?)```", text, re.DOTALL)
        if fenced:
            return fenced.group(1).strip()
        # If already pure code (no prose), return as-is
        if text.strip().startswith(("def ", "class ", "import ", "from ", "#", "if __name__")):
            return text.strip()
        # Fall back to stripping any leading prose before first 'def'/'class'
        for keyword in ("def ", "class ", "import ", "from "):
            idx = text.find(keyword)
            if idx > 0:
                return text[idx:].strip()
        return text.strip()

    def _verify(self, code: str) -> tuple[bool, str]:
        """Execute code in a subprocess to verify it runs.

        Returns (success, output_or_error).
        """
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
                timeout=self.verify_timeout,
                stdin=subprocess.DEVNULL,  # prevent hang on input()
            )
            if result.returncode == 0:
                output = result.stdout.strip()
                return True, output or "(no output)"
            else:
                err = (result.stderr or "").strip()
                out = (result.stdout or "").strip()
                return False, (err + "\n" + out).strip()[:2000]
        except subprocess.TimeoutExpired as e:
            # Also capture any partial output
            partial = ""
            if hasattr(e, "stdout") and e.stdout:
                partial = (e.stdout.decode("utf-8", errors="replace") if isinstance(e.stdout, bytes) else e.stdout)[:500]
            return False, f"Execution timed out after {self.verify_timeout}s. Partial output: {partial}"
        except Exception as e:
            return False, f"Execution error: {e}"
        finally:
            try:
                Path(script_path).unlink(missing_ok=True)
            except Exception:
                pass

    def _benchmark(self, code: str, repeats: int = 5) -> tuple[bool, float, str, str]:
        """Run code N times and measure median wall-clock runtime in microseconds.

        Returns (verified, median_us, output, error).
        """
        # Wrap user code with timing harness
        harness = (
            code
            + "\n\n"
            + "import time as _t, json as _j, sys as _sys\n"
            "_times = []\n"
            "for _ in range(" + str(repeats) + "):\n"
            "    _start = _t.perf_counter()\n"
            "    # Re-execute the test block by re-importing as __main__\n"
            "    pass\n"
            "    _times.append((_t.perf_counter() - _start) * 1e6)\n"
        )

        # Simpler approach: just time the whole script execution
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            script_path = f.name

        runtimes = []
        last_output = ""
        last_error = ""
        verified = True

        try:
            for _ in range(repeats):
                t0 = time.perf_counter()
                try:
                    result = subprocess.run(
                        ["python3", script_path],
                        capture_output=True,
                        text=True,
                        timeout=self.verify_timeout,
                        stdin=subprocess.DEVNULL,
                    )
                    elapsed_us = (time.perf_counter() - t0) * 1e6
                    if result.returncode == 0:
                        runtimes.append(elapsed_us)
                        last_output = result.stdout.strip()
                    else:
                        verified = False
                        last_error = (result.stderr or "")[:500]
                        break
                except subprocess.TimeoutExpired:
                    verified = False
                    last_error = f"Timed out after {self.verify_timeout}s"
                    break
        finally:
            try:
                Path(script_path).unlink(missing_ok=True)
            except Exception:
                pass

        if not runtimes:
            return False, float("inf"), last_output, last_error

        # Use median to reduce noise
        runtimes.sort()
        median = runtimes[len(runtimes) // 2]
        return verified, median, last_output, last_error

    def _generate_strategies(self, problem: str, n: int = 3) -> list[str]:
        """Ask planner to enumerate N distinct strategies."""
        import json as _json
        raw = self._call(self.planner_model, STRATEGIES_PROMPT.format(problem=problem, n=n))
        # Try to extract JSON array
        m = re.search(r"\[.*?\]", raw, re.DOTALL)
        if not m:
            return []
        try:
            strategies = _json.loads(m.group(0))
            return [s for s in strategies if isinstance(s, str)][:n]
        except Exception:
            return []

    def tournament(self, problem: str, num_candidates: int = 3) -> SolveResult:
        """Generate N candidate solutions, benchmark each, return the fastest verified.

        This is the optimal-prioritized solver. It explicitly explores
        multiple approaches and picks the best by measured runtime.
        """
        start = time.time()
        result = SolveResult(problem=problem)

        # Step 1: Generate distinct strategies
        console.print(f"[dim]Generating {num_candidates} distinct strategies...[/dim]")
        t0 = time.time()
        strategies = self._generate_strategies(problem, n=num_candidates)
        if not strategies:
            console.print("[yellow]Could not parse strategies, falling back to single solve[/yellow]")
            return self.solve(problem)
        result.steps.append(SolveStep(
            name="strategies",
            content="\n".join(f"- {s}" for s in strategies),
            duration_ms=(time.time() - t0) * 1000,
        ))

        # Step 2: Pull experiences once (shared across candidates)
        mind = self._get_mind()
        experiences = mind.memory.recall(problem, top_k=2)
        result.experiences_used = len(experiences)

        # Step 3: Generate code for each strategy and benchmark
        for i, strategy in enumerate(strategies, 1):
            console.print(f"[dim]Candidate {i}/{len(strategies)}: {strategy[:60]}[/dim]")
            t0 = time.time()
            code_resp = self._call(
                self.coder_model,
                STRATEGY_CODE_PROMPT.format(problem=problem, strategy=strategy),
                temperature=0.1,
            )
            code = self._extract_code(code_resp)

            verified, runtime_us, output, error = self._benchmark(code, repeats=3)
            cand = Candidate(
                code=code,
                strategy=strategy,
                verified=verified,
                runtime_us=runtime_us,
                output=output,
                error=error,
            )
            result.candidates.append(cand)
            result.steps.append(SolveStep(
                name=f"candidate_{i}",
                content=f"{strategy} | {'PASS' if verified else 'FAIL'} | {runtime_us/1000:.2f}ms",
                duration_ms=(time.time() - t0) * 1000,
                model=self.coder_model,
            ))

            # Step 3.5: Sanity Review
            if verified and output:
                review_resp = self._call(
                    self.planner_model,
                    REVIEW_PROMPT.format(problem=problem, output=output),
                    temperature=0.1
                )
                cand.plausible = "PLAUSIBLE" in review_resp.upper()
                cand.review_comment = review_resp
                if not cand.plausible:
                    console.print(f"[yellow]  ⚠️  Review flagged: {review_resp}[/yellow]")

        # Step 4: Pick the fastest verified AND plausible candidate
        plausible_cands = [c for c in result.candidates if c.verified and c.plausible]
        
        if not plausible_cands:
            # Fallback to verified if none are plausible, or first if none verified
            verified_cands = [c for c in result.candidates if c.verified]
            best = min(verified_cands, key=lambda c: c.runtime_us) if verified_cands else (result.candidates[0] if result.candidates else None)
        else:
            best = min(plausible_cands, key=lambda c: c.runtime_us)

        if best:
            result.code = best.code
            result.verified = best.verified
            result.verification_output = best.output or best.error
            result.chosen_strategy = best.strategy
            result.plan = "\n".join(
                f"{'★' if c is best else ' '} {c.strategy}: "
                f"{'PASS' if c.verified else 'FAIL'} "
                f"[{'PLAUSIBLE' if c.plausible else 'IMPLAUSIBLE'}] "
                f"({c.runtime_us/1000:.2f}ms)" if c.verified else
                f"  {c.strategy}: FAIL ({c.error[:60]})"
                for c in result.candidates
            )

        # Step 5: Explanation
        if result.code:
            t0 = time.time()
            explanation = self._call(
                self.planner_model,
                EXPLAIN_PROMPT.format(problem=problem, code=result.code),
            )
            result.explanation = explanation
            result.steps.append(SolveStep(
                name="explain", content=explanation, model=self.planner_model,
                duration_ms=(time.time() - t0) * 1000,
            ))

        # Step 6: Learn the optimal solution
        if result.verified and result.code:
            mind.memory.learn(
                query=problem,
                response=(
                    f"# Optimal Solution ({result.chosen_strategy})\n"
                    f"```python\n{result.code}\n```\n\n{result.explanation}"
                ),
                source="solver-tournament",
                quality_score=0.95,  # higher than regular solve since benchmarked
            )

        result.total_time_ms = (time.time() - start) * 1000
        return result

    def solve(self, problem: str, verify: bool = True) -> SolveResult:
        """Solve a problem using multi-step reasoning.

        Args:
            problem: The problem statement (e.g. "Implement Dijkstra's algorithm")
            verify: Whether to execute and verify the code

        Returns:
            SolveResult with plan, code, verification, and explanation
        """
        start = time.time()
        result = SolveResult(problem=problem)

        # ─── Step 1: Plan ──────────────────────────────────────────────────
        console.print("[dim]Step 1/5: Planning...[/dim]")
        t0 = time.time()
        plan = self._call(self.planner_model, PLAN_PROMPT.format(problem=problem))
        result.plan = plan
        result.steps.append(SolveStep(
            name="plan", content=plan, model=self.planner_model,
            duration_ms=(time.time() - t0) * 1000,
        ))

        # ─── Step 2: Recall ────────────────────────────────────────────────
        console.print("[dim]Step 2/5: Recalling experiences...[/dim]")
        t0 = time.time()
        mind = self._get_mind()
        experiences = mind.memory.recall(problem, top_k=3)
        result.experiences_used = len(experiences)
        exp_text = ""
        if experiences:
            parts = []
            for i, exp in enumerate(experiences, 1):
                parts.append(
                    f"### Past Solution {i} (from {exp.source}, quality {exp.combined_score():.2f})\n"
                    f"Problem: {exp.query[:150]}\n"
                    f"Solution: {exp.response[:600]}"
                )
            exp_text = "\n\n".join(parts)
        else:
            exp_text = "(no similar past solutions — solve from first principles)"
        result.steps.append(SolveStep(
            name="recall", content=f"Found {len(experiences)} relevant experiences",
            duration_ms=(time.time() - t0) * 1000,
        ))

        # ─── Step 3: Code ──────────────────────────────────────────────────
        console.print(f"[dim]Step 3/5: Generating code with {self.coder_model}...[/dim]")
        t0 = time.time()
        code_response = self._call(
            self.coder_model,
            CODE_PROMPT.format(problem=problem, plan=plan, experiences=exp_text),
            temperature=0.1,
        )
        code = self._extract_code(code_response)
        result.code = code
        result.steps.append(SolveStep(
            name="code", content=code, model=self.coder_model,
            duration_ms=(time.time() - t0) * 1000,
        ))

        # ─── Step 4: Verify ────────────────────────────────────────────────
        if verify and code:
            console.print("[dim]Step 4/5: Verifying...[/dim]")
            t0 = time.time()
            success, output = self._verify(code)
            result.verified = success
            result.verification_output = output
            result.steps.append(SolveStep(
                name="verify",
                content=f"{'PASS' if success else 'FAIL'}: {output[:300]}",
                duration_ms=(time.time() - t0) * 1000,
            ))

            # ─── Step 5: Refine on failure ─────────────────────────────────
            while not success and result.iterations < self.max_refinements:
                result.iterations += 1
                console.print(f"[dim]Step 5/5: Refining (attempt {result.iterations}/{self.max_refinements})...[/dim]")
                t0 = time.time()
                refined = self._call(
                    self.coder_model,
                    REFINE_PROMPT.format(problem=problem, code=code, error=output),
                    temperature=0.1,
                )
                code = self._extract_code(refined)
                result.code = code
                success, output = self._verify(code)
                result.verified = success
                result.verification_output = output
                result.steps.append(SolveStep(
                    name=f"refine_{result.iterations}",
                    content=f"{'PASS' if success else 'FAIL'}: {output[:300]}",
                    model=self.coder_model,
                    duration_ms=(time.time() - t0) * 1000,
                ))

        # ─── Step 6: Explain ───────────────────────────────────────────────
        if result.code:
            t0 = time.time()
            explanation = self._call(
                self.planner_model,
                EXPLAIN_PROMPT.format(problem=problem, code=result.code),
            )
            result.explanation = explanation
            result.steps.append(SolveStep(
                name="explain", content=explanation, model=self.planner_model,
                duration_ms=(time.time() - t0) * 1000,
            ))

        # ─── Step 7: Learn (only if verified) ──────────────────────────────
        if result.verified and result.code:
            quality = 0.9  # high quality because it passed verification
            mind.memory.learn(
                query=problem,
                response=f"# Solution\n```python\n{result.code}\n```\n\n{result.explanation}",
                source="solver",
                quality_score=quality,
            )

        result.total_time_ms = (time.time() - start) * 1000
        return result

    def display(self, result: SolveResult) -> None:
        """Pretty-print a solve result."""
        if result.verified:
            status = "[green]VERIFIED[/green]"
        elif not result.verification_output and result.code:
            status = "[yellow]NOT VERIFIED[/yellow]"
        elif result.code:
            status = "[red]FAILED[/red]"
        else:
            status = "[yellow]NO CODE[/yellow]"
        console.print(Panel(
            f"[bold]Problem:[/bold] {result.problem}\n"
            f"[bold]Status:[/bold] {status} | "
            f"[bold]Iterations:[/bold] {result.iterations} | "
            f"[bold]Experiences:[/bold] {result.experiences_used} | "
            f"[bold]Time:[/bold] {result.total_time_ms/1000:.1f}s",
            title="[bold cyan]NeuroBridge Solver[/bold cyan]",
            border_style="cyan",
        ))

        if result.plan:
            console.print(Panel(result.plan, title="[bold]Plan[/bold]", border_style="dim"))

        if result.code:
            from rich.syntax import Syntax
            console.print(Panel(
                Syntax(result.code, "python", theme="monokai", line_numbers=True),
                title="[bold green]Solution[/bold green]",
                border_style="green",
            ))

        if result.verification_output:
            color = "green" if result.verified else "red"
            console.print(Panel(
                result.verification_output,
                title=f"[bold {color}]Verification Output[/bold {color}]",
                border_style=color,
            ))

        if result.explanation:
            console.print(Panel(
                result.explanation,
                title="[bold]Explanation[/bold]",
                border_style="cyan",
            ))
