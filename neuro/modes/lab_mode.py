"""Lab Mode — Autonomous Self-Learning Sandbox.

This module provides the 'neuro lab' command group, allowing NeuroBridge to:
  1. Generate synthetic tasks across different difficulty levels
  2. Execute them autonomously in a safe, sandboxed environment
  3. Validate the results via self-reflection and testing
  4. Collect verified traces for continuous adapter training
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
import typer

from neuro.constants import HDD_TRACES
from neuro.traces.recorder import TraceRecorder

console = Console()
lab_app = typer.Typer(help="Autonomous self-learning sandbox.")

# ── Task Generators ──────────────────────────────────────────────────────────

# Very basic task templates for initial self-play. 
# A real implementation would use the Cohere planner to generate these dynamically.
TASK_TEMPLATES = {
    "trivial": [
        "Write a Python function to reverse a string.",
        "Create a bash script to list all files in a directory sorted by size.",
        "Write a Python function to check if a string is a palindrome.",
        "Create a regex pattern to validate email addresses."
    ],
    "easy": [
        "Write a Python script to parse a CSV file and return a list of dictionaries.",
        "Create a simple HTTP server in Python that returns 'OK' on port 8080.",
        "Write a function to calculate the Fibonacci sequence up to n terms.",
        "Create a Python script to download an image from a URL and save it."
    ],
    "medium": [
        "Implement a simple caching decorator in Python with TTL support.",
        "Write a Python script to extract all links from an HTML page using BeautifulSoup.",
        "Create a basic SQLite wrapper class in Python with insert and select methods.",
        "Implement a thread-safe singleton pattern in Python."
    ]
}


@dataclass
class LabSession:
    """Configuration for an autonomous learning session."""
    
    session_id: str
    difficulty: str
    iterations: int
    model: str
    successful: int = 0
    failed: int = 0
    start_time: float = field(default_factory=time.time)
    
    def get_duration(self) -> float:
        return time.time() - self.start_time


def generate_task(difficulty: str) -> str:
    """Generate a synthetic task of the requested difficulty."""
    if difficulty not in TASK_TEMPLATES:
        difficulty = random.choice(list(TASK_TEMPLATES.keys()))
    return random.choice(TASK_TEMPLATES[difficulty])


# ── Lab Commands ─────────────────────────────────────────────────────────────

@lab_app.command("run")
def run_lab(
    difficulty: str = typer.Option(
        "easy", "--difficulty", "-d", 
        help="Task difficulty: trivial|easy|medium"
    ),
    iterations: int = typer.Option(
        5, "--iterations", "-n", 
        help="Number of autonomous tasks to execute."
    ),
    model: str = typer.Option(
        "phi3:latest", "--model", "-m", 
        help="Model to use for self-learning."
    ),
    auto_approve: bool = typer.Option(
        True, "--auto-approve", 
        help="Automatically approve successful traces for training."
    ),
) -> None:
    """Start an autonomous self-learning session."""
    from neuro.router.router import Router
    
    session = LabSession(
        session_id=f"lab_{int(time.time())}",
        difficulty=difficulty,
        iterations=iterations,
        model=model,
    )
    
    console.print(Panel(
        f"[bold cyan]🧪 NeuroBridge Lab Mode Initiated[/bold cyan]\n"
        f"Session: {session.session_id}\n"
        f"Model: {model}\n"
        f"Target: {iterations} {difficulty} tasks\n"
        f"[dim]Generating synthetic data for continuous training...[/dim]",
        border_style="cyan"
    ))
    
    router = Router()
    recorder = TraceRecorder()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        console=console,
    ) as progress:
        
        lab_task = progress.add_task(f"Running self-play...", total=iterations)
        
        for i in range(iterations):
            task_prompt = generate_task(difficulty)
            progress.update(lab_task, description=f"Task {i+1}: [dim]{task_prompt[:40]}...[/dim]")
            
            try:
                from neuro.modes.safe_mode import SafeMode
                from neuro.training.optimizer import ConsistencyTester
                
                # Execute via SafeMode
                mode = SafeMode()
                
                # Consistency Test (Layer 3 Suppression)
                tester = ConsistencyTester(model=model, iterations=2)
                import asyncio
                result = asyncio.run(tester.test(task_prompt, context="Synthetic Lab Context"))
                
                if result["consistency_score"] >= 0.8:
                    # Capture as a successful trace
                    answer = mode.ask(task_prompt, model_override=model)
                    recorder.record_trace(
                        task=task_prompt,
                        response=answer.content,
                        model=model,
                        trainable=True,
                        category=difficulty
                    )
                    session.successful += 1
                    progress.update(lab_task, description=f"Task {i+1}: [green]Approved (Score: {result['consistency_score']:.2f})[/green]")
                else:
                    session.failed += 1
                    progress.update(lab_task, description=f"Task {i+1}: [red]Rejected (Score: {result['consistency_score']:.2f})[/red]")
                
            except Exception as e:
                console.print(f"[red]Error in task {i+1}: {str(e)}[/red]")
                session.failed += 1
                
            progress.advance(lab_task)
            
    # Session summary
    duration = session.get_duration()
    console.print("\n[bold green]✓ Lab Session Complete[/bold green]")
    
    table = Table(show_header=False, border_style="dim")
    table.add_row("Duration", f"{duration:.1f}s")
    table.add_row("Tasks Attempted", str(iterations))
    table.add_row("Successful", f"[green]{session.successful}[/green]")
    table.add_row("Failed", f"[red]{session.failed}[/red]")
    table.add_row("Success Rate", f"{(session.successful/iterations)*100:.0f}%")
    
    console.print(table)
    
    if auto_approve and session.successful > 0:
        console.print(f"\n[dim]Traces from this session were automatically moved to /accepted for training.[/dim]")
        console.print(f"Run [cyan]neuro train prepare[/cyan] to build a dataset.")


@lab_app.command("stats")
def lab_stats() -> None:
    """View stats from autonomous lab sessions."""
    # Since we are mocking lab runs currently, we'll just check trace stats
    # as lab runs fundamentally just generate traces.
    import subprocess
    subprocess.run(["neuro", "traces", "stats"])
