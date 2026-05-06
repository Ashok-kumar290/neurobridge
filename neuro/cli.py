"""NeuroBridge CLI — the main entry point.

Commands:
  neuro init          First-time setup
  neuro config        Configuration management
  neuro models        Model management
  neuro index         Index a repository
  neuro ask           Ask about an indexed repo
  neuro memory        Memory operations
  neuro code          Coding tasks (Phase 2+)
  neuro train         Training operations (Phase 6+)
  neuro adapter       Adapter management (Phase 7+)
  neuro lab           Lab mode (Phase 9+)
  neuro safety        Safety tools (Phase 5+)
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, DownloadColumn
from rich.table import Table
from rich.tree import Tree

from neuro.config import NeuroBridgeConfig, get_config, load_config, save_config
from neuro.constants import (
    HDD_ADAPTERS,
    HDD_CHECKPOINTS,
    HDD_CONFIG,
    HDD_DATASETS,
    HDD_EVALS,
    HDD_LOGS,
    HDD_MEMORY,
    HDD_REPOS,
    HDD_ROOT,
    HDD_TRACES,
    MODEL_CODER,
    MODEL_ROUTER,
    SSD_CACHE,
    SSD_CONFIG,
    SSD_INDEXES,
    SSD_VECTORS,
)

console = Console()
app = typer.Typer(
    name="neuro",
    help="NeuroBridge — Local intelligence layer for coding agents.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# ── Sub-command groups ─────────────────────────────────────────────────────────
config_app = typer.Typer(help="Configuration management.")
models_app = typer.Typer(help="Local model management.")
memory_app = typer.Typer(help="Memory operations.")
safety_app = typer.Typer(help="Safety tools.")
traces_app = typer.Typer(help="Trace management.")
evals_app = typer.Typer(help="Evaluation benchmarks.")
train_app = typer.Typer(help="Training & adapter management.")
lab_app = typer.Typer(help="Autonomous self-learning sandbox.")

app.add_typer(config_app, name="config")
app.add_typer(models_app, name="models")
app.add_typer(memory_app, name="memory")
app.add_typer(safety_app, name="safety")
app.add_typer(traces_app, name="traces")
app.add_typer(evals_app, name="eval")
app.add_typer(train_app, name="train")
app.add_typer(lab_app, name="lab")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# neuro init
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@app.command()
def init(
    storage: Path = typer.Option(
        HDD_ROOT,
        "--storage",
        "-s",
        help="Root storage path (HDD brain location).",
    ),
) -> None:
    """First-time NeuroBridge setup — creates directory structure and config."""
    console.print(
        Panel(
            "[bold cyan]NeuroBridge[/bold cyan] — First-time setup",
            border_style="cyan",
        )
    )

    # ── HDD brain directories ─────────────────────────────────────────────
    console.print("\n[bold]1.[/bold] Setting up HDD brain...")
    hdd_dirs = [
        storage / "config",
        storage / "repos",
        storage / "memory",
        storage / "models",
        storage / "traces" / "raw",
        storage / "traces" / "sanitized",
        storage / "traces" / "accepted",
        storage / "traces" / "rejected",
        storage / "datasets",
        storage / "adapters" / "qwen3b",
        storage / "adapters" / "qwen7b",
        storage / "checkpoints" / "snapshots",
        storage / "checkpoints" / "promoted",
        storage / "evals" / "reports",
        storage / "evals" / "hallucination",
        storage / "evals" / "command_safety",
        storage / "evals" / "regression",
        storage / "logs",
    ]
    for d in hdd_dirs:
        d.mkdir(parents=True, exist_ok=True)
    console.print(f"   [green]✓[/green] HDD brain at [cyan]{storage}[/cyan]")

    # ── SSD cache ──────────────────────────────────────────────────────────
    console.print("\n[bold]2.[/bold] Setting up SSD cache...")
    ssd_dirs = [SSD_INDEXES, SSD_VECTORS]
    for d in ssd_dirs:
        d.mkdir(parents=True, exist_ok=True)
    console.print(f"   [green]✓[/green] SSD cache at [cyan]{SSD_CACHE}[/cyan]")

    # ── Default config ─────────────────────────────────────────────────────
    console.print("\n[bold]3.[/bold] Writing default config...")
    config = NeuroBridgeConfig(storage_root=storage)
    hdd_config_path = storage / "config" / "neurobridge.yaml"
    if not hdd_config_path.exists():
        save_config(config, hdd_config_path)
        console.print(f"   [green]✓[/green] Config saved to [cyan]{hdd_config_path}[/cyan]")
    else:
        console.print(f"   [yellow]⊘[/yellow] Config already exists at [cyan]{hdd_config_path}[/cyan]")

    # ── Summary ────────────────────────────────────────────────────────────
    console.print("\n[bold green]✓ NeuroBridge initialized![/bold green]")
    console.print("\nNext steps:")
    console.print("  [dim]1.[/dim] neuro models pull     [dim]# download Qwen coder models[/dim]")
    console.print("  [dim]2.[/dim] neuro config doctor   [dim]# verify everything works[/dim]")
    console.print("  [dim]3.[/dim] neuro index ./repo    [dim]# index your first repo[/dim]")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# neuro config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@config_app.command("doctor")
def config_doctor() -> None:
    """Run health checks on NeuroBridge installation."""
    console.print(
        Panel("[bold cyan]NeuroBridge Doctor[/bold cyan]", border_style="cyan")
    )

    checks: list[tuple[str, bool, str]] = []

    # Check HDD
    hdd_ok = HDD_ROOT.exists() and HDD_ROOT.is_dir()
    checks.append(("HDD Brain", hdd_ok, str(HDD_ROOT)))

    # Check HDD writable
    if hdd_ok:
        try:
            test_file = HDD_ROOT / ".doctor_test"
            test_file.write_text("ok")
            test_file.unlink()
            checks.append(("HDD Writable", True, "read/write OK"))
        except Exception as e:
            checks.append(("HDD Writable", False, str(e)))
    else:
        checks.append(("HDD Writable", False, "HDD not found"))

    # Check SSD cache
    ssd_ok = SSD_CACHE.exists()
    checks.append(("SSD Cache", ssd_ok, str(SSD_CACHE)))

    # Check config
    config_path = HDD_ROOT / "config" / "neurobridge.yaml"
    config_ok = config_path.exists()
    checks.append(("Config File", config_ok, str(config_path)))

    # Check Ollama
    from neuro.runtime.ollama_client import get_ollama_client

    client = get_ollama_client()
    ollama_ok = client.is_running()
    checks.append(("Ollama Server", ollama_ok, "http://localhost:11434"))

    # Check models
    if ollama_ok:
        has_router = client.has_model(MODEL_ROUTER)
        has_coder = client.has_model(MODEL_CODER)
        checks.append(("Router Model", has_router, MODEL_ROUTER))
        checks.append(("Coder Model", has_coder, MODEL_CODER))
    else:
        checks.append(("Router Model", False, "Ollama not running"))
        checks.append(("Coder Model", False, "Ollama not running"))

    # Check key subdirs
    for name, path in [
        ("Memory Dir", HDD_MEMORY),
        ("Traces Dir", HDD_TRACES),
        ("Adapters Dir", HDD_ADAPTERS),
        ("Datasets Dir", HDD_DATASETS),
        ("Evals Dir", HDD_EVALS),
    ]:
        checks.append((name, path.exists(), str(path)))

    # Display results
    table = Table(show_header=True, header_style="bold")
    table.add_column("Check", style="bold")
    table.add_column("Status")
    table.add_column("Details", style="dim")

    all_ok = True
    for name, ok, detail in checks:
        status = "[green]✓ OK[/green]" if ok else "[red]✗ FAIL[/red]"
        if not ok:
            all_ok = False
        table.add_row(name, status, detail)

    console.print(table)

    if all_ok:
        console.print("\n[bold green]All checks passed![/bold green] 🧠")
    else:
        console.print(
            "\n[bold yellow]Some checks failed.[/bold yellow] "
            "Run [cyan]neuro init[/cyan] to fix missing directories."
        )


@config_app.command("show")
def config_show() -> None:
    """Display current NeuroBridge configuration."""
    config = get_config()
    import yaml

    console.print(
        Panel("[bold cyan]Active Configuration[/bold cyan]", border_style="cyan")
    )
    console.print(
        yaml.dump(config.model_dump(mode="json"), default_flow_style=False, sort_keys=False)
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# neuro models
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@models_app.command("list")
def models_list() -> None:
    """List locally available Ollama models."""
    from neuro.runtime.ollama_client import get_ollama_client

    client = get_ollama_client()
    if not client.is_running():
        console.print("[red]✗ Ollama is not running.[/red] Start with: sudo systemctl start ollama")
        raise typer.Exit(1)

    models = client.list_models()
    if not models:
        console.print("[yellow]No models found.[/yellow] Run: neuro models pull")
        return

    table = Table(title="Local Models", show_header=True, header_style="bold cyan")
    table.add_column("Name")
    table.add_column("Size")
    table.add_column("Modified")
    table.add_column("Role", style="dim")

    for m in models:
        name = m.get("name", "unknown")
        size_bytes = m.get("size", 0)
        size_gb = f"{size_bytes / 1e9:.1f} GB"
        modified = m.get("modified_at", "unknown")[:10]

        role = ""
        if "super-qwen:3b" in name.lower() or "3b" in name.lower():
            role = "router/compressor"
        elif "super-qwen:7b" in name.lower() or "7b" in name.lower():
            role = "coder/debugger"

        table.add_row(name, size_gb, modified, role)

    console.print(table)


@models_app.command("pull")
def models_pull(
    model: Optional[str] = typer.Argument(
        None,
        help="Specific model to pull. If omitted, pulls both router (3B) and coder (7B).",
    ),
) -> None:
    """Pull Qwen coder models via Ollama."""
    from neuro.runtime.ollama_client import get_ollama_client

    client = get_ollama_client()
    if not client.is_running():
        console.print("[red]✗ Ollama is not running.[/red]")
        raise typer.Exit(1)

    targets = [model] if model else [MODEL_ROUTER, MODEL_CODER]

    for target in targets:
        console.print(f"\n[bold]Pulling [cyan]{target}[/cyan]...[/bold]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(target, total=None)

            for update in client.pull_model(target):
                status = update.get("status", "")
                completed = update.get("completed", 0)
                total = update.get("total", 0)

                if total > 0:
                    progress.update(task, total=total, completed=completed, description=status)
                else:
                    progress.update(task, description=status)

        console.print(f"[green]✓[/green] {target} ready.")


@models_app.command("status")
def models_status() -> None:
    """Show status of required NeuroBridge models."""
    from neuro.runtime.ollama_client import get_ollama_client

    client = get_ollama_client()
    if not client.is_running():
        console.print("[red]✗ Ollama is not running.[/red]")
        raise typer.Exit(1)

    required = {
        MODEL_ROUTER: "Fast router / compressor (3B)",
        MODEL_CODER: "Local coding worker (7B)",
    }

    table = Table(title="Required Models", show_header=True, header_style="bold cyan")
    table.add_column("Model")
    table.add_column("Status")
    table.add_column("Role", style="dim")

    for model_name, role in required.items():
        available = client.has_model(model_name)
        status = "[green]✓ Available[/green]" if available else "[red]✗ Missing[/red]"
        table.add_row(model_name, status, role)

    console.print(table)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# neuro index
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@app.command()
def index(
    repo_path: Path = typer.Argument(
        ...,
        help="Path to repository to index.",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
) -> None:
    """Index a repository for NeuroBridge memory and search."""
    from neuro.repo.indexer import RepoIndexer

    repo_path = repo_path.resolve()
    console.print(
        Panel(
            f"[bold cyan]Indexing[/bold cyan] [white]{repo_path.name}[/white]",
            border_style="cyan",
        )
    )

    indexer = RepoIndexer(repo_path)
    stats = indexer.run()

    table = Table(title="Index Results", show_header=True, header_style="bold cyan")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    table.add_row("Files scanned", str(stats.get("files_scanned", 0)))
    table.add_row("Files indexed", str(stats.get("files_indexed", 0)))
    table.add_row("Chunks created", str(stats.get("chunks_created", 0)))
    table.add_row("Symbols extracted", str(stats.get("symbols_extracted", 0)))
    table.add_row("Index size", stats.get("index_size", "0 KB"))
    table.add_row("Time", f"{stats.get('duration_seconds', 0):.1f}s")

    console.print(table)
    console.print(f"\n[green]✓[/green] Repo indexed. Use [cyan]neuro ask[/cyan] to query it.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# neuro ask
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question about the indexed repo."),
    repo: Optional[Path] = typer.Option(
        None, "--repo", "-r", help="Path to repo (auto-detected from cwd if omitted)."
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Force a specific model (e.g. 'qwen2.5-coder:7b')."
    ),
) -> None:
    """Ask a question about an indexed repository."""
    from neuro.modes.safe_mode import SafeMode

    repo_path = (repo or Path.cwd()).resolve()

    mode = SafeMode(repo_path=repo_path)
    answer = mode.ask(question, model_override=model)

    console.print(Panel(answer.content, title="[bold cyan]Answer[/bold cyan]", border_style="cyan"))

    if answer.sources:
        console.print("\n[dim]Sources:[/dim]")
        for src in answer.sources:
            console.print(f"  [dim]•[/dim] {src}")

    if answer.tokens_used:
        console.print(f"\n[dim]Tokens: {answer.tokens_used} | Model: {answer.model}[/dim]")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# neuro memory
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@memory_app.command("search")
def memory_search(
    query: str = typer.Argument(..., help="Search query for memory."),
    limit: int = typer.Option(10, "--limit", "-n", help="Max results."),
) -> None:
    """Search NeuroBridge memory."""
    from neuro.memory.session_memory import SessionMemory

    mem = SessionMemory()
    results = mem.search(query, limit=limit)

    if not results:
        console.print("[yellow]No memory matches found.[/yellow]")
        return

    for i, r in enumerate(results, 1):
        console.print(f"\n[bold cyan]{i}.[/bold cyan] {r.get('summary', 'N/A')}")
        console.print(f"   [dim]Repo: {r.get('repo', '?')} | {r.get('timestamp', '?')}[/dim]")


@memory_app.command("stats")
def memory_stats() -> None:
    """Show memory system statistics."""
    from neuro.memory.sqlite_store import get_memory_stats

    stats = get_memory_stats()

    table = Table(title="Memory Statistics", show_header=True, header_style="bold cyan")
    table.add_column("Store")
    table.add_column("Records", justify="right")
    table.add_column("Size", justify="right")

    for store_name, info in stats.items():
        table.add_row(store_name, str(info["records"]), info["size"])

    console.print(table)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# neuro code — the main coding task command
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@app.command()
def code(
    task: str = typer.Argument(..., help="Coding task description."),
    repo: Optional[Path] = typer.Option(
        None, "--repo", "-r", help="Path to repo (auto-detected from cwd if omitted).",
    ),
    expert: Optional[str] = typer.Option(
        None, "--expert", "-e", help="Force expert: codex|claude|cohere",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show routing decision without executing.",
    ),
) -> None:
    """Submit a coding task to NeuroBridge."""
    from neuro.compression.expert_packet import ExpertPacketBuilder
    from neuro.modes.safe_mode import SafeMode
    from neuro.repo.search import RepoSearch
    from neuro.router.router import Router

    repo_path = (repo or Path.cwd()).resolve()
    repo_name = repo_path.name

    # ── Search repo for context ────────────────────────────────────────────
    search_results = []
    search_score = 0.0
    file_count = 0
    try:
        searcher = RepoSearch(repo_name)
        search_results = searcher.search(task, limit=6)
        file_count = len(set(r.file_path for r in search_results))
        search_score = max((r.score for r in search_results), default=0.0)
    except FileNotFoundError:
        console.print("[yellow]⊘ Repo not indexed. Run neuro index first for better routing.[/yellow]")

    # ── Search memory ──────────────────────────────────────────────────────
    from neuro.memory.session_memory import SessionMemory

    memory = SessionMemory()
    memory_hits = memory.search(task, limit=5)

    # ── Build Context String ───────────────────────────────────────────────
    context_parts = []
    for r in search_results:
        context_parts.append(f"File: {r.file_path}\n```{r.language}\n{r.content}\n```")
    context_text = "\n\n".join(context_parts)

    # ── Route the task ─────────────────────────────────────────────────────
    router = Router()
    decision = router.route(
        query=task,
        context=context_text,
        file_count=file_count,
        memory_hits=len(memory_hits),
        search_score=search_score,
        force_expert=expert,
    )

    # ── Display routing decision ───────────────────────────────────────────
    _display_routing(decision)

    if dry_run:
        console.print("\n[dim]--dry-run: stopping before execution.[/dim]")
        return

    # ── Execute ────────────────────────────────────────────────────────────
    if decision.expert_required:
        _execute_expert(task, repo_path, decision, search_results, memory_hits)
    else:
        _execute_local(task, repo_path, decision, search_results)


def _display_routing(decision) -> None:
    """Display a routing decision table."""
    table = Table(title="Routing Decision", show_header=True, header_style="bold cyan")
    table.add_column("Property", style="bold")
    table.add_column("Value")

    # Color the target based on type
    if "local" in decision.target:
        target_str = f"[green]{decision.target}[/green]"
    elif "expert" in decision.target:
        target_str = f"[yellow]{decision.target}[/yellow]"
    else:
        target_str = decision.target

    table.add_row("Target", target_str)
    table.add_row("Model", decision.model)
    table.add_row("Reason", decision.reason)
    table.add_row(
        "Difficulty",
        f"{decision.difficulty.difficulty.value} (score={decision.difficulty.score})",
    )
    table.add_row(
        "Confidence",
        f"{'🟢' if not decision.confidence.should_escalate else '🔴'} {decision.confidence.score}",
    )
    table.add_row(
        "Est. Cost",
        f"${decision.budget.estimated_cost_usd:.6f}" if decision.budget.estimated_cost_usd > 0 else "[green]FREE[/green]",
    )
    table.add_row("Est. Tokens", f"{decision.budget.total_tokens:,}")

    console.print(table)

    # Show reasoning
    if decision.difficulty.reasons:
        console.print("\n[dim]Difficulty signals:[/dim]")
        for r in decision.difficulty.reasons:
            console.print(f"  [dim]•[/dim] {r}")

    if decision.confidence.reasons:
        console.print("\n[dim]Confidence factors:[/dim]")
        for r in decision.confidence.reasons:
            console.print(f"  [dim]•[/dim] {r}")


def _execute_local(task: str, repo_path: Path, decision, search_results=None) -> None:
    """Execute a task with a local model."""
    from neuro.modes.safe_mode import SafeMode

    console.print(f"\n[bold]Executing with [cyan]{decision.model}[/cyan]...[/bold]")

    mode = SafeMode(repo_path=repo_path)
    answer = mode.ask(task, model_override=decision.model, context_override=search_results)

    console.print(Panel(answer.content, title="[bold cyan]Result[/bold cyan]", border_style="cyan"))

    if answer.sources:
        console.print("\n[dim]Sources:[/dim]")
        for src in answer.sources:
            console.print(f"  [dim]•[/dim] {src}")

    console.print(f"\n[dim]Tokens: {answer.tokens_used} | Model: {answer.model} | Time: {answer.duration_ms:.0f}ms[/dim]")


def _execute_expert(task, repo_path, decision, search_results, memory_hits) -> None:
    """Execute a task with an expert model."""
    from neuro.compression.expert_packet import ExpertPacketBuilder

    # Build expert packet
    builder = ExpertPacketBuilder()
    packet = builder.build(
        task=task,
        repo_name=repo_path.name,
        search_results=search_results,
        memory_hits=memory_hits,
    )

    console.print(f"\n[bold]Expert packet: [cyan]{packet.estimated_tokens:,}[/cyan] tokens[/bold]")

    expert = decision.preferred_expert or "claude"

    if expert == "claude":
        from neuro.runtime.claude_bridge import ClaudeBridge

        bridge = ClaudeBridge()
        if not bridge.is_available():
            console.print("[red]✗ Claude Code not installed.[/red] Run: npm install -g @anthropic-ai/claude-code")
            console.print("\n[dim]Expert packet saved. You can manually paste it into Claude.[/dim]")
            console.print(Panel(packet.to_prompt()[:2000] + "...", title="Expert Packet Preview"))
            return

        console.print("[bold]Sending to Claude Code...[/bold]")
        response = bridge.code(task=task, context=packet.to_prompt(), cwd=repo_path)
        console.print(Panel(response.content, title="[bold yellow]Claude Response[/bold yellow]", border_style="yellow"))

    elif expert == "codex":
        from neuro.runtime.codex_bridge import CodexBridge

        bridge = CodexBridge()
        if not bridge.is_available():
            console.print("[red]✗ Codex CLI not installed.[/red] Run: npm install -g @openai/codex")
            console.print(Panel(packet.to_prompt()[:2000] + "...", title="Expert Packet Preview"))
            return

        console.print("[bold]Sending to Codex CLI...[/bold]")
        response = bridge.code(task=task, context=packet.to_prompt(), cwd=repo_path)
        console.print(Panel(response.content, title="[bold yellow]Codex Response[/bold yellow]", border_style="yellow"))

    elif expert == "cohere":
        from neuro.runtime.cohere_client import get_cohere_client

        client = get_cohere_client()
        if not client.available:
            console.print("[red]✗ Cohere API key not set.[/red] Set COHERE_API_KEY env var.")
            console.print(Panel(packet.to_prompt()[:2000] + "...", title="Expert Packet Preview"))
            return

        console.print("[bold]Sending to Cohere Command A...[/bold]")
        response = client.plan(task=task, context=packet.to_prompt())
        console.print(Panel(response.content, title="[bold yellow]Cohere Response[/bold yellow]", border_style="yellow"))

    else:
        console.print(f"[red]Unknown expert: {expert}[/red]")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# neuro route — preview routing decisions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@app.command()
def route(
    task: str = typer.Argument(..., help="Task to analyze routing for."),
    repo: Optional[Path] = typer.Option(
        None, "--repo", "-r", help="Path to repo.",
    ),
) -> None:
    """Preview how NeuroBridge would route a task (no execution)."""
    from neuro.repo.search import RepoSearch
    from neuro.router.router import Router

    repo_path = (repo or Path.cwd()).resolve()
    repo_name = repo_path.name

    # Search for context
    search_score = 0.0
    file_count = 0
    try:
        searcher = RepoSearch(repo_name)
        results = searcher.search(task, limit=6)
        file_count = len(set(r.file_path for r in results))
        search_score = max((r.score for r in results), default=0.0)
    except FileNotFoundError:
        pass

    # Search memory
    from neuro.memory.session_memory import SessionMemory

    memory = SessionMemory()
    memory_hits = memory.search(task, limit=5)

    # Route
    router = Router()
    decision = router.route(
        query=task,
        file_count=file_count,
        memory_hits=len(memory_hits),
        search_score=search_score,
    )

    _display_routing(decision)

    # Token budget comparison
    from neuro.router.token_budget import compare_budgets

    console.print("\n")
    budgets_table = Table(title="Cost Comparison", show_header=True, header_style="bold cyan")
    budgets_table.add_column("Model")
    budgets_table.add_column("Input", justify="right")
    budgets_table.add_column("Output", justify="right")
    budgets_table.add_column("Cost", justify="right")

    budgets = compare_budgets("", task)
    for b in budgets:
        cost_str = "[green]FREE[/green]" if b.estimated_cost_usd == 0 else f"${b.estimated_cost_usd:.6f}"
        budgets_table.add_row(b.model, f"{b.input_tokens:,}", f"{b.output_tokens:,}", cost_str)

    console.print(budgets_table)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# neuro packet — preview expert packets
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@app.command()
def packet(
    task: str = typer.Argument(..., help="Task to build expert packet for."),
    repo: Optional[Path] = typer.Option(
        None, "--repo", "-r", help="Path to repo.",
    ),
    output_format: str = typer.Option(
        "prompt", "--format", "-f", help="Output format: prompt|json",
    ),
) -> None:
    """Preview the expert packet that would be sent to Claude/Codex/Cohere."""
    from neuro.compression.expert_packet import ExpertPacketBuilder
    from neuro.memory.session_memory import SessionMemory
    from neuro.repo.search import RepoSearch

    repo_path = (repo or Path.cwd()).resolve()
    repo_name = repo_path.name

    # Search
    search_results = []
    try:
        searcher = RepoSearch(repo_name)
        search_results = searcher.search(task, limit=6)
    except FileNotFoundError:
        console.print("[yellow]⊘ Repo not indexed.[/yellow]")

    # Memory
    memory = SessionMemory()
    memory_hits = memory.search(task, limit=3)

    # Build packet
    builder = ExpertPacketBuilder()
    pkt = builder.build(
        task=task,
        repo_name=repo_name,
        search_results=search_results,
        memory_hits=memory_hits,
    )

    if output_format == "json":
        console.print(pkt.to_json())
    else:
        console.print(Panel(
            pkt.to_prompt(),
            title=f"[bold cyan]Expert Packet[/bold cyan] ({pkt.estimated_tokens:,} tokens)",
            border_style="cyan",
        ))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# neuro safety — safety tools
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@safety_app.command("scan")
def safety_scan(
    command: str = typer.Argument(..., help="Command to scan for safety."),
) -> None:
    """Scan a command for safety violations."""
    from neuro.safety.command_scanner import scan_command

    result = scan_command(command)

    if result.safe and not result.violations:
        console.print(f"[green]✓ SAFE[/green]: {command}")
    elif result.safe:
        console.print(f"[yellow]⚠ WARNINGS[/yellow]: {command}")
        for v in result.violations:
            console.print(f"  {v}")
    else:
        console.print(f"[red]✗ BLOCKED[/red]: {command}")
        for v in result.violations:
            console.print(f"  {v}")


@safety_app.command("check-secrets")
def safety_check_secrets(
    text: str = typer.Argument(..., help="Text to scan for secrets."),
) -> None:
    """Scan text for secrets and sensitive data."""
    from neuro.safety.secret_detector import scan_text

    result = scan_text(text)

    if result.clean:
        console.print("[green]✓ No secrets detected.[/green]")
    else:
        console.print(f"[red]✗ {len(result.secrets_found)} secret(s) detected:[/red]")
        for s in result.secrets_found:
            console.print(
                f"  [red]•[/red] {s.pattern_name} ({s.secret_type}) "
                f"line {s.line_number} — confidence {s.confidence:.0%}"
            )


@safety_app.command("audit")
def safety_audit(
    limit: int = typer.Option(20, "--limit", "-n", help="Number of events."),
) -> None:
    """View recent audit log events."""
    from neuro.safety.audit_logger import get_audit_logger

    logger = get_audit_logger()
    events = logger.get_recent(limit=limit)

    if not events:
        console.print("[dim]No audit events yet.[/dim]")
        return

    table = Table(title=f"Audit Log (last {limit})", show_header=True, header_style="bold cyan")
    table.add_column("Time", style="dim")
    table.add_column("Event")
    table.add_column("Details", style="dim")

    for event in events[-limit:]:
        event_type = event.get("event", "?")
        iso_time = event.get("iso_time", "?")[-8:]  # just HH:MM:SS

        # Color by type
        if "blocked" in event_type or "rejected" in event_type:
            event_str = f"[red]{event_type}[/red]"
        elif "approved" in event_type or "accepted" in event_type:
            event_str = f"[green]{event_type}[/green]"
        elif "secret" in event_type:
            event_str = f"[red]{event_type}[/red]"
        else:
            event_str = event_type

        # Build details
        details_parts = []
        for key in ["command", "trace_id", "expert", "severity", "reason"]:
            if key in event:
                details_parts.append(f"{key}={event[key]}")
        details = "; ".join(details_parts)[:60]

        table.add_row(iso_time, event_str, details)

    console.print(table)
    console.print(f"\n[dim]Log size: {logger.log_size}[/dim]")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# neuro traces — trace management
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@traces_app.command("stats")
def traces_stats() -> None:
    """Show trace collection statistics."""
    from neuro.traces.recorder import TraceRecorder

    recorder = TraceRecorder()
    stats = recorder.get_stats()

    table = Table(title="Trace Statistics", show_header=True, header_style="bold cyan")
    table.add_column("Directory")
    table.add_column("Count", justify="right")
    table.add_column("Status", style="dim")

    status_map = {
        "raw": "unprocessed captures",
        "sanitized": "secrets redacted",
        "accepted": "approved for training ✓",
        "rejected": "failed verification ✗",
    }

    for directory, count in stats.items():
        table.add_row(directory, str(count), status_map.get(directory, ""))

    console.print(table)


@traces_app.command("list")
def traces_list(
    directory: str = typer.Argument(
        "raw", help="Directory to list: raw|sanitized|accepted|rejected",
    ),
    limit: int = typer.Option(20, "--limit", "-n", help="Max traces."),
) -> None:
    """List traces in a directory."""
    from neuro.traces.recorder import TraceRecorder

    recorder = TraceRecorder()
    traces = recorder.list_traces(directory)

    if not traces:
        console.print(f"[dim]No traces in {directory}/[/dim]")
        return

    table = Table(
        title=f"Traces ({directory})",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("ID", style="dim")
    table.add_column("Task")
    table.add_column("Model")
    table.add_column("Trainable")

    for t in traces[-limit:]:
        trainable = "[green]✓[/green]" if t.get("trainable") else "[dim]✗[/dim]"
        table.add_row(
            t.get("trace_id", "?"),
            t.get("task", "?")[:50],
            t.get("model", "?"),
            trainable,
        )

    console.print(table)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# neuro eval — evaluation benchmarks
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@evals_app.command("suites")
def eval_suites() -> None:
    """List available eval suites."""
    from neuro.evals.suites import list_suites

    table = Table(title="Available Eval Suites", show_header=True, header_style="bold cyan")
    table.add_column("Suite")
    table.add_column("Cases", justify="right")

    for suite in list_suites():
        table.add_row(suite["name"], str(suite["cases"]))

    total = sum(s["cases"] for s in list_suites())
    table.add_row("[bold]all[/bold]", f"[bold]{total}[/bold]")

    console.print(table)


@evals_app.command("run")
def eval_run(
    suite: str = typer.Argument("all", help="Suite: coding_basics|recall|hallucination|safety|all"),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Model to evaluate (default: 3B router).",
    ),
    save: bool = typer.Option(True, "--save/--no-save", help="Save report to HDD."),
) -> None:
    """Run an eval suite against a local model."""
    from neuro.evals.runner import EvalRunner
    from neuro.evals.suites import get_suite

    try:
        cases = get_suite(suite)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    runner = EvalRunner()

    console.print(Panel(
        f"[bold cyan]Running eval:[/bold cyan] {suite} ({len(cases)} cases)",
        border_style="cyan",
    ))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Evaluating...", total=len(cases))

        def on_result(result):
            status = "✓" if result.passed else "✗"
            progress.update(
                task,
                advance=1,
                description=f"{status} {result.case_id} ({result.duration_ms:.0f}ms)",
            )

        report = runner.run_suite(suite, cases, model=model, on_result=on_result)

    # ── Display results ────────────────────────────────────────────────────
    console.print()

    results_table = Table(title="Results", show_header=True, header_style="bold cyan")
    results_table.add_column("Case")
    results_table.add_column("Category", style="dim")
    results_table.add_column("Status")
    results_table.add_column("Time", justify="right", style="dim")

    for r in report.results:
        status = "[green]✓ PASS[/green]" if r.passed else "[red]✗ FAIL[/red]"
        results_table.add_row(r.case_id, r.category, status, f"{r.duration_ms:.0f}ms")

    console.print(results_table)

    # ── Summary ────────────────────────────────────────────────────────────
    console.print()
    summary_table = Table(title="Summary", show_header=True, header_style="bold cyan")
    summary_table.add_column("Category")
    summary_table.add_column("Passed", justify="right")
    summary_table.add_column("Total", justify="right")
    summary_table.add_column("Rate", justify="right")

    for cat, stats in report.by_category().items():
        rate = stats["passed"] / stats["total"] if stats["total"] > 0 else 0
        rate_str = f"[green]{rate:.0%}[/green]" if rate >= 0.7 else f"[red]{rate:.0%}[/red]"
        summary_table.add_row(cat, str(stats["passed"]), str(stats["total"]), rate_str)

    summary_table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{report.passed}[/bold]",
        f"[bold]{report.total}[/bold]",
        f"[bold]{report.pass_rate:.0%}[/bold]",
    )

    console.print(summary_table)
    console.print(f"\n[dim]Model: {report.model} | Avg: {report.avg_duration_ms:.0f}ms/case[/dim]")

    # ── Save ───────────────────────────────────────────────────────────────
    if save:
        path = runner.save_report(report)
        console.print(f"[dim]Report saved: {path}[/dim]")


@evals_app.command("list")
def eval_list(
    suite: Optional[str] = typer.Argument(None, help="Filter by suite name."),
) -> None:
    """List saved eval reports."""
    from neuro.evals.runner import EvalRunner

    runner = EvalRunner()
    reports = runner.load_reports(suite_name=suite)

    if not reports:
        console.print("[dim]No eval reports found.[/dim]")
        return

    table = Table(title="Eval Reports", show_header=True, header_style="bold cyan")
    table.add_column("Suite")
    table.add_column("Model")
    table.add_column("Passed", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Rate", justify="right")

    for r in reports:
        rate = r.get("pass_rate", 0)
        rate_str = f"[green]{rate:.0%}[/green]" if rate >= 0.7 else f"[red]{rate:.0%}[/red]"
        table.add_row(
            r.get("suite_name", "?"),
            r.get("model", "?"),
            str(r.get("passed", 0)),
            str(r.get("total", 0)),
            rate_str,
        )

    console.print(table)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# neuro train — training & adapter management
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@train_app.command("prepare")
def train_prepare(
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Dataset name."),
    min_examples: int = typer.Option(10, "--min", help="Minimum examples required."),
) -> None:
    """Build a training dataset from accepted traces."""
    from neuro.training.dataset_builder import DatasetBuilder

    builder = DatasetBuilder()
    path, stats = builder.build_dataset(name=name, min_examples=min_examples)

    table = Table(title="Dataset Build", show_header=True, header_style="bold cyan")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    table.add_row("Total examples", str(stats.total_examples))
    table.add_row("Est. tokens", f"{stats.total_tokens_est:,.0f}")
    table.add_row("Avg input length", f"{stats.avg_input_length:.0f} chars")
    table.add_row("Avg output length", f"{stats.avg_output_length:.0f} chars")

    for cat, count in stats.by_category.items():
        table.add_row(f"  {cat}", str(count))

    console.print(table)

    if path:
        console.print(f"\n[green]✓ Dataset saved:[/green] {path}")
    else:
        console.print(f"\n[yellow]⚠ Not enough examples ({stats.total_examples}/{min_examples})[/yellow]")
        console.print("[dim]Collect more traces with 'neuro code' and approve them.[/dim]")


@train_app.command("notebook")
def train_notebook(
    model: str = typer.Option(
        "super-qwen:7b", "--model", "-m",
        help="Which model to train: super-qwen:3b or super-qwen:7b.",
    ),
    rank: int = typer.Option(16, "--rank", "-r", help="LoRA rank."),
    epochs: int = typer.Option(3, "--epochs", "-e", help="Training epochs."),
    adapter_name: str = typer.Option(
        "super-qwen-adapter-v1", "--name", "-n", help="Adapter name.",
    ),
) -> None:
    """Generate a Colab training notebook."""
    from neuro.training.colab_generator import generate_colab_notebook, save_notebook

    # Map our model names to HuggingFace names
    hf_models = {
        "super-qwen:3b": "Qwen/Qwen2.5-Coder-3B-Instruct",
        "super-qwen:7b": "Qwen/Qwen2.5-Coder-7B-Instruct",
    }
    base_model = hf_models.get(model, model)

    # Find latest dataset
    from neuro.training.dataset_builder import DatasetBuilder
    builder = DatasetBuilder()
    datasets = builder.get_dataset_info()
    dataset_file = datasets[-1]["name"] + ".jsonl" if datasets else "neurobridge_dataset.jsonl"

    notebook = generate_colab_notebook(
        base_model=base_model,
        dataset_path=dataset_file,
        lora_rank=rank,
        lora_alpha=rank * 2,
        epochs=epochs,
        adapter_name=adapter_name,
    )

    path = save_notebook(notebook, name=f"train_{model.replace('-', '_')}")
    console.print(f"[green]✓ Notebook generated:[/green] {path}")
    console.print(f"\n[bold]Next steps:[/bold]")
    console.print(f"  1. Upload [cyan]{path}[/cyan] to Google Colab")
    console.print(f"  2. Upload the dataset JSONL alongside it")
    console.print(f"  3. Set Runtime → T4 GPU")
    console.print(f"  4. Run All cells")
    console.print(f"  5. Download the adapter zip")
    console.print(f"  6. Register: [cyan]neuro train register {adapter_name} /path/to/adapter.zip[/cyan]")


@train_app.command("datasets")
def train_datasets() -> None:
    """List available training datasets."""
    from neuro.training.dataset_builder import DatasetBuilder

    builder = DatasetBuilder()
    datasets = builder.get_dataset_info()

    if not datasets:
        console.print("[dim]No datasets yet. Run 'neuro train prepare' first.[/dim]")
        return

    table = Table(title="Training Datasets", show_header=True, header_style="bold cyan")
    table.add_column("Name")
    table.add_column("Examples", justify="right")
    table.add_column("Size", justify="right")

    for d in datasets:
        table.add_row(d["name"], str(d["examples"]), d["size"])

    console.print(table)


@train_app.command("register")
def train_register(
    name: str = typer.Argument(..., help="Adapter name."),
    adapter_path: Path = typer.Argument(..., help="Path to adapter files/zip."),
    model: str = typer.Option("super-qwen:7b", "--model", "-m", help="Base model."),
    rank: int = typer.Option(16, "--rank", help="LoRA rank used."),
    epochs: int = typer.Option(3, "--epochs", help="Epochs trained."),
) -> None:
    """Register a trained adapter from Colab."""
    from neuro.training.adapter_manager import AdapterManager

    # Handle zip files
    actual_path = adapter_path
    if adapter_path.suffix == ".zip":
        import zipfile
        import tempfile
        extract_dir = adapter_path.parent / adapter_path.stem
        with zipfile.ZipFile(adapter_path, "r") as z:
            z.extractall(extract_dir)
        actual_path = extract_dir
        console.print(f"[dim]Extracted to {extract_dir}[/dim]")

    manager = AdapterManager()
    info = manager.register(
        name=name,
        adapter_path=actual_path,
        base_model=model,
        lora_rank=rank,
        epochs=epochs,
    )

    console.print(f"[green]✓ Adapter registered:[/green] {info.name}")
    console.print(f"  Base: {info.base_model} | Rank: {info.lora_rank} | Epochs: {info.epochs}")
    console.print(f"\nNext: Run evals to test it:")
    console.print(f"  neuro eval run all --model {model}")


@train_app.command("adapters")
def train_adapters() -> None:
    """List registered adapters."""
    from neuro.training.adapter_manager import AdapterManager

    manager = AdapterManager()
    adapters = manager.list_adapters()
    active = manager.get_active()

    if not adapters:
        console.print("[dim]No adapters registered yet.[/dim]")
        return

    table = Table(title="Registered Adapters", show_header=True, header_style="bold cyan")
    table.add_column("Name")
    table.add_column("Base Model")
    table.add_column("Rank", justify="right")
    table.add_column("Status")
    table.add_column("Active")

    for a in adapters:
        name = a.get("name", "?")
        is_active = "🟢" if name == active else ""
        status = a.get("status", "?")
        if status == "promoted":
            status_str = "[green]promoted[/green]"
        elif status == "retired":
            status_str = "[dim]retired[/dim]"
        else:
            status_str = status

        table.add_row(
            name,
            a.get("base_model", "?"),
            str(a.get("lora_rank", "?")),
            status_str,
            is_active,
        )

    console.print(table)


@train_app.command("promote")
def train_promote(
    name: str = typer.Argument(..., help="Adapter name to promote."),
) -> None:
    """Promote an adapter to active status (requires passing evals)."""
    from neuro.training.adapter_manager import AdapterManager
    from neuro.evals.runner import EvalRunner

    manager = AdapterManager()

    # Check adapter exists
    adapters = manager.list_adapters()
    adapter = next((a for a in adapters if a.get("name") == name), None)
    if not adapter:
        console.print(f"[red]Adapter '{name}' not found.[/red]")
        raise typer.Exit(1)

    # Check for eval reports
    runner = EvalRunner()
    reports = runner.load_reports()

    if not reports:
        console.print("[yellow]⚠ No eval reports found. Run evals first:[/yellow]")
        console.print(f"  neuro eval run all")
        raise typer.Exit(1)

    latest = reports[-1]
    pass_rate = latest.get("pass_rate", 0)

    if pass_rate < 0.6:
        console.print(f"[red]✗ Cannot promote: eval pass rate {pass_rate:.0%} < 60% threshold[/red]")
        raise typer.Exit(1)

    # Promote
    eval_scores = {
        "pass_rate": pass_rate,
        "suite": latest.get("suite_name", "?"),
        "model": latest.get("model", "?"),
    }

    if manager.promote(name, eval_scores):
        console.print(f"[green]✓ Adapter '{name}' promoted in registry![/green]")
    else:
        console.print(f"[red]✗ Promotion failed.[/red]")
        raise typer.Exit(1)

    # Checkpoint and rebuild
    from neuro.training.checkpoint_manager import CheckpointManager
    from neuro.constants import HDD_MODELS
    import subprocess

    cp_manager = CheckpointManager()
    model_name = adapter.get("base_model", "super-qwen:7b")

    # Save current state
    console.print(f"[dim]Creating checkpoint for {model_name}...[/dim]")
    modelfile_path = HDD_MODELS / f"Modelfile.{model_name.replace(':', '-')}"
    
    cp = cp_manager.create_checkpoint(
        model_name=model_name,
        adapter_name=name,
        eval_scores=eval_scores,
        modelfile_path=modelfile_path if modelfile_path.exists() else None,
    )
    console.print(f"[dim]Saved checkpoint: {cp.version}[/dim]")

    # Generate new Modelfile
    adapter_path = manager.get_adapter_path(name)
    if not adapter_path:
        console.print(f"[red]✗ Could not find adapter files for {name}[/red]")
        raise typer.Exit(1)

    base_hf_model = "qwen2.5-coder:7b" if "7b" in model_name else "qwen2.5-coder:3b"
    new_modelfile = cp_manager.generate_promotion_modelfile(
        base_model=base_hf_model,
        adapter_path=adapter_path,
        model_name=model_name,
    )

    # Save and build
    modelfile_path.write_text(new_modelfile)
    console.print(f"\n[cyan]Rebuilding {model_name} in Ollama with adapter...[/cyan]")
    
    result = subprocess.run(
        ["ollama", "create", model_name, "-f", str(modelfile_path)],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        console.print(f"[bold green]✓ Successfully built and loaded new {model_name}![/bold green]")
        console.print(f"  Adapter: {name}")
        console.print(f"  Eval rate: {pass_rate:.0%}")
        from neuro.safety.audit_logger import get_audit_logger
        get_audit_logger().log_generic("model_rebuilt_with_adapter", model=model_name, adapter=name)
    else:
        console.print(f"[red]✗ Failed to build model in Ollama:[/red]")
        console.print(result.stderr)
        console.print(f"[yellow]Attempting rollback...[/yellow]")
        cp_manager.rollback_to(cp.version)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# neuro lab
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Lab mode commands are imported directly from the module to avoid circular imports
from neuro.modes.lab_mode import run_lab, lab_stats
lab_app.command("run")(run_lab)
lab_app.command("stats")(lab_stats)

if __name__ == "__main__":
    app()
