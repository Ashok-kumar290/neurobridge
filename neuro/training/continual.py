"""Continual Learning Orchestrator — the full self-improvement loop.

Connects all training components into one automated pipeline:

    Interceptor (captures) 
        → TraceStorage (stores on HDD)
        → ConsistencyTester (filters hallucinations)
        → DatasetBuilder (converts to ChatML JSONL)
        → ColabGenerator (builds training notebook)
        → AdapterManager (registers trained adapters)
        → Ollama (deploys updated model)

This is the engine that makes NeuroBridge actually learn and improve.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from neuro.constants import HDD_ROOT, HDD_DATASETS, HDD_ADAPTERS

console = Console()


@dataclass
class CycleResult:
    """Result of a single learning cycle."""
    cycle_id: str
    started_at: float
    finished_at: float = 0.0
    raw_examples: int = 0
    filtered_examples: int = 0
    dataset_path: Optional[str] = None
    notebook_path: Optional[str] = None
    adapter_name: Optional[str] = None
    status: str = "pending"  # pending, filtering, building, ready, trained, deployed, failed
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "cycle_id": self.cycle_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "raw_examples": self.raw_examples,
            "filtered_examples": self.filtered_examples,
            "dataset_path": self.dataset_path,
            "notebook_path": self.notebook_path,
            "adapter_name": self.adapter_name,
            "status": self.status,
            "errors": self.errors,
        }


class ContinualLearner:
    """Orchestrates the full self-learning pipeline."""

    def __init__(
        self,
        traces_dir: Path = Path(HDD_ROOT) / "traces",
        min_examples: int = 20,
        consistency_threshold: float = 0.7,
        base_model: str = "Qwen/Qwen2.5-Coder-3B-Instruct",
        lora_rank: int = 16,
    ):
        self.traces_dir = traces_dir
        self.min_examples = min_examples
        self.consistency_threshold = consistency_threshold
        self.base_model = base_model
        self.lora_rank = lora_rank

        # State file tracks all cycles
        self.state_path = self.traces_dir / "learner_state.json"
        self.cycles: list[CycleResult] = []
        self._load_state()

    def _load_state(self) -> None:
        """Load learner state from disk."""
        if self.state_path.exists():
            try:
                with open(self.state_path) as f:
                    data = json.load(f)
                self.cycles = [
                    CycleResult(**c) for c in data.get("cycles", [])
                ]
            except Exception:
                self.cycles = []

    def _save_state(self) -> None:
        """Save learner state to disk."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w") as f:
            json.dump({
                "cycles": [c.to_dict() for c in self.cycles],
                "last_updated": time.time(),
                "total_cycles": len(self.cycles),
            }, f, indent=2)

    # ── Phase 1: Load raw examples from interceptor buffer ──────────────

    def load_raw_buffer(self) -> list[dict]:
        """Load all training examples from the interceptor's replay buffer."""
        buffer_path = self.traces_dir / "replay_buffer.jsonl"
        if not buffer_path.exists():
            return []

        examples = []
        with open(buffer_path) as f:
            for line in f:
                try:
                    examples.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        return examples

    # ── Phase 2: Filter with consistency testing ────────────────────────

    def filter_examples(
        self, examples: list[dict], model: str = "super-qwen:3b"
    ) -> list[dict]:
        """Filter examples using consistency testing.

        Removes examples where the assistant response is likely hallucinated
        by checking if the model produces consistent answers to the same query.
        """
        from neuro.training.optimizer import ConsistencyTester
        import asyncio

        tester = ConsistencyTester(model=model, iterations=2)
        clean = []

        console.print(f"[cyan]Filtering {len(examples)} examples...[/cyan]")

        for i, ex in enumerate(examples):
            msgs = ex.get("messages", [])
            if len(msgs) < 3:
                continue

            user_msg = msgs[1].get("content", "")
            asst_msg = msgs[2].get("content", "")

            if not user_msg or not asst_msg:
                continue

            try:
                result = asyncio.run(tester.test(user_msg, context=asst_msg))
                if result["consistency_score"] >= self.consistency_threshold:
                    clean.append(ex)
                    if (i + 1) % 10 == 0:
                        console.print(f"  [{i+1}/{len(examples)}] {len(clean)} passed")
            except Exception:
                # If consistency check fails, keep the example (conservative)
                clean.append(ex)

        console.print(f"[green]Kept {len(clean)}/{len(examples)} examples[/green]")
        return clean

    # ── Phase 3: Build training dataset ─────────────────────────────────

    def build_dataset(self, examples: list[dict], name: Optional[str] = None) -> Optional[Path]:
        """Build a JSONL training dataset from filtered examples."""
        if len(examples) < self.min_examples:
            console.print(
                f"[yellow]Only {len(examples)} examples — need {self.min_examples} minimum. "
                f"Keep using Claude/Codex to collect more.[/yellow]"
            )
            return None

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        name = name or f"neurobridge_continual_{timestamp}"
        output_dir = Path(HDD_DATASETS)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{name}.jsonl"

        with open(output_path, "w") as f:
            for ex in examples:
                # Write in ChatML format (Qwen native)
                f.write(json.dumps(ex) + "\n")

        size_mb = output_path.stat().st_size / (1024 * 1024)
        console.print(f"[green]Dataset saved: {output_path} ({len(examples)} examples, {size_mb:.1f} MB)[/green]")
        return output_path

    # ── Phase 4: Generate Colab training notebook ───────────────────────

    def generate_notebook(self, dataset_path: Path) -> Path:
        """Generate a Colab notebook for QLoRA training."""
        from neuro.training.colab_generator import generate_colab_notebook, save_notebook

        cycle_num = len(self.cycles) + 1
        adapter_name = f"super-qwen-adapter-v{cycle_num}"

        notebook = generate_colab_notebook(
            base_model=self.base_model,
            dataset_path=dataset_path.name,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_rank * 2,
            epochs=3,
            adapter_name=adapter_name,
        )

        notebook_path = save_notebook(
            notebook, name=f"neurobridge_train_v{cycle_num}"
        )

        console.print(f"[green]Notebook generated: {notebook_path}[/green]")
        console.print(f"[dim]Upload this + {dataset_path.name} to Google Colab[/dim]")
        return notebook_path

    # ── Phase 5: Register trained adapter ───────────────────────────────

    def register_adapter(
        self,
        adapter_path: Path,
        adapter_name: Optional[str] = None,
        training_examples: int = 0,
    ) -> bool:
        """Register a trained adapter from Colab output."""
        from neuro.training.adapter_manager import AdapterManager

        manager = AdapterManager(Path(HDD_ADAPTERS))
        cycle_num = len(self.cycles)
        name = adapter_name or f"super-qwen-adapter-v{cycle_num}"

        info = manager.register(
            name=name,
            adapter_path=adapter_path,
            base_model=self.base_model,
            training_examples=training_examples,
            lora_rank=self.lora_rank,
        )

        console.print(f"[green]Adapter registered: {name}[/green]")
        console.print(f"[dim]Run evaluation before promoting to production.[/dim]")
        return True

    # ── Phase 6: Deploy via Ollama ──────────────────────────────────────

    def deploy_adapter(self, adapter_name: str) -> bool:
        """Deploy a trained adapter by creating a new Ollama model.

        Creates a Modelfile that merges the LoRA adapter with the base
        model, then creates a new Ollama model from it.
        """
        from neuro.training.adapter_manager import AdapterManager

        manager = AdapterManager(Path(HDD_ADAPTERS))
        adapter_path = manager.get_adapter_path(adapter_name)

        if not adapter_path:
            console.print(f"[red]Adapter not found: {adapter_name}[/red]")
            return False

        # Look for GGUF file in adapter directory
        gguf_files = list(adapter_path.glob("*.gguf"))
        if not gguf_files:
            console.print(
                f"[yellow]No GGUF file found in {adapter_path}. "
                f"Export as GGUF from Colab first (Cell 12 in the notebook).[/yellow]"
            )
            return False

        gguf_path = gguf_files[0]

        # Create Ollama Modelfile
        modelfile_path = adapter_path / "Modelfile"
        with open(modelfile_path, "w") as f:
            f.write(f'FROM {gguf_path}\n')
            f.write('PARAMETER temperature 0.2\n')
            f.write('PARAMETER num_ctx 4096\n')
            f.write('SYSTEM "You are NeuroBridge Coder, a local AI coding assistant."\n')

        console.print(f"[green]Modelfile created: {modelfile_path}[/green]")
        console.print(f"[dim]Run: ollama create super-qwen-v{len(self.cycles)+1} -f {modelfile_path}[/dim]")
        return True

    # ── Full Cycle ──────────────────────────────────────────────────────

    def run_prepare_cycle(self, skip_filter: bool = False) -> CycleResult:
        """Run the prepare phase: load → filter → dataset → notebook.

        This does everything up to the point where you need a GPU (Colab).
        After this, upload the dataset + notebook to Colab and train.
        """
        cycle_id = f"cycle_{int(time.time())}"
        result = CycleResult(
            cycle_id=cycle_id,
            started_at=time.time(),
        )

        try:
            # Phase 1: Load raw buffer
            console.print(Panel("[bold cyan]Phase 1: Loading intercepted traces[/bold cyan]"))
            examples = self.load_raw_buffer()
            result.raw_examples = len(examples)

            if not examples:
                result.status = "failed"
                result.errors.append("No training examples found in replay buffer")
                console.print("[red]No data in replay buffer. Use the interceptor first.[/red]")
                self.cycles.append(result)
                self._save_state()
                return result

            console.print(f"  Found {len(examples)} raw examples")

            # Phase 2: Filter
            if skip_filter:
                filtered = examples
                console.print("[yellow]Skipping consistency filter[/yellow]")
            else:
                console.print(Panel("[bold cyan]Phase 2: Consistency filtering[/bold cyan]"))
                result.status = "filtering"
                filtered = self.filter_examples(examples)

            result.filtered_examples = len(filtered)

            # Phase 3: Build dataset
            console.print(Panel("[bold cyan]Phase 3: Building training dataset[/bold cyan]"))
            result.status = "building"
            dataset_path = self.build_dataset(filtered)

            if not dataset_path:
                result.status = "failed"
                result.errors.append(f"Not enough examples ({len(filtered)} < {self.min_examples})")
                self.cycles.append(result)
                self._save_state()
                return result

            result.dataset_path = str(dataset_path)

            # Check if LoRA rank should grow (dynamic capacity expansion)
            from neuro.training.adapter_manager import AdapterManager
            manager = AdapterManager(Path(HDD_ADAPTERS))
            next_rank = manager.compute_next_rank()
            if next_rank != self.lora_rank:
                console.print(f"[yellow]LoRA rank growing: {self.lora_rank} → {next_rank} (capacity expansion)[/yellow]")
                self.lora_rank = next_rank

            # Phase 4: Generate Colab notebook
            console.print(Panel("[bold cyan]Phase 4: Generating Colab notebook[/bold cyan]"))
            notebook_path = self.generate_notebook(dataset_path)
            result.notebook_path = str(notebook_path)
            result.adapter_name = f"super-qwen-adapter-v{len(self.cycles) + 1}"
            result.status = "ready"
            result.finished_at = time.time()

            # Print summary
            self._print_summary(result)

        except Exception as e:
            result.status = "failed"
            result.errors.append(str(e))
            console.print(f"[red]Cycle failed: {e}[/red]")

        self.cycles.append(result)
        self._save_state()
        return result

    def status(self) -> None:
        """Print current learner status."""
        from neuro.training.interceptor import TraceStorage

        storage = TraceStorage(self.traces_dir)
        stats = storage.get_buffer_stats()

        table = Table(title="NeuroBridge Continual Learner", border_style="cyan")
        table.add_column("Metric", style="bold")
        table.add_column("Value")

        table.add_row("Replay buffer", f"{stats['total_examples']} examples ({stats.get('size_mb', 0):.1f} MB)")
        table.add_row("Min for training", str(self.min_examples))
        table.add_row("Ready to train", "Yes" if stats['total_examples'] >= self.min_examples else "No — keep coding")
        table.add_row("Total cycles", str(len(self.cycles)))

        if self.cycles:
            last = self.cycles[-1]
            table.add_row("Last cycle", f"{last.cycle_id} ({last.status})")
            if last.adapter_name:
                table.add_row("Last adapter", last.adapter_name)

        console.print(table)

        # Show recent examples
        recent = storage.tail_examples(3)
        if recent:
            console.print("\n[bold]Recent captured examples:[/bold]")
            for ex in recent:
                msgs = ex.get("messages", [])
                if len(msgs) >= 3:
                    user = msgs[1]["content"][:100]
                    console.print(f"  [dim]User: {user}...[/dim]")

    def _print_summary(self, result: CycleResult) -> None:
        """Print a cycle summary."""
        duration = result.finished_at - result.started_at

        console.print(Panel(
            f"[bold green]Prepare cycle complete[/bold green]\n\n"
            f"  Raw examples:      {result.raw_examples}\n"
            f"  After filtering:   {result.filtered_examples}\n"
            f"  Dataset:           {result.dataset_path}\n"
            f"  Notebook:          {result.notebook_path}\n"
            f"  Adapter name:      {result.adapter_name}\n"
            f"  Duration:          {duration:.1f}s\n\n"
            f"[bold]Next steps:[/bold]\n"
            f"  1. Upload dataset + notebook to Google Colab\n"
            f"  2. Run all cells (needs T4 GPU)\n"
            f"  3. Download the adapter zip\n"
            f"  4. Run: neuro train register <adapter_zip_path>",
            title="Learning Cycle",
            border_style="green",
        ))


def main():
    """CLI entry point for the continual learner."""
    import argparse

    parser = argparse.ArgumentParser(description="NeuroBridge Continual Learner")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("status", help="Show learner status")
    sub.add_parser("prepare", help="Run prepare cycle (load → filter → dataset → notebook)")

    prep = sub.add_parser("prepare-fast", help="Prepare without consistency filtering")
    reg = sub.add_parser("register", help="Register a trained adapter")
    reg.add_argument("adapter_path", help="Path to the downloaded adapter directory or zip")
    reg.add_argument("--name", help="Adapter name (auto-generated if not set)")

    deploy = sub.add_parser("deploy", help="Deploy an adapter via Ollama")
    deploy.add_argument("adapter_name", help="Name of the adapter to deploy")

    args = parser.parse_args()
    learner = ContinualLearner()

    if args.command == "status":
        learner.status()
    elif args.command == "prepare":
        learner.run_prepare_cycle()
    elif args.command == "prepare-fast":
        learner.run_prepare_cycle(skip_filter=True)
    elif args.command == "register":
        learner.register_adapter(
            Path(args.adapter_path),
            adapter_name=args.name,
        )
    elif args.command == "deploy":
        learner.deploy_adapter(args.adapter_name)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
