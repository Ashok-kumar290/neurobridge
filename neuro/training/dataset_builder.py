"""Dataset builder — converts accepted traces into training-ready JSONL.

Takes verified, sanitized, human-approved traces from
  HDD/traces/accepted/
and converts them into instruction-tuning format:
  {"instruction": "...", "input": "...", "output": "..."}

The output JSONL is what gets uploaded to Colab for QLoRA training.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from neuro.constants import HDD_DATASETS, HDD_TRACES


@dataclass
class TrainingExample:
    """A single training example in instruction-tuning format."""

    instruction: str
    input: str
    output: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
        }

    def to_chatml(self) -> dict[str, Any]:
        """Convert to ChatML format (Qwen's native format)."""
        messages = []
        if self.instruction:
            messages.append({"role": "system", "content": self.instruction})
        if self.input:
            messages.append({"role": "user", "content": self.input})
        if self.output:
            messages.append({"role": "assistant", "content": self.output})
        return {"messages": messages}


@dataclass
class DatasetStats:
    """Stats about a built dataset."""

    total_examples: int = 0
    total_tokens_est: int = 0
    by_category: dict[str, int] = field(default_factory=dict)
    avg_input_length: float = 0.0
    avg_output_length: float = 0.0


class DatasetBuilder:
    """Builds training datasets from accepted traces."""

    def __init__(
        self,
        traces_dir: Path = HDD_TRACES / "accepted",
        output_dir: Path = HDD_DATASETS,
    ) -> None:
        self.traces_dir = traces_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_traces(self) -> list[dict[str, Any]]:
        """Load all accepted traces."""
        traces = []
        for path in sorted(self.traces_dir.glob("*.json")):
            try:
                with open(path) as f:
                    trace = json.load(f)
                if trace.get("trainable", False):
                    traces.append(trace)
            except Exception:
                continue
        return traces

    def trace_to_examples(self, trace: dict[str, Any]) -> list[TrainingExample]:
        """Convert a trace into training examples.

        A single trace can produce multiple examples:
          1. Task → full response (main example)
          2. Context → summary (compression training)
          3. Error → fix (debugging training)
        """
        examples = []
        task = trace.get("task", "")
        model = trace.get("model", "unknown")
        steps = trace.get("steps", [])

        # ── Main example: task → response ──────────────────────────────────
        context_parts = []
        response_parts = []

        for step in steps:
            step_type = step.get("step_type", "")
            data = step.get("data", {})

            if step_type == "context":
                # Repo context that was provided
                context_parts.append(data.get("content", ""))

            elif step_type in ("output", "model_call"):
                # Model's response
                resp = data.get("response", "") or data.get("content", "")
                if resp:
                    response_parts.append(resp)

            elif step_type == "search":
                # Search results used as context
                results = data.get("results", [])
                for r in results[:3]:
                    if isinstance(r, dict):
                        context_parts.append(r.get("content", ""))
                    elif isinstance(r, str):
                        context_parts.append(r)

        context = "\n\n".join(p for p in context_parts if p)[:4000]
        response = "\n".join(p for p in response_parts if p)

        if task and response:
            instruction = (
                "You are NeuroBridge Coder, a local AI coding assistant. "
                "Answer the coding task using the provided repository context."
            )

            input_text = task
            if context:
                input_text = f"## Repository Context\n{context}\n\n## Task\n{task}"

            examples.append(TrainingExample(
                instruction=instruction,
                input=input_text,
                output=response,
                metadata={
                    "trace_id": trace.get("trace_id", ""),
                    "model": model,
                    "category": "coding",
                },
            ))

        return examples

    def build_dataset(
        self,
        name: str | None = None,
        format: str = "chatml",
        min_examples: int = 10,
    ) -> tuple[Path | None, DatasetStats]:
        """Build a training dataset from accepted traces.

        Args:
            name: Dataset name (auto-generated if None)
            format: "chatml" (Qwen native) or "alpaca" (instruction format)
            min_examples: Minimum examples required

        Returns:
            (output_path, stats) or (None, stats) if not enough data
        """
        traces = self.load_traces()
        examples: list[TrainingExample] = []

        for trace in traces:
            examples.extend(self.trace_to_examples(trace))

        stats = DatasetStats(
            total_examples=len(examples),
            total_tokens_est=sum(
                len((e.input + e.output).split()) * 1.3  # rough token estimate
                for e in examples
            ),
        )

        if not examples:
            return None, stats

        # Calculate averages
        stats.avg_input_length = sum(len(e.input) for e in examples) / len(examples)
        stats.avg_output_length = sum(len(e.output) for e in examples) / len(examples)

        # Count by category
        for e in examples:
            cat = e.metadata.get("category", "unknown")
            stats.by_category[cat] = stats.by_category.get(cat, 0) + 1

        if len(examples) < min_examples:
            return None, stats

        # ── Write dataset ──────────────────────────────────────────────────
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        name = name or f"neurobridge_{timestamp}"
        output_path = self.output_dir / f"{name}.jsonl"

        with open(output_path, "w") as f:
            for example in examples:
                if format == "chatml":
                    f.write(json.dumps(example.to_chatml()) + "\n")
                else:
                    f.write(json.dumps(example.to_dict()) + "\n")

        stats.total_examples = len(examples)
        return output_path, stats

    def get_dataset_info(self) -> list[dict[str, Any]]:
        """List existing datasets."""
        datasets = []
        for path in sorted(self.output_dir.glob("*.jsonl")):
            line_count = 0
            with open(path) as f:
                for _ in f:
                    line_count += 1
            datasets.append({
                "name": path.stem,
                "path": str(path),
                "examples": line_count,
                "size": f"{path.stat().st_size / 1024:.1f} KB",
            })
        return datasets
