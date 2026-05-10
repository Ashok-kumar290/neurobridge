"""Adapter manager — handles LoRA adapter lifecycle.

Manages the full adapter lifecycle:
  1. Register a new adapter from Colab training output
  2. Test adapter against eval baselines
  3. Promote adapter (create new Ollama model with adapter merged)
  4. Rollback to previous adapter if regression detected

Adapters are stored on HDD: /media/.../neurobridge/adapters/
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from neuro.constants import HDD_ADAPTERS


@dataclass
class AdapterInfo:
    """Metadata about a trained adapter."""

    name: str
    base_model: str
    created_at: float
    training_examples: int = 0
    lora_rank: int = 16
    lora_alpha: int = 32
    epochs: int = 3
    eval_scores: dict[str, float] = field(default_factory=dict)
    status: str = "registered"  # registered → tested → promoted → retired
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "base_model": self.base_model,
            "created_at": self.created_at,
            "training_examples": self.training_examples,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "epochs": self.epochs,
            "eval_scores": self.eval_scores,
            "status": self.status,
            "notes": self.notes,
        }


class AdapterManager:
    """Manages LoRA adapters on HDD."""

    def __init__(self, adapters_dir: Path = HDD_ADAPTERS) -> None:
        self.adapters_dir = adapters_dir
        self.adapters_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.adapters_dir / "registry.json"

    def _load_registry(self) -> dict[str, Any]:
        """Load the adapter registry."""
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                return json.load(f)
        return {"adapters": [], "active": None}

    def _save_registry(self, registry: dict[str, Any]) -> None:
        """Save the adapter registry."""
        with open(self.registry_path, "w") as f:
            json.dump(registry, f, indent=2)

    def register(
        self,
        name: str,
        adapter_path: Path,
        base_model: str,
        training_examples: int = 0,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        epochs: int = 3,
        notes: str = "",
    ) -> AdapterInfo:
        """Register a new adapter from training output.

        Copies adapter files to HDD and adds to registry.
        """
        # Create adapter directory
        adapter_dir = self.adapters_dir / name
        adapter_dir.mkdir(parents=True, exist_ok=True)

        # Copy adapter files
        if adapter_path.is_dir():
            for f in adapter_path.iterdir():
                shutil.copy2(f, adapter_dir / f.name)
        elif adapter_path.is_file():
            shutil.copy2(adapter_path, adapter_dir / adapter_path.name)

        # Create adapter info
        info = AdapterInfo(
            name=name,
            base_model=base_model,
            created_at=time.time(),
            training_examples=training_examples,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            epochs=epochs,
            notes=notes,
        )

        # Save metadata
        with open(adapter_dir / "metadata.json", "w") as f:
            json.dump(info.to_dict(), f, indent=2)

        # Add to registry
        registry = self._load_registry()
        registry["adapters"].append(info.to_dict())
        self._save_registry(registry)

        return info

    def list_adapters(self) -> list[dict[str, Any]]:
        """List all registered adapters."""
        registry = self._load_registry()
        return registry.get("adapters", [])

    def get_active(self) -> str | None:
        """Get the currently active adapter name."""
        registry = self._load_registry()
        return registry.get("active")

    def promote(self, name: str, eval_scores: dict[str, float]) -> bool:
        """Promote an adapter to active status.

        This updates the registry and logs the promotion.
        To actually use the adapter, a new Ollama model must be created
        with the adapter merged (done via Modelfile + ADAPTER directive).
        """
        from neuro.safety.audit_logger import get_audit_logger

        registry = self._load_registry()

        # Find adapter
        found = False
        for adapter in registry["adapters"]:
            if adapter["name"] == name:
                adapter["status"] = "promoted"
                adapter["eval_scores"] = eval_scores
                found = True
            elif adapter.get("status") == "promoted":
                adapter["status"] = "retired"

        if not found:
            return False

        registry["active"] = name
        self._save_registry(registry)

        get_audit_logger().log_adapter_promoted(name, eval_scores)
        return True

    def get_adapter_path(self, name: str) -> Path | None:
        """Get the path to an adapter's files."""
        adapter_dir = self.adapters_dir / name
        if adapter_dir.exists():
            return adapter_dir
        return None

    def get_stats(self) -> dict[str, Any]:
        """Get adapter statistics."""
        adapters = self.list_adapters()
        return {
            "total": len(adapters),
            "active": self.get_active(),
            "by_status": {
                status: sum(1 for a in adapters if a.get("status") == status)
                for status in ["registered", "tested", "promoted", "retired"]
            },
        }

    def compute_next_rank(self) -> int:
        """Compute the next LoRA rank based on adapter history.

        This implements the 'dynamic growth' mechanism:
        - Starts at rank 16
        - If the last adapter's eval scores improved, keep the same rank
        - If they plateaued (< 2% improvement), double the rank
        - Never exceeds MAX_LORA_RANK (64)

        The intuition: when a rank can no longer compress new information,
        the model needs more capacity — like growing new neurons.
        """
        from neuro.constants import DEFAULT_LORA_RANK, MAX_LORA_RANK

        adapters = self.list_adapters()
        if not adapters:
            return DEFAULT_LORA_RANK

        # Find the last two promoted adapters to compare
        promoted = [a for a in adapters if a.get("status") in ("promoted", "retired")]
        if len(promoted) < 2:
            # Not enough history — use the last adapter's rank
            last = adapters[-1]
            return min(last.get("lora_rank", DEFAULT_LORA_RANK), MAX_LORA_RANK)

        prev = promoted[-2]
        curr = promoted[-1]

        # Compare eval scores
        prev_scores = prev.get("eval_scores", {})
        curr_scores = curr.get("eval_scores", {})

        if not prev_scores or not curr_scores:
            return curr.get("lora_rank", DEFAULT_LORA_RANK)

        # Calculate average improvement
        common_keys = set(prev_scores.keys()) & set(curr_scores.keys())
        if not common_keys:
            return curr.get("lora_rank", DEFAULT_LORA_RANK)

        improvements = [
            curr_scores[k] - prev_scores[k]
            for k in common_keys
        ]
        avg_improvement = sum(improvements) / len(improvements)

        current_rank = curr.get("lora_rank", DEFAULT_LORA_RANK)

        if avg_improvement < 0.02:
            # Plateaued — grow capacity
            new_rank = min(current_rank * 2, MAX_LORA_RANK)
            return new_rank
        else:
            # Still improving — keep current rank
            return current_rank

    def get_growth_history(self) -> list[dict[str, Any]]:
        """Get the rank growth history across all adapters."""
        adapters = self.list_adapters()
        return [
            {
                "name": a.get("name"),
                "rank": a.get("lora_rank"),
                "examples": a.get("training_examples"),
                "scores": a.get("eval_scores", {}),
                "status": a.get("status"),
            }
            for a in adapters
        ]
