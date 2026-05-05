"""Checkpoint manager — handles model versioning and rollback on HDD.

Every time an adapter is promoted, the current model state is
checkpointed. If a new adapter causes regression, we can roll back
to the previous checkpoint instantly.

Checkpoint structure on HDD:
  adapters/checkpoints/
    v1_20260505_140000/
      metadata.json
      Modelfile
      eval_report.json
    v2_20260505_150000/
      ...
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
class Checkpoint:
    """A model checkpoint."""

    version: str
    model_name: str
    adapter_name: str | None
    created_at: float
    eval_scores: dict[str, float] = field(default_factory=dict)
    modelfile_content: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "model_name": self.model_name,
            "adapter_name": self.adapter_name,
            "created_at": self.created_at,
            "iso_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.created_at)),
            "eval_scores": self.eval_scores,
            "modelfile_content": self.modelfile_content,
            "notes": self.notes,
        }


class CheckpointManager:
    """Manages model checkpoints on HDD."""

    def __init__(self) -> None:
        self.checkpoints_dir = HDD_ADAPTERS / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.checkpoints_dir / "registry.json"

    def _load_registry(self) -> dict[str, Any]:
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                return json.load(f)
        return {"checkpoints": [], "active_version": None}

    def _save_registry(self, registry: dict[str, Any]) -> None:
        with open(self.registry_path, "w") as f:
            json.dump(registry, f, indent=2)

    def create_checkpoint(
        self,
        model_name: str,
        adapter_name: str | None = None,
        eval_scores: dict[str, float] | None = None,
        modelfile_path: Path | None = None,
        notes: str = "",
    ) -> Checkpoint:
        """Create a new checkpoint before promoting an adapter.

        This saves the current model state so we can rollback.
        """
        registry = self._load_registry()
        version_num = len(registry["checkpoints"]) + 1
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        version = f"v{version_num}"

        # Create checkpoint directory
        cp_dir = self.checkpoints_dir / f"{version}_{timestamp}"
        cp_dir.mkdir(parents=True, exist_ok=True)

        # Read Modelfile content if available
        modelfile_content = ""
        if modelfile_path and modelfile_path.exists():
            modelfile_content = modelfile_path.read_text()

        checkpoint = Checkpoint(
            version=version,
            model_name=model_name,
            adapter_name=adapter_name,
            created_at=time.time(),
            eval_scores=eval_scores or {},
            modelfile_content=modelfile_content,
            notes=notes,
        )

        # Save metadata
        with open(cp_dir / "metadata.json", "w") as f:
            json.dump(checkpoint.to_dict(), f, indent=2)

        # Save Modelfile
        if modelfile_content:
            with open(cp_dir / "Modelfile", "w") as f:
                f.write(modelfile_content)

        # Save eval scores
        if eval_scores:
            with open(cp_dir / "eval_scores.json", "w") as f:
                json.dump(eval_scores, f, indent=2)

        # Update registry
        registry["checkpoints"].append(checkpoint.to_dict())
        registry["active_version"] = version
        self._save_registry(registry)

        return checkpoint

    def list_checkpoints(self) -> list[dict[str, Any]]:
        """List all checkpoints."""
        registry = self._load_registry()
        return registry.get("checkpoints", [])

    def get_active_version(self) -> str | None:
        """Get the currently active checkpoint version."""
        registry = self._load_registry()
        return registry.get("active_version")

    def get_checkpoint(self, version: str) -> dict[str, Any] | None:
        """Get a specific checkpoint by version."""
        for cp in self.list_checkpoints():
            if cp["version"] == version:
                return cp
        return None

    def rollback_to(self, version: str) -> dict[str, Any] | None:
        """Rollback to a previous checkpoint version.

        This updates the registry to point to the rollback version.
        The actual Ollama model recreation must be done separately
        using the Modelfile stored in the checkpoint.
        """
        from neuro.safety.audit_logger import get_audit_logger

        checkpoint = self.get_checkpoint(version)
        if not checkpoint:
            return None

        registry = self._load_registry()
        registry["active_version"] = version
        self._save_registry(registry)

        get_audit_logger().log_generic(
            "checkpoint_rollback",
            version=version,
            model=checkpoint.get("model_name", "?"),
        )

        return checkpoint

    def should_rollback(
        self,
        current_scores: dict[str, float],
        baseline_scores: dict[str, float],
        threshold: float = 0.05,
    ) -> tuple[bool, str]:
        """Determine if we should rollback based on eval regression.

        Returns (should_rollback, reason).
        A rollback is recommended if pass_rate drops by more than threshold.
        """
        current_rate = current_scores.get("pass_rate", 0)
        baseline_rate = baseline_scores.get("pass_rate", 0)
        delta = current_rate - baseline_rate

        if delta < -threshold:
            return True, (
                f"Regression detected: {current_rate:.0%} vs {baseline_rate:.0%} "
                f"(Δ={delta:+.0%}, threshold={threshold:.0%})"
            )

        # Check individual categories
        for cat in baseline_scores:
            if cat == "pass_rate":
                continue
            current_cat = current_scores.get(cat, 0)
            baseline_cat = baseline_scores.get(cat, 0)
            cat_delta = current_cat - baseline_cat
            if cat_delta < -(threshold * 2):  # stricter for individual categories
                return True, (
                    f"Category regression in {cat}: {current_cat:.0%} vs "
                    f"{baseline_cat:.0%} (Δ={cat_delta:+.0%})"
                )

        return False, f"No regression (Δ={delta:+.0%})"

    def generate_promotion_modelfile(
        self,
        base_model: str,
        adapter_path: Path,
        model_name: str,
        system_prompt: str | None = None,
    ) -> str:
        """Generate an Ollama Modelfile that includes an adapter.

        This creates a Modelfile with the ADAPTER directive
        so Ollama loads the LoRA adapter on top of the base model.
        """
        if system_prompt is None:
            system_prompt = (
                "You are Super-Qwen, a powerful local AI coding assistant "
                "built by NeuroBridge. You write clean, production-quality code."
            )

        modelfile = f"""# {model_name} — NeuroBridge Promoted Model
# Auto-generated by checkpoint manager

FROM {base_model}

ADAPTER {adapter_path}

SYSTEM \"\"\"{system_prompt}\"\"\"

PARAMETER temperature 0.2
PARAMETER top_k 40
PARAMETER top_p 0.85
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096
PARAMETER num_predict 2048
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
"""
        return modelfile

    def get_stats(self) -> dict[str, Any]:
        """Get checkpoint statistics."""
        checkpoints = self.list_checkpoints()
        return {
            "total_checkpoints": len(checkpoints),
            "active_version": self.get_active_version(),
            "latest": checkpoints[-1] if checkpoints else None,
        }
