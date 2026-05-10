"""Mechanistic Interpretability Lens for NeuroBridge.

Probes the hidden states of models to identify factuality and hallucination directions.

Two operating modes:
  1. Embedding-space steering (works with Ollama — no GPU needed locally)
     Uses cosine similarity in embedding space to score factuality.
  2. Activation-space steering (requires direct model access — Colab)
     Injects steering vectors at specific transformer layers via forward hooks.
     See brain_surgery.py for vector extraction.
"""

import numpy as np
from pathlib import Path
from rich.console import Console

console = Console()

# Default HDD path for steering vectors produced by brain_surgery.py
DEFAULT_VECTOR_DIR = Path("/media/seyominaoto/x/neurobridge/checkpoints/steering_vectors")


class SteeringLens:
    """Interprets and steers model activations at inference time."""

    def __init__(self, model_name: str, vector_dir: Path = DEFAULT_VECTOR_DIR):
        self.model_name = model_name
        self.vector_dir = vector_dir
        self.steering_vectors: dict[str, np.ndarray] = {}
        self.metadata: dict[str, dict] = {}
        self.factuality_layer: int = 16
        self.optimal_alpha: float = 5.0

        # Try to auto-load saved vectors from HDD
        self._auto_load()

    def _auto_load(self) -> None:
        """Auto-load any saved steering vectors from HDD."""
        if not self.vector_dir.exists():
            return

        for npz_file in self.vector_dir.glob("*.npz"):
            try:
                self.load_steering_vector(npz_file.stem, npz_file)
            except Exception as e:
                console.print(f"[dim yellow]Could not load {npz_file.name}: {e}[/dim yellow]")

    def load_steering_vector(self, name: str, vector_path: Path) -> None:
        """Load a pre-calculated steering vector from an .npz file.

        These are produced by brain_surgery.py (Colab) and saved to HDD.
        The .npz contains: steering_vector, layer_idx, optimal_alpha,
        model_id, probe_accuracies, magnitudes.
        """
        vector_path = Path(vector_path)
        if not vector_path.exists():
            console.print(f"[red]Steering vector not found: {vector_path}[/red]")
            return

        data = np.load(vector_path, allow_pickle=True)

        # Handle both key naming conventions:
        #   brain_surgery.py → steering_vector, layer_idx, optimal_alpha, model_id
        #   reconstruct_vector.py → vector, layer, alpha, model
        if "steering_vector" in data:
            self.steering_vectors[name] = data["steering_vector"]
        elif "vector" in data:
            self.steering_vectors[name] = data["vector"]
        else:
            console.print(f"[red]No vector data found in {vector_path}[/red]")
            return

        # Load metadata if present
        meta = {}
        layer = data.get("layer_idx", data.get("layer", None))
        if layer is not None:
            meta["layer_idx"] = int(layer)
            self.factuality_layer = meta["layer_idx"]
        alpha = data.get("optimal_alpha", data.get("alpha", None))
        if alpha is not None:
            meta["optimal_alpha"] = float(alpha)
            self.optimal_alpha = meta["optimal_alpha"]
        model_id = data.get("model_id", data.get("model", None))
        if model_id is not None:
            meta["model_id"] = str(model_id)
        if "probe_accuracies" in data:
            accs = data["probe_accuracies"]
            meta["best_probe_accuracy"] = float(np.max(accs))

        self.metadata[name] = meta
        console.print(
            f"[dim]Loaded steering vector: {name} "
            f"(L{self.factuality_layer}, α={self.optimal_alpha}, "
            f"dim={self.steering_vectors[name].shape[0]})[/dim]"
        )

    def save_steering_vector(self, name: str, path: Path | None = None) -> Path:
        """Save a steering vector to HDD for persistence."""
        if name not in self.steering_vectors:
            raise ValueError(f"No steering vector named '{name}'")

        self.vector_dir.mkdir(parents=True, exist_ok=True)
        save_path = path or (self.vector_dir / f"{name}.npz")

        save_data = {
            "steering_vector": self.steering_vectors[name],
            "layer_idx": self.factuality_layer,
            "optimal_alpha": self.optimal_alpha,
            "model_id": self.model_name,
        }
        np.savez(save_path, **save_data)
        console.print(f"[green]Steering vector saved: {save_path}[/green]")
        return save_path

    def get_steering_prefix(self, query: str) -> str:
        """Generates a system-level steering prefix for Ollama inference.

        Since Ollama doesn't expose hidden states, we use prompt-based
        steering that activates the factuality circuits discovered
        during mechanistic interpretability analysis.
        """
        prefix = (
            "IMPORTANT: Ground all responses in verifiable facts from the provided context. "
            "If you are uncertain, say so explicitly rather than guessing. "
            "Do not fabricate function names, file paths, or API signatures. "
        )

        # If we have factuality metadata, include it
        if "credibility" in self.metadata:
            meta = self.metadata["credibility"]
            acc = meta.get("best_probe_accuracy", 0)
            if acc > 0.85:
                prefix += (
                    "Your internal credibility circuits have been verified. "
                    "Apply your inherent skepticism to all claims. "
                )

        return prefix

    def find_factuality_direction(self, truthful_examples: list[str], hallucinated_examples: list[str]) -> None:
        """Calculates the factuality direction in embedding space.

        Uses Ollama embeddings to compute the mean difference between
        truthful and hallucinated text — a lightweight version of the
        full activation-space analysis done in brain_surgery.py.
        """
        from neuro.runtime.ollama_client import get_ollama_client
        ollama = get_ollama_client()

        truth_vecs = []
        for text in truthful_examples:
            emb = ollama.embeddings(self.model_name, text)
            if emb:
                truth_vecs.append(np.array(emb))

        halluc_vecs = []
        for text in hallucinated_examples:
            emb = ollama.embeddings(self.model_name, text)
            if emb:
                halluc_vecs.append(np.array(emb))

        if not truth_vecs or not halluc_vecs:
            console.print("[red]Could not compute embeddings — is Ollama running?[/red]")
            return

        truth_mean = np.mean(truth_vecs, axis=0)
        halluc_mean = np.mean(halluc_vecs, axis=0)

        direction = truth_mean - halluc_mean
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm

        self.steering_vectors["factuality"] = direction
        console.print(
            f"[green]✓ Factuality direction computed in embedding space "
            f"(dim={direction.shape[0]}, separation={norm:.2f})[/green]"
        )

    def score_factuality(self, response: str) -> float:
        """Scores a response based on its alignment with the factuality vector.

        Returns a score between -1.0 (hallucinated) and 1.0 (factual).
        If no factuality vector is loaded, returns 0.0 (neutral).
        """
        # Check both possible vector names
        vec_name = None
        if "factuality" in self.steering_vectors:
            vec_name = "factuality"
        elif "credibility" in self.steering_vectors:
            vec_name = "credibility"

        if vec_name is None:
            return 0.0

        from neuro.runtime.ollama_client import get_ollama_client
        ollama = get_ollama_client()

        emb = ollama.embeddings(self.model_name, response)
        if not emb:
            return 0.0

        v = np.array(emb)
        truth_vec = self.steering_vectors[vec_name]

        # Handle dimension mismatch (embedding vs activation space)
        if v.shape[0] != truth_vec.shape[0]:
            return 0.0

        norm_v = np.linalg.norm(v)
        norm_t = np.linalg.norm(truth_vec)
        if norm_v == 0 or norm_t == 0:
            return 0.0

        score = np.dot(v, truth_vec) / (norm_v * norm_t)
        return float(score)

    def has_vectors(self) -> bool:
        """Check if any steering vectors are loaded."""
        return len(self.steering_vectors) > 0

    def info(self) -> dict:
        """Get lens status information."""
        return {
            "model": self.model_name,
            "vectors_loaded": list(self.steering_vectors.keys()),
            "factuality_layer": self.factuality_layer,
            "optimal_alpha": self.optimal_alpha,
            "vector_dir": str(self.vector_dir),
            "metadata": {k: v for k, v in self.metadata.items()},
        }
