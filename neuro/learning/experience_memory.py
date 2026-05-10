"""Experience Memory — the brain's episodic memory system.

Instead of traditional weight updates (which need GPU), NeuroBridge learns
by accumulating experiences and retrieving them at inference time.

How it works:
  1. Every interaction (from Codex, Claude, or user feedback) becomes an "experience"
  2. Each experience is embedded and stored in a vector index on HDD
  3. At inference time, similar past experiences are retrieved and injected into the prompt
  4. The model generates a response grounded in successful past solutions
  5. Responses are scored and the score updates the experience quality

This is non-parametric learning — knowledge grows by expanding memory, not updating weights.
The more you use it, the smarter it gets. Instantly. No training loop.

Inspired by:
  - Episodic memory in cognitive science
  - RAG but for the model's own past experiences  
  - Few-shot learning via retrieval
"""

from __future__ import annotations

import json
import time
import hashlib
import numpy as np
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

from rich.console import Console

console = Console()

# HDD storage for the experience memory
DEFAULT_MEMORY_DIR = Path("/media/seyominaoto/x/neurobridge/brain/experiences")


@dataclass
class Experience:
    """A single learned experience."""
    
    id: str                          # unique hash
    query: str                       # what was asked
    response: str                    # the best response
    context: str = ""                # repo context, file contents, etc.
    source: str = "unknown"          # "codex", "claude", "user", "self"
    quality_score: float = 0.5       # 0.0 (terrible) to 1.0 (perfect)
    factuality_score: float = 0.5    # from SteeringLens
    consistency_score: float = 0.5   # from ConsistencyTester
    access_count: int = 0            # how often this experience was retrieved
    created_at: float = 0.0
    updated_at: float = 0.0
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @staticmethod
    def make_id(query: str, response: str) -> str:
        """Generate a deterministic ID from query+response."""
        content = f"{query}|||{response}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def combined_score(self) -> float:
        """Weighted combination of all quality signals."""
        return (
            0.4 * self.quality_score +
            0.3 * self.factuality_score +
            0.2 * self.consistency_score +
            0.1 * min(self.access_count / 10.0, 1.0)  # usage bonus
        )
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "Experience":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class VectorIndex:
    """Simple but effective vector similarity index using numpy.
    
    Stores embeddings as a memory-mapped numpy array on HDD.
    For NeuroBridge's scale (thousands of experiences, not millions),
    brute-force cosine similarity is fast enough (<10ms for 10K vectors).
    """
    
    def __init__(self, index_dir: Path, dim: int = 768):
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        
        self.vectors_path = self.index_dir / "vectors.npy"
        self.ids_path = self.index_dir / "ids.json"
        
        # Load or initialize
        self.vectors: np.ndarray  # shape: (N, dim)
        self.ids: list[str]
        self._load()
    
    def _load(self) -> None:
        """Load index from disk."""
        if self.vectors_path.exists() and self.ids_path.exists():
            self.vectors = np.load(self.vectors_path)
            with open(self.ids_path) as f:
                self.ids = json.load(f)
            # Ensure dimension consistency
            if self.vectors.shape[0] > 0:
                self.dim = self.vectors.shape[1]
        else:
            self.vectors = np.zeros((0, self.dim), dtype=np.float32)
            self.ids = []
    
    def _save(self) -> None:
        """Persist index to disk."""
        np.save(self.vectors_path, self.vectors)
        with open(self.ids_path, "w") as f:
            json.dump(self.ids, f)
    
    def add(self, id: str, vector: np.ndarray) -> None:
        """Add a vector to the index."""
        vector = np.array(vector, dtype=np.float32).reshape(1, -1)
        
        # Handle dimension change (first vector sets the dimension)
        if self.vectors.shape[0] == 0:
            self.dim = vector.shape[1]
            self.vectors = vector
        else:
            if vector.shape[1] != self.dim:
                return  # skip mismatched dimensions
            self.vectors = np.vstack([self.vectors, vector])
        
        self.ids.append(id)
        self._save()
    
    def update(self, id: str, vector: np.ndarray) -> None:
        """Update an existing vector."""
        if id in self.ids:
            idx = self.ids.index(id)
            self.vectors[idx] = np.array(vector, dtype=np.float32)
            self._save()
        else:
            self.add(id, vector)
    
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        """Find the top-K most similar vectors by cosine similarity.
        
        Returns list of (id, similarity_score) tuples, sorted by score descending.
        """
        if self.vectors.shape[0] == 0:
            return []
        
        query = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        
        # Cosine similarity: dot(q, v) / (|q| * |v|)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return []
        
        vec_norms = np.linalg.norm(self.vectors, axis=1)
        # Avoid division by zero
        valid_mask = vec_norms > 0
        
        similarities = np.zeros(self.vectors.shape[0])
        if valid_mask.any():
            similarities[valid_mask] = (
                np.dot(self.vectors[valid_mask], query.T).flatten() /
                (vec_norms[valid_mask] * query_norm)
            )
        
        # Top-K
        top_k = min(top_k, len(similarities))
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append((self.ids[idx], float(similarities[idx])))
        
        return results
    
    def size(self) -> int:
        return len(self.ids)
    
    def remove(self, id: str) -> None:
        """Remove a vector from the index."""
        if id not in self.ids:
            return
        idx = self.ids.index(id)
        self.vectors = np.delete(self.vectors, idx, axis=0)
        self.ids.pop(idx)
        self._save()


class ExperienceMemory:
    """The brain's episodic memory — learns from every interaction.
    
    Usage:
        memory = ExperienceMemory()
        
        # Learn from a Codex response
        memory.learn(query="How to read a CSV?", response="Use pandas...", source="codex")
        
        # Recall similar experiences at inference time
        experiences = memory.recall("How to parse a CSV file?", top_k=3)
        
        # Update quality after feedback
        memory.reinforce(experience_id, score=0.9)
    """
    
    def __init__(
        self,
        memory_dir: Path = DEFAULT_MEMORY_DIR,
        embed_model: str = "nomic-embed-text",
    ):
        self.memory_dir = memory_dir
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.embed_model = embed_model
        
        # Storage
        self.experiences_path = self.memory_dir / "experiences.jsonl"
        self.index = VectorIndex(self.memory_dir / "index")
        
        # In-memory cache of experiences (loaded from disk)
        self._cache: dict[str, Experience] = {}
        self._load_experiences()
    
    def _load_experiences(self) -> None:
        """Load all experiences from JSONL into memory."""
        if not self.experiences_path.exists():
            return
        
        with open(self.experiences_path) as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    exp = Experience.from_dict(data)
                    self._cache[exp.id] = exp
                except (json.JSONDecodeError, Exception):
                    continue
    
    def _save_experience(self, exp: Experience) -> None:
        """Append a single experience to the JSONL store."""
        with open(self.experiences_path, "a") as f:
            f.write(json.dumps(exp.to_dict()) + "\n")
    
    def _rewrite_experiences(self) -> None:
        """Rewrite the entire experiences file (after updates)."""
        with open(self.experiences_path, "w") as f:
            for exp in self._cache.values():
                f.write(json.dumps(exp.to_dict()) + "\n")
    
    def _embed(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for text via Ollama."""
        try:
            from neuro.runtime.ollama_client import get_ollama_client
            client = get_ollama_client()
            emb = client.embeddings(self.embed_model, text)
            if emb:
                return np.array(emb, dtype=np.float32)
        except Exception as e:
            console.print(f"[dim red]Embedding failed: {e}[/dim red]")
        return None
    
    # ── Learn ───────────────────────────────────────────────────────────
    
    def learn(
        self,
        query: str,
        response: str,
        context: str = "",
        source: str = "unknown",
        quality_score: float = 0.7,
        tags: list[str] | None = None,
        metadata: dict | None = None,
    ) -> Optional[Experience]:
        """Learn from a new interaction. Returns the created Experience."""
        
        exp_id = Experience.make_id(query, response)
        
        # Skip if we already have this exact experience
        if exp_id in self._cache:
            # Just bump the quality if it's from a better source
            existing = self._cache[exp_id]
            if quality_score > existing.quality_score:
                existing.quality_score = quality_score
                existing.updated_at = time.time()
                self._rewrite_experiences()
            return existing
        
        # Create experience
        exp = Experience(
            id=exp_id,
            query=query,
            response=response,
            context=context,
            source=source,
            quality_score=quality_score,
            tags=tags or [],
            metadata=metadata or {},
            created_at=time.time(),
            updated_at=time.time(),
        )
        
        # Embed and index
        # Embed the query (what was asked) — this is what we search by
        embedding = self._embed(query)
        if embedding is not None:
            self.index.add(exp_id, embedding)
        
        # Store
        self._cache[exp_id] = exp
        self._save_experience(exp)
        
        return exp
    
    def learn_from_buffer(self, buffer_path: Path | None = None) -> int:
        """Bulk-learn from the interceptor's replay buffer.
        
        Returns the number of new experiences learned.
        """
        if buffer_path is None:
            buffer_path = Path("/media/seyominaoto/x/neurobridge/traces/replay_buffer.jsonl")
        
        if not buffer_path.exists():
            return 0
        
        learned = 0
        with open(buffer_path) as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    msgs = data.get("messages", [])
                    if len(msgs) < 3:
                        continue
                    
                    query = msgs[1].get("content", "")
                    response = msgs[2].get("content", "")
                    source = data.get("metadata", {}).get("tool", "unknown")
                    
                    if len(query) < 5 or len(response) < 10:
                        continue
                    
                    exp = self.learn(
                        query=query,
                        response=response,
                        source=source,
                        quality_score=0.7 if source in ("codex", "claude") else 0.5,
                    )
                    if exp:
                        learned += 1
                except (json.JSONDecodeError, Exception):
                    continue
        
        return learned
    
    # ── Recall ──────────────────────────────────────────────────────────
    
    def recall(self, query: str, top_k: int = 3, min_quality: float = 0.1) -> list[Experience]:
        """Recall relevant past experiences for a query.
        
        Returns the top-K most similar experiences, filtered by minimum quality.
        """
        embedding = self._embed(query)
        if embedding is None:
            return []
        
        results = self.index.search(embedding, top_k=top_k * 2)  # fetch extra, then filter
        
        experiences = []
        for exp_id, similarity in results:
            if exp_id not in self._cache:
                continue
            
            exp = self._cache[exp_id]
            
            # Skip low-quality experiences
            if exp.combined_score() < min_quality:
                continue
            
            # Bump access count
            exp.access_count += 1
            
            experiences.append(exp)
            if len(experiences) >= top_k:
                break
        
        return experiences
    
    def recall_as_prompt(self, query: str, top_k: int = 3) -> str:
        """Recall experiences and format them as a prompt injection.
        
        This is the key method — it turns past experiences into
        few-shot examples that guide the model's response.
        """
        experiences = self.recall(query, top_k=top_k)
        
        if not experiences:
            return ""
        
        lines = ["## Relevant Past Experiences (use these to inform your response):\n"]
        
        for i, exp in enumerate(experiences, 1):
            lines.append(f"### Experience {i} (quality: {exp.combined_score():.1f}, source: {exp.source})")
            lines.append(f"**Question:** {exp.query[:500]}")
            # Truncate long responses to keep prompt manageable
            resp = exp.response[:1500]
            if len(exp.response) > 1500:
                resp += "\n... (truncated)"
            lines.append(f"**Answer:** {resp}")
            lines.append("")
        
        lines.append("Use the above experiences as reference. Adapt them to the current query.\n")
        
        return "\n".join(lines)
    
    # ── Reinforce ───────────────────────────────────────────────────────
    
    def reinforce(self, exp_id: str, score: float) -> None:
        """Update the quality score of an experience based on feedback.
        
        Call this when:
          - User confirms a response was helpful → score up
          - Consistency test passes → score up
          - Factuality check fails → score down
          - User corrects a response → score down, learn the correction
        """
        if exp_id not in self._cache:
            return
        
        exp = self._cache[exp_id]
        # Exponential moving average — recent feedback matters more
        alpha = 0.3
        exp.quality_score = alpha * score + (1 - alpha) * exp.quality_score
        exp.updated_at = time.time()
        self._rewrite_experiences()
    
    def reinforce_factuality(self, exp_id: str, factuality_score: float) -> None:
        """Update factuality score from SteeringLens evaluation."""
        if exp_id not in self._cache:
            return
        exp = self._cache[exp_id]
        exp.factuality_score = factuality_score
        exp.updated_at = time.time()
    
    # ── Prune ───────────────────────────────────────────────────────────
    
    def prune(self, min_score: float = 0.2) -> int:
        """Remove low-quality experiences (hallucinations, junk, etc.).
        
        Returns number of pruned experiences.
        """
        to_remove = [
            exp_id for exp_id, exp in self._cache.items()
            if exp.combined_score() < min_score
        ]
        
        for exp_id in to_remove:
            del self._cache[exp_id]
            self.index.remove(exp_id)
        
        if to_remove:
            self._rewrite_experiences()
        
        return len(to_remove)
    
    # ── Stats ───────────────────────────────────────────────────────────
    
    def stats(self) -> dict:
        """Get memory statistics."""
        if not self._cache:
            return {
                "total_experiences": 0,
                "avg_quality": 0.0,
                "by_source": {},
                "index_size": self.index.size(),
            }
        
        scores = [e.combined_score() for e in self._cache.values()]
        sources = {}
        for exp in self._cache.values():
            sources[exp.source] = sources.get(exp.source, 0) + 1
        
        return {
            "total_experiences": len(self._cache),
            "avg_quality": sum(scores) / len(scores),
            "top_quality": max(scores),
            "by_source": sources,
            "index_size": self.index.size(),
            "total_accesses": sum(e.access_count for e in self._cache.values()),
        }
    
    def info(self) -> None:
        """Print memory status."""
        s = self.stats()
        console.print(f"\n[bold cyan]Experience Memory[/bold cyan]")
        console.print(f"  Experiences: {s['total_experiences']}")
        console.print(f"  Avg quality: {s['avg_quality']:.2f}")
        console.print(f"  Sources: {s['by_source']}")
        console.print(f"  Index vectors: {s['index_size']}")
        console.print(f"  Total recalls: {s['total_accesses']}")
