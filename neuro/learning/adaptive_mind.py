"""Adaptive Mind — the self-improving inference engine.

This is the core of NeuroBridge's novelty. Instead of a static LLM that
gives the same quality response regardless of history, the Adaptive Mind:

  1. RECALLS relevant past experiences before generating
  2. AUGMENTS the prompt with successful past solutions
  3. GENERATES a response grounded in both context and experience
  4. SCORES the response for factuality and consistency
  5. LEARNS the (query, response, score) as a new experience
  6. PRUNES bad experiences over time

The result: a model that gets measurably better the more you use it.
No GPU. No training loop. No Colab. Just use it and it learns.

Architecture:
    Query → [Recall] → [Augment] → [Generate] → [Score] → [Learn]
                ↑                                    ↓
            Experience ← ← ← ← ← ← ← ← ← ← ← Feedback
            Memory (HDD)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from rich.console import Console

console = Console()


@dataclass
class MindResponse:
    """Response from the Adaptive Mind, with metadata."""
    content: str
    model: str
    experience_ids: list[str] = field(default_factory=list)  # which experiences were used
    num_experiences_recalled: int = 0
    factuality_score: float = 0.0
    quality_score: float = 0.0
    generation_time_ms: float = 0.0
    tokens_used: int = 0
    augmented: bool = False  # was the prompt augmented with experiences?
    
    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "model": self.model,
            "experience_ids": self.experience_ids,
            "num_experiences_recalled": self.num_experiences_recalled,
            "factuality_score": self.factuality_score,
            "quality_score": self.quality_score,
            "generation_time_ms": self.generation_time_ms,
            "augmented": self.augmented,
        }


class AdaptiveMind:
    """The self-improving inference engine.
    
    Usage:
        mind = AdaptiveMind()
        
        # Ingest past Codex/Claude sessions
        mind.ingest_buffer()
        
        # Ask a question — experiences are automatically recalled
        response = mind.think("How do I parse JSON in Python?")
        
        # Provide feedback
        mind.feedback(response, score=0.9)
    """
    
    def __init__(
        self,
        model: str = "super-qwen:3b",  # 3B is default — fast on CPU. Use 7B only with GPU.
        memory_dir: Path | None = None,
        embed_model: str = "nomic-embed-text",
        use_steering: bool = True,
        max_recall: int = 3,
        auto_learn: bool = True,
    ):
        from neuro.learning.experience_memory import ExperienceMemory, DEFAULT_MEMORY_DIR
        
        self.model = model
        self.use_steering = use_steering
        self.max_recall = max_recall
        self.auto_learn = auto_learn
        
        # Initialize experience memory
        self.memory = ExperienceMemory(
            memory_dir=memory_dir or DEFAULT_MEMORY_DIR,
            embed_model=embed_model,
        )
        
        # Initialize steering lens (factuality scoring)
        self.lens = None
        if use_steering:
            try:
                from neuro.interpretability.lens import SteeringLens
                self.lens = SteeringLens(model_name=model)
                if self.lens.has_vectors():
                    console.print(f"[dim]Steering lens active: {list(self.lens.steering_vectors.keys())}[/dim]")
            except Exception:
                self.lens = None
        
        # Stats
        self._total_queries = 0
        self._total_augmented = 0
        self._avg_quality = 0.0
    
    # ── Core: Think ─────────────────────────────────────────────────────
    
    def think(
        self,
        query: str,
        context: str = "",
        system: str = "",
        temperature: float = 0.2,
    ) -> MindResponse:
        """Generate a response, augmented by past experiences.
        
        This is the main entry point. It:
          1. Recalls relevant experiences
          2. Builds an augmented prompt
          3. Generates via Ollama
          4. Scores the response
          5. Learns from the interaction
        """
        from neuro.runtime.ollama_client import get_ollama_client
        
        start = time.time()
        self._total_queries += 1
        
        # ── Step 1: Recall ──────────────────────────────────────────────
        experience_context = self.memory.recall_as_prompt(query, top_k=self.max_recall)
        recalled = self.memory.recall(query, top_k=self.max_recall)
        experience_ids = [e.id for e in recalled]
        
        augmented = bool(experience_context)
        if augmented:
            self._total_augmented += 1
        
        # ── Step 2: Build augmented prompt ──────────────────────────────
        system_parts = []
        
        # Base system prompt
        if system:
            system_parts.append(system)
        else:
            system_parts.append(
                "You are NeuroBridge, a local AI coding assistant that learns from experience. "
                "Provide accurate, concise responses grounded in the context provided."
            )
        
        # Steering prefix (factuality bias)
        if self.lens:
            steering_prefix = self.lens.get_steering_prefix(query)
            system_parts.append(steering_prefix)
        
        full_system = "\n\n".join(system_parts)
        
        # Build the user message with experience context
        user_parts = []
        if experience_context:
            user_parts.append(experience_context)
        if context:
            user_parts.append(f"## Current Context\n{context}")
        user_parts.append(f"## Query\n{query}")
        
        full_user = "\n\n".join(user_parts)
        
        # ── Step 3: Generate ────────────────────────────────────────────
        client = get_ollama_client()
        
        messages = [
            {"role": "system", "content": full_system},
            {"role": "user", "content": full_user},
        ]
        
        ollama_resp = client.chat(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        
        generation_time = (time.time() - start) * 1000
        
        # ── Step 4: Score ───────────────────────────────────────────────
        factuality_score = 0.5  # neutral default
        if self.lens:
            factuality_score = self.lens.score_factuality(ollama_resp.content)
            # Normalize from [-1, 1] to [0, 1]
            factuality_score = (factuality_score + 1.0) / 2.0
        
        # Simple quality heuristic based on response characteristics
        quality_score = self._estimate_quality(query, ollama_resp.content, augmented)
        
        # ── Step 5: Learn ───────────────────────────────────────────────
        if self.auto_learn and len(ollama_resp.content) > 20:
            self.memory.learn(
                query=query,
                response=ollama_resp.content,
                context=context,
                source="self",
                quality_score=quality_score,
            )
        
        return MindResponse(
            content=ollama_resp.content,
            model=self.model,
            experience_ids=experience_ids,
            num_experiences_recalled=len(recalled),
            factuality_score=factuality_score,
            quality_score=quality_score,
            generation_time_ms=generation_time,
            tokens_used=ollama_resp.eval_count,
            augmented=augmented,
        )

    # ── Feedback ────────────────────────────────────────────────────────
    
    def feedback(self, response: MindResponse, score: float) -> None:
        """Provide feedback on a response to improve future recalls.
        
        score: 0.0 (terrible) to 1.0 (perfect)
        """
        for exp_id in response.experience_ids:
            # Reinforce experiences that were used in this generation
            self.memory.reinforce(exp_id, score)
    
    # ── Ingest ──────────────────────────────────────────────────────────
    
    def ingest_buffer(self) -> int:
        """Ingest all captured Codex/Claude sessions into experience memory."""
        count = self.memory.learn_from_buffer()
        if count > 0:
            console.print(f"[green]Learned {count} experiences from interceptor buffer[/green]")
        return count
    
    def ingest_example(self, query: str, response: str, source: str = "expert") -> None:
        """Manually teach the mind a single example."""
        self.memory.learn(
            query=query,
            response=response,
            source=source,
            quality_score=0.9,  # manually provided = high quality
        )
    
    # ── Maintenance ─────────────────────────────────────────────────────
    
    def prune(self) -> int:
        """Remove low-quality experiences (hallucinations, junk)."""
        return self.memory.prune(min_score=0.2)
    
    def _estimate_quality(self, query: str, response: str, augmented: bool) -> float:
        """Heuristic quality estimation for a response.
        
        Better than nothing — gets refined by user feedback.
        """
        score = 0.5
        
        # Length appropriateness
        q_len = len(query)
        r_len = len(response)
        if r_len < 10:
            score -= 0.3  # too short
        elif r_len > q_len * 0.5:
            score += 0.1  # reasonable length
        
        # Contains code (good for coding assistant)
        if "```" in response or "def " in response or "import " in response:
            score += 0.1
        
        # Contains hedging (may indicate uncertainty — slightly negative)
        hedges = ["i'm not sure", "i think", "maybe", "possibly", "i don't know"]
        if any(h in response.lower() for h in hedges):
            score -= 0.05
        
        # Augmented responses tend to be better
        if augmented:
            score += 0.05
        
        return max(0.0, min(1.0, score))
    
    # ── Status ──────────────────────────────────────────────────────────
    
    def status(self) -> dict:
        """Get mind status."""
        mem_stats = self.memory.stats()
        return {
            "model": self.model,
            "total_queries": self._total_queries,
            "queries_augmented": self._total_augmented,
            "augmentation_rate": (
                self._total_augmented / max(self._total_queries, 1)
            ),
            "steering_active": self.lens is not None and self.lens.has_vectors(),
            "auto_learn": self.auto_learn,
            **mem_stats,
        }
    
    def info(self) -> None:
        """Print mind status."""
        s = self.status()
        console.print(f"\n[bold cyan]Adaptive Mind Status[/bold cyan]")
        console.print(f"  Model: {s['model']}")
        console.print(f"  Experiences: {s['total_experiences']}")
        console.print(f"  Queries: {s['total_queries']} ({s['queries_augmented']} augmented)")
        console.print(f"  Augmentation rate: {s['augmentation_rate']:.0%}")
        console.print(f"  Steering: {'active' if s['steering_active'] else 'off'}")
        console.print(f"  Avg quality: {s['avg_quality']:.2f}")
        console.print(f"  Sources: {s['by_source']}")
