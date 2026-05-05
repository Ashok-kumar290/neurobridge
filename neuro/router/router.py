"""Master router — the brain that decides which model handles each task.

Decision flow:
  1. Classify task difficulty
  2. Search repo + memory for context
  3. Estimate confidence for local model
  4. If confident → route to local 3B or 7B
  5. If not confident → build expert packet → route to expert
  6. Track routing decisions for learning
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from neuro.router.confidence import ConfidenceScore, estimate_confidence
from neuro.router.difficulty import Difficulty, DifficultyAssessment, estimate_difficulty
from neuro.router.token_budget import TokenBudget, estimate_budget


@dataclass
class RoutingDecision:
    """Complete routing decision for a task."""

    # What was decided
    target: str                    # "local:3b", "local:7b", "cohere", "claude", "codex"
    model: str                     # actual model identifier
    reason: str                    # human-readable explanation

    # Analysis
    difficulty: DifficultyAssessment
    confidence: ConfidenceScore
    budget: TokenBudget

    # Expert-specific
    expert_required: bool = False
    preferred_expert: str | None = None


class Router:
    """Master task router for NeuroBridge."""

    def __init__(self, config: Any = None) -> None:
        from neuro.config import get_config
        self.config = config or get_config()

    def route(
        self,
        query: str,
        context: str = "",
        file_count: int = 0,
        memory_hits: int = 0,
        error_trace: str | None = None,
        previous_failures: int = 0,
        search_score: float = 0.0,
        force_expert: str | None = None,
    ) -> RoutingDecision:
        """Make a routing decision for a task.

        Args:
            query: User's task description
            context: Built context from repo search
            file_count: Number of relevant files
            memory_hits: Similar problems in memory
            error_trace: Error/traceback if debugging
            previous_failures: Local model failure count for this task
            search_score: Best search relevance score
            force_expert: Override to force specific expert ("claude", "codex", "cohere")

        Returns:
            RoutingDecision with full analysis
        """
        # 1. Estimate difficulty
        difficulty = estimate_difficulty(
            query=query,
            file_count=file_count,
            error_trace=error_trace,
            memory_hits=memory_hits,
        )

        # 2. Estimate confidence
        confidence = estimate_confidence(
            search_score=search_score,
            memory_hits=memory_hits,
            file_count=file_count,
            difficulty_score=difficulty.score,
            previous_failures=previous_failures,
        )

        # 3. Force expert if requested
        if force_expert:
            return self._route_to_expert(
                force_expert, query, context, difficulty, confidence,
                reason=f"expert forced by user: --expert {force_expert}",
            )

        # 4. Expert escalation after repeated failures
        if previous_failures >= self.config.routing.expert_after_local_failures:
            preferred = self._pick_expert(query, difficulty)
            return self._route_to_expert(
                preferred, query, context, difficulty, confidence,
                reason=f"local model failed {previous_failures}x, escalating to {preferred}",
            )

        # 5. Auto-escalate expert-level tasks
        if difficulty.difficulty == Difficulty.EXPERT and confidence.should_escalate:
            preferred = self._pick_expert(query, difficulty)
            return self._route_to_expert(
                preferred, query, context, difficulty, confidence,
                reason=f"task rated expert-level (score={difficulty.score}), confidence too low ({confidence.score})",
            )

        # 6. Route to local model
        if difficulty.difficulty in (Difficulty.TRIVIAL, Difficulty.EASY):
            target = "local:3b"
            model = self.config.router.model
            reason = f"simple task (difficulty={difficulty.score}) → 3B router"
        elif difficulty.difficulty == Difficulty.MEDIUM:
            target = "local:7b"
            model = self.config.coder.model
            reason = f"medium task (difficulty={difficulty.score}) → 7B coder"
        else:
            # Hard but confident enough to try locally first
            target = "local:7b"
            model = self.config.coder.model
            reason = f"hard task but local confidence OK ({confidence.score}) → trying 7B first"

        budget = estimate_budget(context, query, model)

        return RoutingDecision(
            target=target,
            model=model,
            reason=reason,
            difficulty=difficulty,
            confidence=confidence,
            budget=budget,
            expert_required=False,
        )

    def _pick_expert(self, query: str, difficulty: DifficultyAssessment) -> str:
        """Pick the best expert model based on task characteristics."""
        query_lower = query.lower()

        # Architecture / design / long-context → Cohere
        if any(w in query_lower for w in ["architecture", "design", "plan", "review"]):
            if self.config.routing.prefer_claude_for_architecture:
                return "claude"
            return "cohere"

        # Patch / fix / command-line → Codex
        if any(w in query_lower for w in ["patch", "fix", "command", "terminal", "cli"]):
            if self.config.routing.prefer_codex_for_patch_tasks:
                return "codex"

        # Long context / RAG → Cohere
        if any(w in query_lower for w in ["entire", "whole", "all files", "long"]):
            if self.config.routing.prefer_cohere_for_long_context:
                return "cohere"

        # Default: Codex for coding, Claude for reasoning
        if difficulty.score > 0.7:
            return "claude"
        return "codex"

    def _route_to_expert(
        self,
        expert: str,
        query: str,
        context: str,
        difficulty: DifficultyAssessment,
        confidence: ConfidenceScore,
        reason: str,
    ) -> RoutingDecision:
        """Create a routing decision for an expert model."""
        model_map = {
            "claude": "claude-code",
            "codex": "codex-cli",
            "cohere": self.config.cohere.planner_model,
        }
        model = model_map.get(expert, expert)
        budget = estimate_budget(context, query, model)

        return RoutingDecision(
            target=f"expert:{expert}",
            model=model,
            reason=reason,
            difficulty=difficulty,
            confidence=confidence,
            budget=budget,
            expert_required=True,
            preferred_expert=expert,
        )
