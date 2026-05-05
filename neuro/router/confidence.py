"""Confidence scorer — estimates local model confidence for a task.

Uses search quality, memory coverage, and task alignment to decide
whether the local model can handle this or needs expert escalation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ConfidenceScore:
    """How confident we are that the local model can handle this."""

    score: float          # 0.0 (no confidence) to 1.0 (very confident)
    should_escalate: bool
    reasons: list[str]


def estimate_confidence(
    search_score: float = 0.0,
    memory_hits: int = 0,
    file_count: int = 0,
    difficulty_score: float = 0.5,
    previous_failures: int = 0,
) -> ConfidenceScore:
    """Estimate confidence that local model can solve this task.

    Args:
        search_score: Best search relevance score (higher = better match)
        memory_hits: Number of similar solved problems in memory
        file_count: Number of relevant files found
        difficulty_score: From difficulty estimator (0-1)
        previous_failures: How many times local model already failed this task

    Returns:
        ConfidenceScore with escalation recommendation
    """
    from neuro.constants import EXPERT_ESCALATION_AFTER_FAILURES

    reasons: list[str] = []
    confidence = 0.5  # neutral start

    # ── Search quality ─────────────────────────────────────────────────────
    if search_score > 5.0:
        confidence += 0.2
        reasons.append("strong search matches")
    elif search_score > 1.0:
        confidence += 0.1
        reasons.append("decent search matches")
    elif search_score < 0.5 and file_count > 0:
        confidence -= 0.1
        reasons.append("weak search relevance")

    # ── Memory coverage ────────────────────────────────────────────────────
    if memory_hits > 2:
        confidence += 0.25
        reasons.append(f"solved similar problems before ({memory_hits}x)")
    elif memory_hits > 0:
        confidence += 0.1
        reasons.append("some related memory")
    else:
        confidence -= 0.05
        reasons.append("no memory of similar tasks")

    # ── File availability ──────────────────────────────────────────────────
    if file_count >= 3:
        confidence += 0.1
        reasons.append("good file coverage")
    elif file_count == 0:
        confidence -= 0.2
        reasons.append("no relevant files found")

    # ── Difficulty inverse ─────────────────────────────────────────────────
    if difficulty_score > 0.7:
        confidence -= 0.2
        reasons.append("task is rated hard/expert")
    elif difficulty_score < 0.3:
        confidence += 0.15
        reasons.append("task is rated easy/trivial")

    # ── Previous failures ──────────────────────────────────────────────────
    if previous_failures >= EXPERT_ESCALATION_AFTER_FAILURES:
        confidence -= 0.4
        reasons.append(f"local model failed {previous_failures}x already")
    elif previous_failures > 0:
        confidence -= 0.15
        reasons.append(f"local model failed {previous_failures}x")

    # ── Clamp ──────────────────────────────────────────────────────────────
    confidence = max(0.0, min(1.0, confidence))

    # Escalate if confidence is low
    should_escalate = confidence < 0.35

    return ConfidenceScore(
        score=round(confidence, 3),
        should_escalate=should_escalate,
        reasons=reasons,
    )
