"""Difficulty estimator — classifies task complexity.

Uses heuristics + optional 3B model assessment to estimate
how hard a coding task is, which drives model selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class Difficulty(str, Enum):
    """Task difficulty levels."""

    TRIVIAL = "trivial"      # simple lookup, file find, explain variable
    EASY = "easy"            # single-file fix, simple refactor, add comment
    MEDIUM = "medium"        # multi-file awareness, debugging, test writing
    HARD = "hard"            # architecture, complex debugging, multi-file refactor
    EXPERT = "expert"        # requires frontier model capabilities


@dataclass
class DifficultyAssessment:
    """Result of difficulty estimation."""

    difficulty: Difficulty
    score: float             # 0.0 (trivial) to 1.0 (expert)
    reasons: list[str]
    suggested_model: str     # model identifier
    estimated_tokens: int    # rough estimate of tokens needed


# ── Keyword-based signals ──────────────────────────────────────────────────────

_TRIVIAL_SIGNALS = [
    "where is", "what is", "find", "list", "show", "which file",
    "how many", "count", "path to",
]

_EASY_SIGNALS = [
    "rename", "add comment", "format", "lint", "type hint",
    "simple fix", "typo", "import", "add logging",
]

_MEDIUM_SIGNALS = [
    "fix", "debug", "test", "refactor", "update", "change",
    "modify", "improve", "optimize", "handle error",
]

_HARD_SIGNALS = [
    "architecture", "design", "migrate", "rewrite", "complex",
    "multi-file", "async", "concurrency", "security", "performance",
    "integrate", "api design",
]

_EXPERT_SIGNALS = [
    "review entire", "redesign", "from scratch", "distributed",
    "system design", "production deploy", "critical bug",
]


def _count_signals(query: str, signals: list[str]) -> int:
    """Count how many signal phrases appear in the query."""
    query_lower = query.lower()
    return sum(1 for s in signals if s in query_lower)


def estimate_difficulty(
    query: str,
    file_count: int = 0,
    error_trace: str | None = None,
    memory_hits: int = 0,
) -> DifficultyAssessment:
    """Estimate task difficulty from query text and context signals.

    Args:
        query: The user's task description
        file_count: Number of relevant files found by search
        error_trace: Optional error/traceback text
        memory_hits: Number of similar past solutions in memory

    Returns:
        DifficultyAssessment with difficulty level, score, and model suggestion
    """
    from neuro.constants import MODEL_CODER, MODEL_ROUTER

    reasons: list[str] = []
    score = 0.5  # start at medium

    # ── Query signal analysis ──────────────────────────────────────────────
    trivial_hits = _count_signals(query, _TRIVIAL_SIGNALS)
    easy_hits = _count_signals(query, _EASY_SIGNALS)
    medium_hits = _count_signals(query, _MEDIUM_SIGNALS)
    hard_hits = _count_signals(query, _HARD_SIGNALS)
    expert_hits = _count_signals(query, _EXPERT_SIGNALS)

    if expert_hits > 0:
        score += 0.3 * expert_hits
        reasons.append(f"expert-level keywords detected ({expert_hits})")
    if hard_hits > 0:
        score += 0.15 * hard_hits
        reasons.append(f"hard keywords detected ({hard_hits})")
    if medium_hits > 0:
        score += 0.05 * medium_hits
    if easy_hits > 0:
        score -= 0.15 * easy_hits
        reasons.append(f"easy keywords detected ({easy_hits})")
    if trivial_hits > 0:
        score -= 0.25 * trivial_hits
        reasons.append(f"trivial keywords detected ({trivial_hits})")

    # ── Query length ───────────────────────────────────────────────────────
    words = len(query.split())
    if words > 50:
        score += 0.1
        reasons.append("long query (complex request)")
    elif words < 10:
        score -= 0.1
        reasons.append("short query (likely simple)")

    # ── File count ─────────────────────────────────────────────────────────
    if file_count > 5:
        score += 0.15
        reasons.append(f"many relevant files ({file_count})")
    elif file_count <= 1:
        score -= 0.1
        reasons.append("single file scope")

    # ── Error trace ────────────────────────────────────────────────────────
    if error_trace:
        trace_lines = error_trace.count("\n")
        if trace_lines > 20:
            score += 0.15
            reasons.append("deep error trace")
        else:
            score += 0.05
            reasons.append("error trace present")

    # ── Memory hits (similar solved problems reduce difficulty) ─────────────
    if memory_hits > 2:
        score -= 0.2
        reasons.append(f"similar problems solved before ({memory_hits} hits)")
    elif memory_hits > 0:
        score -= 0.1
        reasons.append(f"some memory hits ({memory_hits})")

    # ── Clamp and classify ─────────────────────────────────────────────────
    score = max(0.0, min(1.0, score))

    if score < 0.15:
        difficulty = Difficulty.TRIVIAL
    elif score < 0.35:
        difficulty = Difficulty.EASY
    elif score < 0.55:
        difficulty = Difficulty.MEDIUM
    elif score < 0.75:
        difficulty = Difficulty.HARD
    else:
        difficulty = Difficulty.EXPERT

    # ── Model suggestion ───────────────────────────────────────────────────
    if difficulty in (Difficulty.TRIVIAL, Difficulty.EASY):
        suggested_model = MODEL_ROUTER  # 3B handles simple tasks
    elif difficulty == Difficulty.MEDIUM:
        suggested_model = MODEL_CODER   # 7B for medium tasks
    elif difficulty == Difficulty.HARD:
        suggested_model = MODEL_CODER   # 7B first, may escalate
    else:
        suggested_model = "expert"      # needs Claude/Codex/Cohere

    # ── Token estimate ─────────────────────────────────────────────────────
    if difficulty == Difficulty.TRIVIAL:
        estimated_tokens = 500
    elif difficulty == Difficulty.EASY:
        estimated_tokens = 1500
    elif difficulty == Difficulty.MEDIUM:
        estimated_tokens = 3000
    elif difficulty == Difficulty.HARD:
        estimated_tokens = 6000
    else:
        estimated_tokens = 8000

    return DifficultyAssessment(
        difficulty=difficulty,
        score=round(score, 3),
        reasons=reasons,
        suggested_model=suggested_model,
        estimated_tokens=estimated_tokens,
    )
