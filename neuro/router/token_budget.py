"""Token budget estimator — predicts token usage and cost before calling models.

Helps NeuroBridge decide whether to use local models (free) or
expensive expert models (paid), and how much context to include.
"""

from __future__ import annotations

from dataclasses import dataclass


# ── Approximate pricing (USD per 1M tokens, as of 2026) ───────────────────────
# These are rough estimates — update as pricing changes

MODEL_PRICING = {
    # Local models (free) — Super-Qwen custom
    "super-qwen:3b": {"input": 0.0, "output": 0.0},
    "super-qwen:7b": {"input": 0.0, "output": 0.0},
    # Local models (free) — base Qwen
    "qwen2.5-coder:3b": {"input": 0.0, "output": 0.0},
    "qwen2.5-coder:7b": {"input": 0.0, "output": 0.0},
    # Cohere
    "command-a-03-2025": {"input": 2.5, "output": 10.0},
    "command-r7b-12-2024": {"input": 0.375, "output": 1.5},
    "embed-v4.0": {"input": 0.1, "output": 0.0},
    "rerank-v4.0-fast": {"input": 0.0, "output": 0.0},  # per-search pricing
    # Claude (estimated)
    "claude-code": {"input": 3.0, "output": 15.0},
    # Codex (estimated)
    "codex-cli": {"input": 2.0, "output": 8.0},
}


@dataclass
class TokenBudget:
    """Estimated token budget for a task."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    model: str
    breakdown: dict[str, int]


def estimate_tokens(text: str) -> int:
    """Rough token count estimate (1 token ≈ 4 chars for code)."""
    return max(1, len(text) // 4)


def estimate_budget(
    context: str,
    query: str,
    model: str,
    max_output_tokens: int = 2048,
) -> TokenBudget:
    """Estimate token budget and cost for a model call.

    Args:
        context: The context/prompt text to send
        query: The user query
        model: Model identifier
        max_output_tokens: Expected max output length

    Returns:
        TokenBudget with cost estimate
    """
    system_tokens = 100  # system prompt overhead
    context_tokens = estimate_tokens(context)
    query_tokens = estimate_tokens(query)

    input_tokens = system_tokens + context_tokens + query_tokens
    output_tokens = max_output_tokens
    total_tokens = input_tokens + output_tokens

    # Calculate cost
    pricing = MODEL_PRICING.get(model, {"input": 0.0, "output": 0.0})
    cost = (
        (input_tokens / 1_000_000) * pricing["input"]
        + (output_tokens / 1_000_000) * pricing["output"]
    )

    return TokenBudget(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        estimated_cost_usd=round(cost, 6),
        model=model,
        breakdown={
            "system": system_tokens,
            "context": context_tokens,
            "query": query_tokens,
            "output_estimate": output_tokens,
        },
    )


def compare_budgets(
    context: str,
    query: str,
    models: list[str] | None = None,
) -> list[TokenBudget]:
    """Compare token budgets across multiple models.

    Returns list sorted by cost (cheapest first).
    """
    if models is None:
        models = list(MODEL_PRICING.keys())

    budgets = [estimate_budget(context, query, m) for m in models]
    budgets.sort(key=lambda b: b.estimated_cost_usd)
    return budgets
