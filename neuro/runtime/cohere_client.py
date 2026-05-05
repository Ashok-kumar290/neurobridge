"""Cohere client — planner, embeddings, reranking, and judge.

Roles in NeuroBridge:
  - Command A: long-context expert, architecture planner
  - Embed v4: memory and repo embeddings
  - Rerank v4: improve retrieval relevance
  - Judge: evaluate traces for trainability
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from rich.console import Console

console = Console()


@dataclass
class CohereResponse:
    """Response from Cohere API."""

    content: str
    model: str
    tokens_used: int = 0
    success: bool = True


class CohereClient:
    """Client for Cohere API services."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("COHERE_API_KEY") or os.environ.get("CO_API_KEY")
        self._client = None

    @property
    def available(self) -> bool:
        """Check if Cohere is configured."""
        return self.api_key is not None

    def _get_client(self) -> Any:
        """Lazy-load the Cohere client."""
        if self._client is None:
            try:
                import cohere
                self._client = cohere.ClientV2(api_key=self.api_key)
            except ImportError:
                raise RuntimeError(
                    "Cohere SDK not installed. Run: pip install cohere"
                )
        return self._client

    # ── Chat / Planning ────────────────────────────────────────────────────

    def chat(
        self,
        message: str,
        model: str | None = None,
        system: str | None = None,
        temperature: float = 0.3,
    ) -> CohereResponse:
        """Send a chat message to Cohere Command."""
        if not self.available:
            return CohereResponse(
                content="Cohere API key not configured. Set COHERE_API_KEY.",
                model="none",
                success=False,
            )

        from neuro.constants import MODEL_COHERE_PLANNER
        model = model or MODEL_COHERE_PLANNER

        try:
            client = self._get_client()
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": message})

            response = client.chat(
                model=model,
                messages=messages,
                temperature=temperature,
            )

            content = response.message.content[0].text if response.message.content else ""

            return CohereResponse(
                content=content,
                model=model,
                tokens_used=(
                    getattr(response.usage, "tokens", {}).get("input_tokens", 0)
                    + getattr(response.usage, "tokens", {}).get("output_tokens", 0)
                ) if hasattr(response, "usage") else 0,
            )
        except Exception as e:
            return CohereResponse(
                content=f"Cohere error: {e}",
                model=model,
                success=False,
            )

    def plan(self, task: str, context: str = "") -> CohereResponse:
        """Use Cohere as an architecture planner."""
        system = (
            "You are a senior software architect. Analyze the task and context, "
            "then provide a clear plan with specific files to modify, "
            "approach, and potential pitfalls."
        )
        message = task
        if context:
            message = f"{context}\n\n---\n\nTask: {task}"

        return self.chat(message, system=system)

    def judge(self, trace: str) -> CohereResponse:
        """Use Cohere as a trainability judge.

        Evaluates whether a coding trace is suitable for training data.
        """
        system = (
            "You are a data quality judge for an AI training pipeline. "
            "Evaluate if this coding trace is suitable for training. "
            "Check for: correctness, relevance, no secrets/PII, "
            "no harmful commands, minimal/clean code. "
            "Reply with JSON: {\"trainable\": true/false, \"reason\": \"...\", \"confidence\": 0.0-1.0}"
        )
        return self.chat(trace, system=system)

    # ── Reranking ──────────────────────────────────────────────────────────

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int = 10,
        model: str | None = None,
    ) -> list[dict[str, Any]]:
        """Rerank documents by relevance to query.

        Returns list of {index, relevance_score, document} sorted by score.
        """
        if not self.available:
            # Fallback: return documents in original order
            return [{"index": i, "relevance_score": 1.0, "document": d}
                    for i, d in enumerate(documents[:top_n])]

        from neuro.constants import MODEL_COHERE_RERANK
        model = model or MODEL_COHERE_RERANK

        try:
            client = self._get_client()
            response = client.rerank(
                model=model,
                query=query,
                documents=documents,
                top_n=min(top_n, len(documents)),
            )

            return [
                {
                    "index": r.index,
                    "relevance_score": r.relevance_score,
                    "document": documents[r.index],
                }
                for r in response.results
            ]
        except Exception as e:
            console.print(f"[yellow]Cohere rerank failed: {e}[/yellow]")
            return [{"index": i, "relevance_score": 1.0, "document": d}
                    for i, d in enumerate(documents[:top_n])]

    # ── Embeddings ─────────────────────────────────────────────────────────

    def embed(
        self,
        texts: list[str],
        model: str | None = None,
        input_type: str = "search_document",
    ) -> list[list[float]]:
        """Generate embeddings for texts.

        input_type: "search_document" for indexing, "search_query" for searching
        """
        if not self.available:
            raise RuntimeError("Cohere API key not configured.")

        from neuro.constants import MODEL_COHERE_EMBED
        model = model or MODEL_COHERE_EMBED

        client = self._get_client()
        response = client.embed(
            model=model,
            texts=texts,
            input_type=input_type,
            embedding_types=["float"],
        )

        return response.embeddings.float_


# ── Singleton ──────────────────────────────────────────────────────────────────

_client: CohereClient | None = None


def get_cohere_client() -> CohereClient:
    """Get or create the global Cohere client."""
    global _client
    if _client is None:
        _client = CohereClient()
    return _client
