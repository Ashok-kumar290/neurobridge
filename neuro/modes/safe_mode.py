"""Safe Mode — the Core Mode orchestrator for NeuroBridge.

Handles the safe coding-memory workflow:
  1. Parse query
  2. Search repo index (FTS + symbols)
  3. Search memory for similar past queries
  4. Build context from top chunks
  5. Send to local model via Ollama
  6. Display answer with citations
  7. Store in session memory
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rich.console import Console

from neuro.config import get_config
from neuro.constants import MODEL_CODER, MODEL_ROUTER
from neuro.memory.session_memory import SessionMemory
from neuro.repo.search import RepoSearch, SearchResult
from neuro.runtime.ollama_client import OllamaClient, OllamaResponse, get_ollama_client

console = Console()


@dataclass
class AskAnswer:
    """Response from a SafeMode ask query."""

    content: str
    model: str
    sources: list[str] = field(default_factory=list)
    tokens_used: int = 0
    duration_ms: float = 0


class SafeMode:
    """Core Mode orchestrator — safe coding-memory system."""

    def __init__(self, repo_path: Path | None = None) -> None:
        self.repo_path = repo_path
        self.repo_name = repo_path.name if repo_path else None
        self.config = get_config()
        self.session_memory = SessionMemory()
        self.ollama = get_ollama_client()

    def _build_context(self, query: str, max_chunks: int = 8) -> tuple[str, list[str]]:
        """Build context from repo search and memory.

        Returns:
            (context_text, list_of_source_citations)
        """
        context_parts: list[str] = []
        sources: list[str] = []

        # Search repo index
        if self.repo_name:
            try:
                searcher = RepoSearch(self.repo_name)
                results = searcher.search(query, limit=max_chunks)

                if results:
                    context_parts.append("## Relevant code from repository:\n")
                    for r in results:
                        context_parts.append(
                            f"### {r.file_path} (lines {r.start_line}-{r.end_line})\n"
                            f"```{r.language}\n{r.content}\n```\n"
                        )
                        source = f"{r.file_path}:{r.start_line}-{r.end_line}"
                        if source not in sources:
                            sources.append(source)

                # Also check symbols
                symbol_results = searcher.symbol_search(query, limit=5)
                if symbol_results:
                    context_parts.append("\n## Related symbols:\n")
                    for sym in symbol_results:
                        context_parts.append(
                            f"- `{sym['kind']}` **{sym['name']}** "
                            f"in `{sym['file_path']}:{sym['start_line']}`"
                        )
                        if sym.get("signature"):
                            context_parts.append(f"  `{sym['signature']}`")
                        context_parts.append("")

            except FileNotFoundError:
                context_parts.append(
                    "*No repo index found. Run `neuro index <path>` first.*\n"
                )

        # Search session memory for similar past queries
        memory_hits = self.session_memory.search(query, limit=3)
        if memory_hits:
            context_parts.append("\n## Previous related sessions:\n")
            for hit in memory_hits:
                context_parts.append(
                    f"- **Q:** {hit.get('summary', 'N/A')}\n"
                    f"  **A (summary):** {(hit.get('answer', '') or '')[:200]}...\n"
                )

        return "\n".join(context_parts), sources

    def _select_model(self, query: str, context_length: int) -> tuple[str, float, int]:
        """Select which local model to use.

        Returns: (model_name, temperature, context_window)
        """
        # Simple heuristic: use 3B for short/simple queries, 7B for complex
        query_lower = query.lower()

        complex_indicators = [
            "fix", "debug", "refactor", "implement", "write", "create",
            "architecture", "design", "explain how", "why does",
        ]

        is_complex = any(ind in query_lower for ind in complex_indicators)

        if is_complex and self.ollama.has_model(MODEL_CODER):
            return (
                self.config.coder.model,
                self.config.coder.temperature,
                self.config.coder.context,
            )

        return (
            self.config.router.model,
            self.config.router.temperature,
            self.config.router.context,
        )

    def ask(self, query: str, model_override: str | None = None, context_override: list[SearchResult] | None = None) -> AskAnswer:
        """Answer a question about the repository.

        Flow:
          1. Search repo index for relevant code
          2. Search memory for similar past queries
          3. Build context (limited to 4 chunks for CPU speed)
          4. Send to local model via streaming (avoids timeout)
          5. Store session
          6. Return answer with citations
        """
        start_time = time.time()

        # 1-2. Build context from repo + memory (4 chunks max for CPU)
        if context_override:
            context_parts = []
            sources = []
            for r in context_override:
                context_parts.append(f"File: {r.file_path}\n```{r.language}\n{r.content}\n```")
                sources.append(f"{r.file_path}:{r.start_line}")
            context = "\n\n".join(context_parts)
        else:
            context, sources = self._build_context(query, max_chunks=4)

        # 3. Select model
        if model_override:
            model_name = model_override
            temperature = 0.2
            ctx_window = 4096
        else:
            model_name, temperature, ctx_window = self._select_model(query, len(context))

        # 4. Check Ollama is running
        if not self.ollama.is_running():
            return AskAnswer(
                content="❌ Ollama is not running. Start with: sudo systemctl start ollama",
                model="none",
            )

        # Check model is available
        if not self.ollama.has_model(model_name):
            return AskAnswer(
                content=f"❌ Model '{model_name}' not found. Run: neuro models pull",
                model="none",
            )

        # 5. Build messages
        system_prompt = (
            "You are NeuroBridge, a local AI coding assistant. "
            "Answer questions about the repository using the provided context. "
            "Be concise and precise. Cite specific files and line numbers. "
            "If you don't know the answer from the context, say so."
        )

        user_message = query
        if context.strip():
            user_message = f"{context}\n\n---\n\n**Question:** {query}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        # 6. Call model using STREAMING to avoid timeout on CPU
        console.print(f"[dim]Thinking with {model_name}...[/dim]")
        full_content = []
        for chunk in self.ollama.chat_stream(
            model=model_name,
            messages=messages,
            temperature=temperature,
            context_length=ctx_window,
        ):
            full_content.append(chunk)

        answer_text = "".join(full_content)
        duration_ms = (time.time() - start_time) * 1000

        # Estimate tokens (streaming doesn't return exact counts)
        estimated_tokens = len(answer_text.split()) * 2  # rough estimate

        # 7. Store in session memory
        self.session_memory.store_session(
            query=query,
            answer=answer_text,
            repo=self.repo_name,
            model=model_name,
            tokens_used=estimated_tokens,
            sources=sources,
            duration_ms=duration_ms,
        )

        return AskAnswer(
            content=answer_text,
            model=model_name,
            sources=sources,
            tokens_used=estimated_tokens,
            duration_ms=duration_ms,
        )
