"""Ollama HTTP client for local model inference.

Handles chat completions, model management, and health checks
against the Ollama REST API running on localhost.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Generator, Optional

import httpx
from rich.console import Console

from neuro.constants import (
    OLLAMA_API_CHAT,
    OLLAMA_API_GENERATE,
    OLLAMA_API_PULL,
    OLLAMA_API_SHOW,
    OLLAMA_API_TAGS,
    OLLAMA_BASE_URL,
)

console = Console()


@dataclass
class OllamaResponse:
    """Response from an Ollama API call."""

    content: str
    model: str
    total_duration_ns: int = 0
    eval_count: int = 0
    prompt_eval_count: int = 0
    done: bool = True

    @property
    def total_duration_ms(self) -> float:
        return self.total_duration_ns / 1_000_000

    @property
    def tokens_per_second(self) -> float:
        if self.total_duration_ns == 0:
            return 0.0
        return self.eval_count / (self.total_duration_ns / 1_000_000_000)


@dataclass
class OllamaClient:
    """Client for the Ollama REST API."""

    base_url: str = OLLAMA_BASE_URL
    timeout: float = 600.0  # 10 min — generous for CPU inference on i5 with HDD I/O
    _client: httpx.Client = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout, connect=10.0),
        )

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    # ── Health ─────────────────────────────────────────────────────────────

    def is_running(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            resp = self._client.get("/", timeout=5.0)
            return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    # ── Model management ───────────────────────────────────────────────────

    def list_models(self) -> list[dict[str, Any]]:
        """List all locally available models."""
        resp = self._client.get(OLLAMA_API_TAGS)
        resp.raise_for_status()
        return resp.json().get("models", [])

    def has_model(self, model_name: str) -> bool:
        """Check if a specific model is available locally."""
        models = self.list_models()
        return any(m.get("name", "").startswith(model_name) for m in models)

    def show_model(self, model_name: str) -> dict[str, Any]:
        """Get model details."""
        resp = self._client.post(OLLAMA_API_SHOW, json={"name": model_name})
        resp.raise_for_status()
        return resp.json()

    def pull_model(self, model_name: str) -> Generator[dict[str, Any], None, None]:
        """Pull a model with streaming progress.

        Yields progress dicts: {"status": "...", "completed": N, "total": N}
        """
        with self._client.stream(
            "POST",
            OLLAMA_API_PULL,
            json={"name": model_name, "stream": True},
            timeout=httpx.Timeout(3600.0, connect=30.0),  # 1 hour for large models
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue

    # ── Inference ──────────────────────────────────────────────────────────

    def generate(
        self,
        model: str,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.2,
        context_length: int = 4096,
        stream: bool = False,
    ) -> OllamaResponse:
        """Generate a completion (non-chat, single-turn)."""
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": context_length,
            },
        }
        if system:
            payload["system"] = system

        resp = self._client.post(OLLAMA_API_GENERATE, json=payload)
        resp.raise_for_status()
        data = resp.json()

        return OllamaResponse(
            content=data.get("response", ""),
            model=data.get("model", model),
            total_duration_ns=data.get("total_duration", 0),
            eval_count=data.get("eval_count", 0),
            prompt_eval_count=data.get("prompt_eval_count", 0),
            done=data.get("done", True),
        )

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        context_length: int = 4096,
        stream: bool = False,
    ) -> OllamaResponse:
        """Chat completion (multi-turn)."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": context_length,
            },
        }

        resp = self._client.post(OLLAMA_API_CHAT, json=payload)
        resp.raise_for_status()
        data = resp.json()

        msg = data.get("message", {})
        return OllamaResponse(
            content=msg.get("content", ""),
            model=data.get("model", model),
            total_duration_ns=data.get("total_duration", 0),
            eval_count=data.get("eval_count", 0),
            prompt_eval_count=data.get("prompt_eval_count", 0),
            done=data.get("done", True),
        )

    def chat_stream(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        context_length: int = 4096,
    ) -> Generator[str, None, OllamaResponse | None]:
        """Streaming chat completion. Yields content chunks.

        Returns the final OllamaResponse metadata after stream completes.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_ctx": context_length,
            },
        }

        final_data: dict[str, Any] = {}
        with self._client.stream("POST", OLLAMA_API_CHAT, json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if data.get("done"):
                    final_data = data
                    break

                msg = data.get("message", {})
                chunk = msg.get("content", "")
                if chunk:
                    yield chunk

        if final_data:
            return OllamaResponse(
                content="",  # already streamed
                model=final_data.get("model", model),
                total_duration_ns=final_data.get("total_duration", 0),
                eval_count=final_data.get("eval_count", 0),
                prompt_eval_count=final_data.get("prompt_eval_count", 0),
                done=True,
            )
        return None


# ── Convenience singleton ──────────────────────────────────────────────────────

_client: Optional[OllamaClient] = None


def get_ollama_client() -> OllamaClient:
    """Get or create the global Ollama client."""
    global _client
    if _client is None:
        _client = OllamaClient()
    return _client
