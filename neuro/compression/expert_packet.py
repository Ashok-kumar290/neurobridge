"""Expert packet generator — the core token-saving architecture.

Turns an 80K-token messy repo dump into a 4-8K precise expert packet.
This is what gets sent to Claude/Codex/Cohere instead of the raw repo.

Expert packet format:
{
    "task": "Fix failing refresh token test",
    "repo_summary": "FastAPI backend using JWT auth and pytest.",
    "failure": { "command": "...", "summary": "..." },
    "relevant_files": [ { "path": "...", "reason": "...", "important_lines": "..." } ],
    "memory_hits": [ { "summary": "...", "confidence": 0.82 } ],
    "constraints": [ "Return minimal patch", ... ]
}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from neuro.constants import MAX_EXPERT_PACKET_TOKENS
from neuro.router.token_budget import estimate_tokens


@dataclass
class ExpertPacket:
    """Compressed context packet for expert models."""

    task: str
    repo_summary: str
    relevant_files: list[dict[str, str]]
    memory_hits: list[dict[str, Any]] = field(default_factory=list)
    failure: dict[str, str] | None = None
    constraints: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        d: dict[str, Any] = {
            "task": self.task,
            "repo_summary": self.repo_summary,
            "relevant_files": self.relevant_files,
        }
        if self.failure:
            d["failure"] = self.failure
        if self.memory_hits:
            d["memory_hits"] = self.memory_hits
        if self.constraints:
            d["constraints"] = self.constraints
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_prompt(self) -> str:
        """Convert to a prompt string for expert models."""
        parts = [f"## Task\n{self.task}\n"]

        parts.append(f"## Repository\n{self.repo_summary}\n")

        if self.failure:
            parts.append("## Failure\n")
            if self.failure.get("command"):
                parts.append(f"Command: `{self.failure['command']}`\n")
            if self.failure.get("summary"):
                parts.append(f"Error: {self.failure['summary']}\n")

        if self.relevant_files:
            parts.append("## Relevant Files\n")
            for f in self.relevant_files:
                parts.append(f"### {f['path']}")
                if f.get("reason"):
                    parts.append(f"Reason: {f['reason']}")
                if f.get("content"):
                    lang = f.get("language", "")
                    parts.append(f"```{lang}\n{f['content']}\n```")
                elif f.get("important_lines"):
                    parts.append(f"Key lines: {f['important_lines']}")
                parts.append("")

        if self.memory_hits:
            parts.append("## Similar Past Solutions\n")
            for hit in self.memory_hits:
                conf = hit.get("confidence", "?")
                parts.append(f"- {hit.get('summary', 'N/A')} (confidence: {conf})")
            parts.append("")

        if self.constraints:
            parts.append("## Constraints\n")
            for c in self.constraints:
                parts.append(f"- {c}")

        return "\n".join(parts)

    @property
    def estimated_tokens(self) -> int:
        """Estimate token count of this packet."""
        return estimate_tokens(self.to_prompt())


class ExpertPacketBuilder:
    """Builds expert packets from search results and memory."""

    def __init__(
        self,
        max_tokens: int = MAX_EXPERT_PACKET_TOKENS,
    ) -> None:
        self.max_tokens = max_tokens

    def build(
        self,
        task: str,
        repo_name: str,
        repo_description: str = "",
        search_results: list[Any] | None = None,
        memory_hits: list[dict] | None = None,
        error_trace: str | None = None,
        error_command: str | None = None,
        constraints: list[str] | None = None,
    ) -> ExpertPacket:
        """Build an expert packet from available context.

        Automatically compresses to fit within max_tokens.
        """
        # ── Build repo summary ─────────────────────────────────────────────
        repo_summary = repo_description or f"Repository: {repo_name}"

        # ── Build relevant files ───────────────────────────────────────────
        relevant_files: list[dict[str, str]] = []
        token_budget = self.max_tokens - estimate_tokens(task) - estimate_tokens(repo_summary) - 500

        if search_results:
            for result in search_results:
                file_entry: dict[str, str] = {
                    "path": result.file_path,
                    "language": result.language,
                }

                # Trim content to fit budget
                content = result.content
                content_tokens = estimate_tokens(content)

                if content_tokens <= token_budget:
                    file_entry["content"] = content
                    file_entry["important_lines"] = f"{result.start_line}-{result.end_line}"
                    token_budget -= content_tokens
                elif token_budget > 100:
                    # Truncate to fit
                    max_chars = token_budget * 4
                    file_entry["content"] = content[:max_chars] + "\n... (truncated)"
                    file_entry["important_lines"] = f"{result.start_line}-{result.end_line}"
                    token_budget = 0
                else:
                    # Just reference the file
                    file_entry["important_lines"] = f"{result.start_line}-{result.end_line}"
                    file_entry["reason"] = f"{result.chunk_type} in {result.language}"

                relevant_files.append(file_entry)

                if token_budget <= 0:
                    break

        # ── Build failure info ─────────────────────────────────────────────
        failure = None
        if error_trace or error_command:
            failure = {}
            if error_command:
                failure["command"] = error_command
            if error_trace:
                # Truncate long traces
                if len(error_trace) > 2000:
                    failure["summary"] = error_trace[:1000] + "\n...\n" + error_trace[-800:]
                else:
                    failure["summary"] = error_trace

        # ── Build memory hits ──────────────────────────────────────────────
        packet_memory: list[dict[str, Any]] = []
        if memory_hits:
            for hit in memory_hits[:3]:  # max 3 memory hits
                packet_memory.append({
                    "summary": hit.get("summary", hit.get("query", "N/A"))[:200],
                    "confidence": round(hit.get("score", 0.5), 2),
                })

        # ── Default constraints ────────────────────────────────────────────
        default_constraints = [
            "Return minimal, testable patch",
            "Do not change public API unless necessary",
            "Include test commands to verify the fix",
        ]
        all_constraints = (constraints or []) + default_constraints

        return ExpertPacket(
            task=task,
            repo_summary=repo_summary,
            relevant_files=relevant_files,
            memory_hits=packet_memory,
            failure=failure,
            constraints=all_constraints,
            metadata={
                "repo": repo_name,
                "files_included": len(relevant_files),
                "max_tokens": self.max_tokens,
            },
        )
