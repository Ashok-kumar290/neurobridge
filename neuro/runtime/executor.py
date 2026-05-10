"""Executor — connects the Router to the Adaptive Mind for inference.

This is the bridge between routing decisions and actual model execution.
When a task is routed to a local model, the Adaptive Mind augments
the prompt with relevant past experiences before generating.

Flow:
    User query → Router (decide) → Executor (execute) → Response
                                        ↓
                                  AdaptiveMind
                                  (recall → augment → generate → learn)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from rich.console import Console

from neuro.router.router import Router, RoutingDecision
from neuro.learning.adaptive_mind import AdaptiveMind, MindResponse

console = Console()


@dataclass
class ExecutionResult:
    """Full result of a routed and executed task."""
    content: str
    model: str
    target: str                          # "local:3b", "local:7b", "expert:codex", etc.
    routing_reason: str
    experiences_used: int = 0
    augmented: bool = False
    factuality_score: float = 0.0
    quality_score: float = 0.0
    generation_time_ms: float = 0.0
    tokens_used: int = 0
    difficulty_score: float = 0.0
    confidence_score: float = 0.0


class Executor:
    """Executes routed tasks through the Adaptive Mind.
    
    Usage:
        executor = Executor()
        result = executor.run("How do I parse JSON in Python?")
        print(result.content)
    """
    
    def __init__(
        self,
        mind: AdaptiveMind | None = None,
        router: Router | None = None,
        auto_ingest: bool = True,
    ):
        self.router = router or Router()
        self.mind = mind or AdaptiveMind(
            model="super-qwen:3b",
            use_steering=True,
            auto_learn=True,
        )
        
        # Auto-ingest from replay buffer on startup
        if auto_ingest:
            count = self.mind.ingest_buffer()
            if count > 0:
                console.print(f"[dim]Ingested {count} experiences from replay buffer[/dim]")
    
    def run(
        self,
        query: str,
        context: str = "",
        file_count: int = 0,
        memory_hits: int = 0,
        error_trace: str | None = None,
        previous_failures: int = 0,
        search_score: float = 0.0,
        force_expert: str | None = None,
        system: str = "",
        temperature: float = 0.2,
    ) -> ExecutionResult:
        """Route and execute a task.
        
        For local models: uses the Adaptive Mind (experience-augmented).
        For expert models: returns routing info (user handles expert call).
        """
        start = time.time()
        
        # Step 1: Route
        decision = self.router.route(
            query=query,
            context=context,
            file_count=file_count,
            memory_hits=memory_hits,
            error_trace=error_trace,
            previous_failures=previous_failures,
            search_score=search_score,
            force_expert=force_expert,
        )
        
        # Step 2: Execute based on routing target
        if decision.target.startswith("local"):
            return self._execute_local(query, context, system, temperature, decision)
        elif decision.expert_required:
            return self._handle_expert(query, context, decision)
        else:
            return self._execute_local(query, context, system, temperature, decision)
    
    def _execute_local(
        self,
        query: str,
        context: str,
        system: str,
        temperature: float,
        decision: RoutingDecision,
    ) -> ExecutionResult:
        """Execute locally via the Adaptive Mind."""
        # Switch model based on routing decision
        self.mind.model = decision.model
        
        # Think (recall → augment → generate → score → learn)
        response = self.mind.think(
            query=query,
            context=context,
            system=system,
            temperature=temperature,
        )
        
        return ExecutionResult(
            content=response.content,
            model=response.model,
            target=decision.target,
            routing_reason=decision.reason,
            experiences_used=response.num_experiences_recalled,
            augmented=response.augmented,
            factuality_score=response.factuality_score,
            quality_score=response.quality_score,
            generation_time_ms=response.generation_time_ms,
            tokens_used=response.tokens_used,
            difficulty_score=decision.difficulty.score,
            confidence_score=decision.confidence.score,
        )
    
    def _handle_expert(
        self,
        query: str,
        context: str,
        decision: RoutingDecision,
    ) -> ExecutionResult:
        """Handle expert routing — return info for the user to act on."""
        msg = (
            f"Task routed to **{decision.preferred_expert}** "
            f"(difficulty={decision.difficulty.score:.1f}, "
            f"confidence={decision.confidence.score:.1f}).\n\n"
            f"Reason: {decision.reason}\n\n"
            f"Run this through the interceptor to capture the response:\n"
            f"  python -m neuro.training.interceptor --tool {decision.preferred_expert}"
        )
        return ExecutionResult(
            content=msg,
            model=decision.model,
            target=decision.target,
            routing_reason=decision.reason,
            difficulty_score=decision.difficulty.score,
            confidence_score=decision.confidence.score,
        )
    
    def feedback(self, result: ExecutionResult, score: float) -> None:
        """Provide feedback on an execution result."""
        # Create a fake MindResponse to pass to the mind's feedback
        fake = MindResponse(
            content=result.content,
            model=result.model,
            experience_ids=[],  # TODO: track experience IDs through execution
        )
        self.mind.feedback(fake, score)
    
    def status(self) -> dict:
        """Get executor status."""
        mind_status = self.mind.status()
        return {
            "executor": "active",
            **mind_status,
        }
