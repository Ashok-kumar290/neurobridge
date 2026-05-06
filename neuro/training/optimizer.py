"""Plasticity Engine — self-optimization and hallucination suppression."""

import asyncio
from typing import Any
from rich.console import Console
from neuro.runtime.ollama_client import get_ollama_client

console = Console()

class ConsistencyTester:
    """Tests model consistency across multiple runs to detect hallucinations."""

    def __init__(self, model: str, iterations: int = 3):
        self.model = model
        self.iterations = iterations
        self.ollama = get_ollama_client()

    async def test(self, query: str, context: str) -> dict[str, Any]:
        """Runs the query multiple times and checks for consistency."""
        responses = []
        
        console.print(f"[dim]Running consistency check ({self.iterations} iterations)...[/dim]")
        
        for i in range(self.iterations):
            response = self.ollama.generate(
                model=self.model,
                prompt=f"Context: {context}\n\nQuestion: {query}",
                temperature=0.7, # Higher temperature to find variance
            )
            responses.append(response.content)

        # Basic similarity check (can be improved with embeddings)
        consistency_score = self._calculate_consistency(responses)
        
        return {
            "responses": responses,
            "consistency_score": consistency_score,
            "is_hallucinating": consistency_score < 0.7
        }

    def _calculate_consistency(self, responses: list[str]) -> float:
        """Calculates semantic consistency using cosine similarity of embeddings."""
        if len(responses) < 2:
            return 1.0
        
        import numpy as np
        
        # Get embeddings for all responses
        embeddings = []
        for resp in responses:
            emb = self.ollama.embeddings(model=self.model, prompt=resp)
            if emb:
                embeddings.append(emb)

        if not embeddings:
            return 0.0

        # Calculate pairwise cosine similarities
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                similarities.append(cosine_similarity(embeddings[i], embeddings[j]))

        return float(np.mean(similarities)) if similarities else 0.0

class PlasticityEngine:
    """Manages the 'stale weight' pruning and optimization cycle."""

    def __init__(self, model: str):
        self.model = model
        self.tester = ConsistencyTester(model)

    async def run_cycle(self, traces: list[dict]) -> list[dict]:
        """Processes a batch of traces and prunes those with low consistency."""
        clean_traces = []
        
        console.print(f"[bold cyan]Initiating Plasticity Cycle for {len(traces)} traces...[/bold cyan]")
        
        for trace in traces:
            task = trace.get("task", "")
            context = ""
            for step in trace.get("steps", []):
                if step.get("step_type") == "context":
                    context += step.get("data", {}).get("content", "") + "\n"
            
            result = await self.tester.test(task, context)
            
            if not result["is_hallucinating"]:
                clean_traces.append(trace)
                console.print(f"  [green]✓ Trace {trace.get('trace_id', '???')[:8]} preserved.[/green]")
            else:
                console.print(f"  [red]⚠ Trace {trace.get('trace_id', '???')[:8]} pruned (Hallucination detected).[/red]")
                
        return clean_traces
