"""Eval runner — orchestrates benchmark suites against local models.

Runs predefined eval cases against local 3B/7B models and produces
structured reports stored on HDD. These baselines are what you compare
against AFTER adapter training to prove improvement (or regression).

Usage:
    runner = EvalRunner()
    report = runner.run_suite("coding_tasks")
    runner.save_report(report)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np

from neuro.constants import HDD_EVALS


@dataclass
class EvalCase:
    """A single evaluation case."""

    case_id: str
    category: str          # "coding", "recall", "hallucination", "safety"
    prompt: str
    expected: str | None = None      # expected output (exact or substring)
    expected_contains: list[str] = field(default_factory=list)  # must contain these
    expected_not_contains: list[str] = field(default_factory=list)  # must NOT contain these
    max_tokens: int = 1024
    metadata: dict[str, Any] = field(default_factory=dict)
    semantic_check: bool = False  # If True, use embedding similarity
    expected_semantic_similarity: float = 0.8  # Required score (0.0 to 1.0)
    ideal_response: str | None = None  # Reference response for semantic check


@dataclass
class EvalResult:
    """Result of running a single eval case."""

    case_id: str
    category: str
    passed: bool
    model: str
    response: str
    duration_ms: float
    checks: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


@dataclass
class EvalReport:
    """Complete eval suite report."""

    suite_name: str
    model: str
    results: list[EvalResult] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0

    @property
    def avg_duration_ms(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.duration_ms for r in self.results) / len(self.results)

    def by_category(self) -> dict[str, dict[str, int]]:
        """Group results by category."""
        cats: dict[str, dict[str, int]] = {}
        for r in self.results:
            if r.category not in cats:
                cats[r.category] = {"passed": 0, "failed": 0, "total": 0}
            cats[r.category]["total"] += 1
            if r.passed:
                cats[r.category]["passed"] += 1
            else:
                cats[r.category]["failed"] += 1
        return cats

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_name": self.suite_name,
            "model": self.model,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": round(self.pass_rate, 4),
            "avg_duration_ms": round(self.avg_duration_ms, 1),
            "by_category": self.by_category(),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "results": [
                {
                    "case_id": r.case_id,
                    "category": r.category,
                    "passed": r.passed,
                    "model": r.model,
                    "duration_ms": round(r.duration_ms, 1),
                    "checks": r.checks,
                    "error": r.error,
                    # Don't store full responses in report (too large)
                    "response_length": len(r.response),
                }
                for r in self.results
            ],
        }


class EvalRunner:
    """Runs eval suites against local models."""

    def __init__(self) -> None:
        from neuro.runtime.ollama_client import get_ollama_client
        self.client = get_ollama_client()
        self.reports_dir = HDD_EVALS / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def run_case(self, case: EvalCase, model: str) -> EvalResult:
        """Run a single eval case."""
        start = time.time()
        checks: list[dict[str, Any]] = []
        error = None

        try:
            resp = self.client.generate(
                model=model,
                prompt=case.prompt,
                temperature=0.1,
            )
            response_text = resp.content
        except Exception as e:
            return EvalResult(
                case_id=case.case_id,
                category=case.category,
                passed=False,
                model=model,
                response="",
                duration_ms=(time.time() - start) * 1000,
                error=str(e),
            )

        duration_ms = (time.time() - start) * 1000
        passed = True

        # ── Check: expected exact ──────────────────────────────────────────
        if case.expected is not None:
            exact_match = case.expected.strip().lower() in response_text.strip().lower()
            checks.append({"check": "exact_match", "passed": exact_match})
            if not exact_match:
                passed = False

        # ── Check: contains ────────────────────────────────────────────────
        for expected in case.expected_contains:
            found = expected.lower() in response_text.lower()
            checks.append({"check": f"contains:{expected[:30]}", "passed": found})
            if not found:
                passed = False

        # ── Check: not contains ────────────────────────────────────────────
        for forbidden in case.expected_not_contains:
            absent = forbidden.lower() not in response_text.lower()
            checks.append({"check": f"not_contains:{forbidden[:30]}", "passed": absent})
            if not absent:
                passed = False
        # ── Check: semantic similarity ─────────────────────────────────────
        if case.semantic_check and case.ideal_response:
            from neuro.training.optimizer import ConsistencyTester
            tester = ConsistencyTester(model=model)
            # Use calculate_consistency logic or similar
            # Since we have Ollama embeddings now:
            v_ref = np.array(self.client.embeddings(model, case.ideal_response))
            v_resp = np.array(self.client.embeddings(model, response_text))
            
            if v_ref.any() and v_resp.any():
                similarity = np.dot(v_ref, v_resp) / (np.linalg.norm(v_ref) * np.linalg.norm(v_resp))
                passed_semantic = float(similarity) >= case.expected_semantic_similarity
                checks.append({
                    "check": "semantic_similarity", 
                    "passed": passed_semantic, 
                    "score": round(float(similarity), 4)
                })
                if not passed_semantic:
                    passed = False
            else:
                checks.append({"check": "semantic_similarity", "passed": False, "error": "embedding failure"})
                passed = False
        # ── Check: non-empty ───────────────────────────────────────────────
        if not response_text.strip():
            checks.append({"check": "non_empty", "passed": False})
            passed = False
        else:
            checks.append({"check": "non_empty", "passed": True})

        return EvalResult(
            case_id=case.case_id,
            category=case.category,
            passed=passed,
            model=model,
            response=response_text,
            duration_ms=duration_ms,
            checks=checks,
            error=error,
        )

    def run_suite(
        self,
        suite_name: str,
        cases: list[EvalCase],
        model: str | None = None,
        on_result: Any = None,
    ) -> EvalReport:
        """Run a full eval suite.

        Args:
            suite_name: Name of the suite
            cases: List of eval cases to run
            model: Model to test (defaults to 3B router)
            on_result: Optional callback(result) for progress reporting
        """
        from neuro.constants import MODEL_ROUTER
        model = model or MODEL_ROUTER

        report = EvalReport(suite_name=suite_name, model=model)

        for case in cases:
            result = self.run_case(case, model)
            report.results.append(result)

            if on_result:
                on_result(result)

        report.finished_at = time.time()
        return report

    def save_report(self, report: EvalReport) -> Path:
        """Save an eval report to HDD."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{report.suite_name}_{report.model.replace(':', '_')}_{timestamp}.json"
        path = self.reports_dir / filename

        with open(path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        return path

    def load_reports(self, suite_name: str | None = None) -> list[dict[str, Any]]:
        """Load saved eval reports."""
        reports = []
        for path in sorted(self.reports_dir.glob("*.json")):
            if suite_name and suite_name not in path.name:
                continue
            try:
                with open(path) as f:
                    reports.append(json.load(f))
            except Exception:
                continue
        return reports

    def compare_reports(
        self,
        baseline: dict[str, Any],
        current: dict[str, Any],
    ) -> dict[str, Any]:
        """Compare two eval reports to show improvement/regression."""
        comparison = {
            "baseline_model": baseline.get("model"),
            "current_model": current.get("model"),
            "baseline_pass_rate": baseline.get("pass_rate", 0),
            "current_pass_rate": current.get("pass_rate", 0),
            "delta": round(
                current.get("pass_rate", 0) - baseline.get("pass_rate", 0), 4
            ),
            "improved": current.get("pass_rate", 0) > baseline.get("pass_rate", 0),
            "regression": current.get("pass_rate", 0) < baseline.get("pass_rate", 0),
            "by_category": {},
        }

        # Compare by category
        for cat in set(list(baseline.get("by_category", {}).keys()) +
                       list(current.get("by_category", {}).keys())):
            b = baseline.get("by_category", {}).get(cat, {"passed": 0, "total": 0})
            c = current.get("by_category", {}).get(cat, {"passed": 0, "total": 0})
            b_rate = b["passed"] / b["total"] if b["total"] > 0 else 0
            c_rate = c["passed"] / c["total"] if c["total"] > 0 else 0
            comparison["by_category"][cat] = {
                "baseline": round(b_rate, 4),
                "current": round(c_rate, 4),
                "delta": round(c_rate - b_rate, 4),
            }

        return comparison
