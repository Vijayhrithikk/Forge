"""Optimization Framework — evidence-based, reversible optimizations."""
import json, time, copy
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Callable
from app.core import settings, get_logger
from app.performance.benchmark import BenchmarkEngine
from app.performance.analyzer import PerformanceAnalyzer
from app.performance.budget import BudgetEvaluator

logger = get_logger("app.performance.optimization")

OPTIMIZATION_REGISTRY_PATH = Path("data/performance/optimization_registry.json")


class OptimizationSandbox:
    """Isolated optimization testing — never modifies production Runtime."""
    def __init__(self):
        self._baseline: Dict = {}
        self._optimized: Dict = {}

    def run(self, candidate_name: str, apply_fn: Callable, revert_fn: Callable,
            callbacks: Dict[str, Callable]) -> Dict[str, Any]:
        engine = BenchmarkEngine()
        self._baseline = engine.run_all(callbacks)
        apply_fn()
        self._optimized = engine.run_all(callbacks)
        revert_fn()
        delta = self._optimized.get("total_duration", 0) - self._baseline.get("total_duration", 0)
        improved = delta < 0
        return {
            "candidate": candidate_name,
            "baseline_duration": self._baseline.get("total_duration"),
            "optimized_duration": self._optimized.get("total_duration"),
            "delta": round(delta, 3),
            "improved": improved,
            "improvement_pct": round(abs(delta) / max(self._baseline.get("total_duration", 0.001), 0.001) * 100, 1),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


class OptimizationRegistry:
    """Append-only optimization history."""
    def __init__(self, path: Path = OPTIMIZATION_REGISTRY_PATH):
        self._path = path; self._path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, result: Dict, accepted: bool, reason: str = "") -> Dict:
        entry = {**result, "accepted": accepted, "reason": reason,
                 "forge_version": settings.app_version, "recorded_at": datetime.now(timezone.utc).isoformat()}
        entries = self._load(); entries.append(entry)
        with open(self._path, "w") as f: json.dump({"entries": entries, "schema_version": "1.0"}, f, indent=2)
        return entry

    def _load(self) -> list:
        if self._path.exists():
            with open(self._path) as f: return json.load(f).get("entries", [])
        return []

    def list_accepted(self) -> List[Dict]: return [e for e in self._load() if e.get("accepted")]
    def list_rejected(self) -> List[Dict]: return [e for e in self._load() if not e.get("accepted")]


class OptimizationEngine:
    """Coordinates optimization candidates, sandbox testing, and acceptance."""
    def __init__(self):
        self._sandbox = OptimizationSandbox()
        self._registry = OptimizationRegistry()
        self._analyzer = PerformanceAnalyzer()
        self._budget_eval = BudgetEvaluator()

    def evaluate_candidate(self, name: str, apply_fn: Callable, revert_fn: Callable,
                           callbacks: Dict[str, Callable]) -> Dict[str, Any]:
        result = self._sandbox.run(name, apply_fn, revert_fn, callbacks)
        budget_eval = self._budget_eval.evaluate(self._sandbox._optimized)
        regression_free = len(budget_eval.get("details", [])) == 0
        accepted = result["improved"] and regression_free
        reason = ""
        if not result["improved"]: reason = "No performance improvement"
        elif not regression_free: reason = "Regression detected in optimized state"
        else: reason = "Performance improved with no regressions"
        self._registry.record(result, accepted, reason)
        logger.info("optimization_evaluated", candidate=name, accepted=accepted, improvement=result.get("improvement_pct", 0))
        return {**result, "accepted": accepted, "reason": reason}
