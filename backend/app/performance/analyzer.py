"""Performance Analyzer — comparison, regression, trends, recommendations, scores."""
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from app.core import settings
from app.performance.registry import PerformanceRegistry
from app.performance.budget import BudgetEvaluator, DEFAULT_BUDGETS


class PerformanceAnalyzer:
    def __init__(self, registry: PerformanceRegistry = None):
        self._registry = registry or PerformanceRegistry()

    def compare(self, current: Dict, previous: Optional[Dict] = None) -> Dict[str, Any]:
        prev = previous or self._registry.latest()
        if not prev or not prev.get("stages"):
            return {"status": "NO_BASELINE", "message": "No previous benchmark for comparison"}

        deltas = {}
        for stage, data in current.get("stages", {}).items():
            cur_dur = data.get("duration", 0)
            prev_dur = prev.get("stages", {}).get(stage, {}).get("duration", 0)
            if prev_dur > 0:
                deltas[stage] = round(cur_dur - prev_dur, 3)
        return {
            "schema_version": "1.0", "generated_at": datetime.now(timezone.utc).isoformat(),
            "current_duration": current.get("total_duration", 0),
            "previous_duration": prev.get("total_duration", 0),
            "delta": round(current.get("total_duration", 0) - prev.get("total_duration", 0), 2),
            "stage_deltas": deltas,
            "worsened": [s for s, d in deltas.items() if d > 0.5],
            "improved": [s for s, d in deltas.items() if d < -0.5],
        }

    def detect_regression(self, current: Dict, budget_eval: Dict) -> Dict[str, Any]:
        regressions = []
        for v in budget_eval.get("details", []):
            level = v["severity"]
            regressions.append({"stage": v["stage"], "level": level,
                                "budget": v["budget"], "actual": v["actual"]})
        return {
            "schema_version": "1.0", "generated_at": datetime.now(timezone.utc).isoformat(),
            "regression_count": len(regressions),
            "level": "CRITICAL" if any(r["level"] == "MAJOR" for r in regressions) else \
                     ("MINOR" if regressions else "NONE"),
            "regressions": regressions,
        }

    def analyze_trends(self) -> Dict[str, Any]:
        history = self._registry.history()
        if len(history) < 2:
            return {"status": "INSUFFICIENT_DATA", "runs": len(history)}

        durations = [h.get("total_duration", 0) for h in history]
        first = durations[0]; last = durations[-1]
        trend = "stable"
        if last < first * 0.9: trend = "improving"
        elif last > first * 1.1: trend = "regressing"

        return {
            "schema_version": "1.0", "generated_at": datetime.now(timezone.utc).isoformat(),
            "runs": len(history), "first_duration": first, "last_duration": last,
            "trend": trend, "min_duration": min(durations), "max_duration": max(durations),
        }

    def score(self, results: Dict, budget_eval: Dict) -> Dict[str, Any]:
        stages = results.get("stages", {})
        scores = {}
        for stage, budget in DEFAULT_BUDGETS.items():
            if stage == "total_startup": continue
            actual = stages.get(stage, {}).get("duration", 0)
            scores[stage] = min(100, max(0, int((1 - actual / max(budget, 0.001)) * 100))) if budget > 0 else 100
        overall = sum(scores.values()) // max(len(scores), 1)
        return {
            "schema_version": "1.0", "generated_at": datetime.now(timezone.utc).isoformat(),
            "stage_scores": scores, "overall": overall,
            "grade": "A" if overall >= 90 else ("B" if overall >= 75 else ("C" if overall >= 50 else "D")),
        }

    def recommend(self, results: Dict, budget_eval: Dict) -> List[str]:
        recs = []
        for v in budget_eval.get("details", []):
            recs.append(f"[{v['severity']}] Stage '{v['stage']}' exceeded budget: {v['actual']}s > {v['budget']}s")
        if results.get("profiler", {}).get("peak_ram_mb", 0) > 500:
            recs.append("[WARNING] High memory usage detected (>500MB peak)")
        if not recs:
            recs.append("[INFO] All performance budgets met. No recommendations.")
        return recs
