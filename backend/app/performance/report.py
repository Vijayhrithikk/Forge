"""Performance report generators."""
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any
from app.core import settings

from app.performance.budget import DEFAULT_BUDGETS

def generate_baseline(results: Dict, budget_eval: Dict) -> Dict[str, Any]:
    return {
        "schema_version": "1.0", "forge_version": settings.app_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": results, "budget_evaluation": budget_eval,
    }

def generate_scorecard(results: Dict, budget_eval: Dict) -> Dict[str, Any]:
    stages = results.get("stages", {})
    scores = {}
    for stage, budget in DEFAULT_BUDGETS.items():
        if stage == "total_startup": continue
        actual = stages.get(stage, {}).get("duration", 0)
        scores[stage] = min(100, max(0, int((1 - actual / max(budget, 0.001)) * 100))) if budget > 0 else 100
    overall = sum(scores.values()) // max(len(scores), 1)
    return {
        "schema_version": "1.0", "forge_version": settings.app_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stage_scores": scores, "overall": overall,
        "grade": "A" if overall >= 90 else ("B" if overall >= 75 else ("C" if overall >= 50 else "D")),
    }

def generate_certificate(results: Dict, scorecard: Dict, budget_eval: Dict) -> Dict[str, Any]:
    score = scorecard.get("overall", 0)
    budget_pass = budget_eval.get("status") == "PASS"
    if score >= 80 and budget_pass:
        status = "PERFORMANCE_CERTIFIED"
    elif score >= 50:
        status = "PERFORMANCE_ACCEPTABLE"
    else:
        status = "PERFORMANCE_NOT_READY"
    return {
        "schema_version": "1.0", "forge_version": settings.app_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall_status": status, "performance_score": score,
        "budget_compliance": budget_eval.get("status"),
        "stages_measured": results.get("executed", 0),
    }
