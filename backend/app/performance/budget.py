"""Performance Budgets — configurable thresholds."""
from typing import Dict, Any, List
from app.core import settings

DEFAULT_BUDGETS = {
    "environment_scan": 1.0, "execution_planning": 0.5,
    "workspace_preparation": 2.0, "dataset_loading": 1.0,
    "peft_injection": 5.0, "trainer_building": 3.0,
    "checkpoint_save": 2.0, "recovery_classification": 0.5,
    "validation_orchestration": 3.0, "execution_preparation": 2.0,
    "total_startup": 30.0,
}

class BudgetEvaluator:
    def __init__(self, budgets: Dict[str, float] = None):
        self._budgets = budgets or DEFAULT_BUDGETS

    def evaluate(self, results: Dict) -> Dict[str, Any]:
        violations = []
        stages = results.get("stages", {})
        for stage, budget in self._budgets.items():
            if stage == "total_startup":
                actual = results.get("total_duration", 0)
            else:
                actual = stages.get(stage, {}).get("duration", 0)
            if actual > budget and actual > 0:
                violations.append({
                    "stage": stage, "budget": budget, "actual": actual,
                    "over_by": round(actual - budget, 2),
                    "severity": "MAJOR" if actual > budget * 2 else "MINOR",
                })
        return {
            "total_budgets": len(self._budgets),
            "violations": len(violations),
            "compliant": len(self._budgets) - len(violations),
            "details": violations,
            "status": "PASS" if len(violations) == 0 else ("WARNING" if len(violations) <= 2 else "FAIL"),
        }
