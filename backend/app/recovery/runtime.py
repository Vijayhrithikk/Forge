"""
Recovery Runtime — orchestrates failure analysis, classification, planning, and execution.

The Runtime owns recovery. Trainers never recover themselves.
"""

import json, time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Callable
from app.core import settings, get_logger

logger = get_logger("app.recovery.runtime")


class FailureClassifier:
    """Classifies failures as recoverable or non-recoverable."""
    RECOVERABLE = {"oom", "cuda_oom", "disk_full", "network_timeout", "filesystem_latency",
                   "download_timeout", "checkpoint_write_failure", "gpu_busy"}
    NON_RECOVERABLE = {"corrupt_checkpoint", "corrupt_manifest", "unsupported_gpu",
                       "invalid_training_plan", "dataset_corruption", "trainer_crash"}

    def classify(self, error_type: str, context: Dict = None) -> Dict[str, Any]:
        is_recoverable = error_type in self.RECOVERABLE
        strategy = self._strategy(error_type) if is_recoverable else "manual_intervention"
        return {
            "error_type": error_type, "recoverable": is_recoverable,
            "strategy": strategy, "severity": "HIGH" if not is_recoverable else "MEDIUM",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "context": context or {},
        }

    def _strategy(self, error_type: str) -> str:
        strategies = {
            "oom": "reduce_batch_size", "cuda_oom": "clear_cache_retry",
            "disk_full": "free_space", "network_timeout": "retry_backoff",
            "checkpoint_write_failure": "retry_write",
        }
        return strategies.get(error_type, "retry")


class RecoveryPlanner:
    """Plans recovery actions based on failure classification."""
    def plan(self, classification: Dict, checkpoint_path: Optional[Path] = None) -> Dict[str, Any]:
        plan = {
            "classification": classification,
            "checkpoint_available": checkpoint_path is not None,
            "checkpoint": str(checkpoint_path) if checkpoint_path else None,
            "actions": [],
            "can_resume": False,
        }
        if classification["recoverable"] and classification["strategy"] == "reduce_batch_size":
            plan["actions"] = ["clear_cuda_cache", "halve_batch_size", "reload_trainer", "resume"]
            plan["can_resume"] = True
        elif classification["recoverable"] and checkpoint_path:
            plan["actions"] = ["verify_checkpoint", "reload_trainer_from_checkpoint", "resume"]
            plan["can_resume"] = True
        elif classification["recoverable"]:
            plan["actions"] = ["retry_with_backoff", "resume"]
            plan["can_resume"] = True
        return plan


class RecoveryRuntime:
    """Orchestrates recovery: classify, plan, execute, resume."""

    def __init__(self, event_emitter: Optional[Callable] = None):
        self._classifier = FailureClassifier()
        self._planner = RecoveryPlanner()
        self._emit = event_emitter or (lambda *a, **kw: None)
        self._recoveries: list = []

    def handle_failure(self, error_type: str, context: Dict = None,
                       checkpoint_path: Optional[Path] = None) -> Dict[str, Any]:
        self._emit("FAILURE_DETECTED", f"Failure detected: {error_type}")
        classification = self._classifier.classify(error_type, context)
        plan = self._planner.plan(classification, checkpoint_path)

        recovery = {
            "recovery_id": f"rec_{int(time.time())}",
            "classification": classification,
            "plan": plan,
            "executed_at": datetime.now(timezone.utc).isoformat(),
            "forge_version": settings.app_version, "schema_version": "1.0",
        }
        self._recoveries.append(recovery)

        if not classification["recoverable"]:
            self._emit("RECOVERY_FAILED", f"Unrecoverable: {error_type}")

        logger.info("failure_handled", error_type=error_type,
                     recoverable=classification["recoverable"], strategy=classification["strategy"])
        return recovery

    def cancel(self, reason: str = "User requested") -> Dict[str, Any]:
        self._emit("CANCELLATION_REQUESTED", reason)
        return {"status": "cancelled", "reason": reason, "timestamp": datetime.now(timezone.utc).isoformat()}

    def get_recoveries(self) -> list:
        return self._recoveries

    def save_manifest(self, recovery: Dict, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "recovery_manifest.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(recovery, f, indent=2, ensure_ascii=False)
        return path
