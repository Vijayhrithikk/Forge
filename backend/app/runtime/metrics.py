"""
Runtime Metrics — continuous progress tracking for execution.

Metrics update after every state transition. Progress is
stage-based, not time-based. The frontend consumes this.
"""

import json
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List

from app.runtime.state_machine import RuntimeState


# Progress weights per stage (for progress calculation)
STAGE_WEIGHTS: Dict[str, int] = {
    "workspace_preparation": 5,
    "workspace_validation": 5,
    "environment_validation": 10,
    "gpu_discovery": 5,
    "gpu_validation": 5,
    "manifest_creation": 5,
    "preparation_complete": 5,
    # Future: Mission B
    "model_download": 10,
    "model_validation": 5,
    "tokenizer_loading": 5,
    "lora_injection": 5,
    "training_setup": 5,
    "training": 25,
    "checkpointing": 3,
    "evaluation": 5,
    "artifact_packaging": 5,
}


class RuntimeMetrics:
    """Tracks execution progress, timing, and completion status."""

    def __init__(self, runtime_id: str):
        self.runtime_id = runtime_id
        self._start_time = time.time()
        self._completed_stages: List[str] = []
        self._current_stage: str = ""
        self._stage_times: Dict[str, float] = {}
        self._warnings: int = 0
        self._errors: int = 0
        self._recovery_attempts: int = 0
        self._stage_start: float = self._start_time

    # ------------------------------------------------------------------
    # Stage tracking
    # ------------------------------------------------------------------

    def begin_stage(self, stage: str) -> None:
        self._current_stage = stage
        self._stage_start = time.time()

    def complete_stage(self, stage: str) -> None:
        elapsed = time.time() - self._stage_start
        self._completed_stages.append(stage)
        self._stage_times[stage] = round(elapsed, 2)

    def add_warning(self) -> None:
        self._warnings += 1

    def add_error(self) -> None:
        self._errors += 1

    def add_recovery_attempt(self) -> None:
        self._recovery_attempts += 1

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    @property
    def progress(self) -> float:
        """Compute current progress as percentage (0-100)."""
        total = 0
        for stage in self._completed_stages:
            total += STAGE_WEIGHTS.get(stage, 5)
        return min(100.0, total)

    @property
    def elapsed_seconds(self) -> float:
        return round(time.time() - self._start_time, 1)

    @property
    def remaining_stages(self) -> List[str]:
        all_stages = list(STAGE_WEIGHTS.keys())
        return [s for s in all_stages if s not in self._completed_stages]

    @property
    def average_stage_duration(self) -> float:
        if not self._stage_times:
            return 0.0
        return round(sum(self._stage_times.values()) / len(self._stage_times), 2)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "runtime_id": self.runtime_id,
            "progress": self.progress,
            "elapsed_seconds": self.elapsed_seconds,
            "current_stage": self._current_stage,
            "completed_stages": self._completed_stages,
            "remaining_stages": self.remaining_stages,
            "average_stage_duration": self.average_stage_duration,
            "warnings": self._warnings,
            "errors": self._errors,
            "recovery_attempts": self._recovery_attempts,
            "stage_times": self._stage_times,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def save(self, metrics_dir: Path) -> Path:
        """Persist metrics to disk."""
        metrics_dir.mkdir(parents=True, exist_ok=True)
        path = metrics_dir / "runtime_metrics.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return path
