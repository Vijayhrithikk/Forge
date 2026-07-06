"""Benchmark Engine — measures every Runtime stage."""
import time, json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Callable, Optional
from app.core import settings
from app.performance.profiler import RuntimeProfiler

RUNTIME_STAGES = [
    "environment_scan", "execution_planning", "workspace_preparation",
    "dataset_loading", "tokenization", "peft_injection",
    "trainer_building", "checkpoint_save", "recovery_classification",
    "validation_orchestration", "execution_preparation",
]

class BenchmarkEngine:
    """Measures every Runtime stage with profiling."""
    def __init__(self):
        self._results: Dict[str, Dict] = {}
        self._profiler = RuntimeProfiler()

    def run_all(self, callbacks: Optional[Dict[str, Callable]] = None) -> Dict[str, Any]:
        t0 = time.time()
        self._profiler.start()
        for stage in RUNTIME_STAGES:
            t1 = time.time()
            status = "SKIPPED"
            if callbacks and stage in callbacks:
                try:
                    callbacks[stage]()
                    status = "PASS"
                except Exception as e:
                    status = f"FAIL: {e}"
            self._results[stage] = {
                "duration": round(time.time() - t1, 3),
                "status": status,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        profile = self._profiler.stop()
        report = {
            "schema_version": "1.0", "forge_version": settings.app_version,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_duration": round(time.time() - t0, 2),
            "stages": self._results,
            "profiler": profile,
            "executed": sum(1 for r in self._results.values() if r["status"] == "PASS"),
            "skipped": sum(1 for r in self._results.values() if r["status"] == "SKIPPED"),
        }
        return report

    def save(self, report: Dict, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f: json.dump(report, f, indent=2)
        return path
