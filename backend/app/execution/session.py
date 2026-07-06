"""Execution Session — root artifact for validation execution."""
import json, time, hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from app.core import settings

class ExecutionSession:
    def __init__(self, target_name: str, env_hash: str, plan_hash: str, dataset_hash: str, model: str):
        self.session_id = f"exec_{int(time.time())}"
        self.target = target_name; self.env_hash = env_hash; self.plan_hash = plan_hash
        self.dataset_hash = dataset_hash; self.model = model
        self.state = "CREATED"; self._stages: Dict = {}; self._started = datetime.now(timezone.utc)
        self._warnings: list = []; self._failures: list = []

    def stage(self, name: str, status: str, detail: str = ""):
        self._stages[name] = {"status": status, "detail": detail, "timestamp": datetime.now(timezone.utc).isoformat()}
        if status == "FAIL": self._failures.append(name)
        elif status == "WARNING": self._warnings.append(name)

    def finalize(self, status: str) -> Dict[str, Any]:
        self.state = status
        return self.generate()

    def generate(self) -> Dict[str, Any]:
        data = {
            "schema_version": "1.0", "forge_version": settings.app_version,
            "session_id": self.session_id, "execution_target": self.target,
            "environment_capability_hash": self.env_hash, "validation_plan_hash": self.plan_hash,
            "dataset_hash": self.dataset_hash, "model": self.model,
            "state": self.state, "stages": self._stages,
            "warnings": self._warnings, "failures": self._failures,
            "started_at": self._started.isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        return data

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f: json.dump(self.generate(), f, indent=2)
        return path
