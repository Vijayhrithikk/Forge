"""Validation History — stores every execution, never overwrites previous results."""
import json, shutil
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List
from app.core import settings

HISTORY_DIR = Path("validation_history")


class ValidationHistory:
    def __init__(self, base_dir: Path = HISTORY_DIR):
        self._dir = base_dir; self._dir.mkdir(parents=True, exist_ok=True)

    def store(self, session_id: str, source_dir: Path) -> Path:
        exec_dir = self._dir / session_id
        exec_dir.mkdir(parents=True, exist_ok=True)
        for f in source_dir.glob("*.json"):
            shutil.copy2(f, exec_dir / f.name)
        self._update_index()
        return exec_dir

    def list_executions(self) -> List[Dict]:
        idx_path = self._dir / "history_index.json"
        if idx_path.exists():
            with open(idx_path) as f: return json.load(f).get("executions", [])
        return []

    def _update_index(self):
        executions = []
        for d in sorted(self._dir.iterdir()):
            if d.is_dir() and d.name.startswith("exec_"):
                executions.append({"execution_id": d.name, "stored_at": datetime.now(timezone.utc).isoformat()})
        with open(self._dir / "history_index.json", "w") as f:
            json.dump({"executions": executions, "updated_at": datetime.now(timezone.utc).isoformat(),
                        "forge_version": settings.app_version}, f, indent=2)
