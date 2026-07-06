"""Performance Registry — append-only benchmark history."""
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List
from app.core import settings

REGISTRY_PATH = Path("data/performance/performance_registry.json")

class PerformanceRegistry:
    def __init__(self, path: Path = REGISTRY_PATH):
        self._path = path; self._path.parent.mkdir(parents=True, exist_ok=True)
        self._entries = self._load()

    def _load(self) -> list:
        if self._path.exists():
            with open(self._path) as f: return json.load(f).get("entries", [])
        return []

    def record(self, benchmark_id: str, results: Dict, target: str = "local") -> Dict:
        entry = {
            "benchmark_id": benchmark_id,
            "target": target,
            "forge_version": settings.app_version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_duration": results.get("total_duration", 0),
            "stages": results.get("stages", {}),
        }
        self._entries.append(entry)
        with open(self._path, "w") as f:
            json.dump({"schema_version": "1.0", "entries": self._entries,
                        "updated_at": datetime.now(timezone.utc).isoformat()}, f, indent=2)
        return entry

    def latest(self) -> Dict: return self._entries[-1] if self._entries else {}
    def history(self) -> List[Dict]: return self._entries
