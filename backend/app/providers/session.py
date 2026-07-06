"""Provider Session — tracks remote execution lifecycle."""
import json, time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from app.core import settings

class ProviderSession:
    def __init__(self, provider_name: str, package_hash: str = ""):
        self.session_id = f"prov_{int(time.time())}"
        self.provider = provider_name; self.package_hash = package_hash
        self.state = "IDLE"; self.job_id = ""; self._started = datetime.now(timezone.utc)
        self._stages: Dict = {}

    def record(self, stage: str, status: str, detail: str = ""):
        self._stages[stage] = {"status": status, "detail": detail, "timestamp": datetime.now(timezone.utc).isoformat()}

    def generate(self) -> Dict:
        return {"schema_version": "1.0", "forge_version": settings.app_version,
                "session_id": self.session_id, "provider": self.provider, "package_hash": self.package_hash,
                "state": self.state, "job_id": self.job_id, "stages": self._stages,
                "started_at": self._started.isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat()}

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f: json.dump(self.generate(), f, indent=2)
        return path
