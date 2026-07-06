"""Provider Health Monitoring."""
import json, time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any
from app.providers.registry import provider_registry
from app.core import settings

class ProviderHealthMonitor:
    def check_all(self) -> Dict[str, Any]:
        results = {}
        for name in provider_registry.list_providers():
            try: results[name] = provider_registry.get(name).health()
            except Exception as e: results[name] = {"status": "FAIL", "error": str(e)}
        return {"providers": results, "timestamp": datetime.now(timezone.utc).isoformat(),
                "forge_version": settings.app_version}
    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f: json.dump(self.check_all(), f, indent=2)
        return path
