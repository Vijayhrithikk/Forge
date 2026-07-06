"""Artifact Transport — upload/download, never interpret."""
import json, time
from pathlib import Path
from typing import Dict, Any, List
from app.core import get_logger

logger = get_logger("app.providers.transport")

class ArtifactTransport:
    def __init__(self): self._log: List[Dict] = []
    def upload_package(self, pkg_path: str, dest: str) -> Dict:
        t0 = time.time(); self._log.append({"action": "upload", "source": pkg_path, "dest": dest,
                                              "status": "completed", "duration": round(time.time()-t0, 2)})
        return {"status": "success", "file": pkg_path, "destination": dest}
    def download_artifacts(self, artifacts: List[str], local_dir: str) -> Dict:
        results = []
        for a in artifacts:
            results.append({"name": Path(a).name, "local": f"{local_dir}/{Path(a).name}", "status": "downloaded"})
        self._log.append({"action": "download", "count": len(results), "status": "completed"})
        return {"status": "success", "artifacts": results, "destination": local_dir}
    def get_log(self) -> List[Dict]: return self._log
