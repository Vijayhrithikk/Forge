"""Storage validator — disk space, workspace, cache, checkpoint directories."""
import shutil
from pathlib import Path
from typing import Dict, Any

def validate_storage(workspace: Path = Path(".")) -> Dict[str, Any]:
    result = {}
    for name, path in [("workspace", workspace), ("cache", Path("models/cache")),
                        ("temp", Path("data/tmp")), ("output", Path("output"))]:
        try:
            path.mkdir(parents=True, exist_ok=True)
            disk = shutil.disk_usage(path)
            free_gb = disk.free / (1024**3)
            total_gb = disk.total / (1024**3)
            status = "PASS" if free_gb >= 5 else ("WARNING" if free_gb >= 1 else "FAIL")
            result[name] = {"status": status, "free_gb": round(free_gb, 1),
                           "total_gb": round(total_gb, 1), "path": str(path)}
        except Exception as e:
            result[name] = {"status": "FAIL", "error": str(e)}
    result["minimum_model_size_gb"] = 3.0  # approx for 1.5B model
    result["safety_margin_gb"] = 10.0
    return result
