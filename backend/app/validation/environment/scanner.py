"""Environment Capability Scanner — orchestrates all hardware/software/network scans."""
import json, time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any
from app.core import settings, get_logger
from app.validation.environment import hardware, cuda, dependencies, storage, network

logger = get_logger("app.validation.environment.scanner")


class EnvironmentScanner:
    """Scans the complete execution environment and determines capability."""

    def scan(self) -> Dict[str, Any]:
        t0 = time.time()
        hw = hardware.scan_hardware()
        cu = cuda.validate_cuda()
        deps = dependencies.validate_dependencies()
        disk = storage.validate_storage()
        net = network.validate_network()

        # Determine overall readiness
        dep_statuses = [d["status"] for d in deps.values() if d.get("required_pkg")]
        has_cuda = cu.get("cuda_available", False)
        has_network = net.get("internet", {}).get("status") == "PASS"

        if "FAIL" in dep_statuses:
            overall = "NOT_READY"
        elif not has_cuda:
            overall = "LIMITED"
        elif not has_network:
            overall = "LIMITED"
        else:
            overall = "READY"

        report = {
            "schema_version": "1.0", "forge_version": settings.app_version,
            "scanned_at": datetime.now(timezone.utc).isoformat(),
            "scan_duration_seconds": round(time.time() - t0, 2),
            "hardware": hw, "cuda": cu, "dependencies": deps,
            "storage": disk, "network": net,
            "overall_status": overall,
        }
        logger.info("environment_scan_complete", status=overall, duration=round(time.time()-t0, 2))
        return report

    def save_report(self, report: Dict, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        return path


scanner = EnvironmentScanner()
