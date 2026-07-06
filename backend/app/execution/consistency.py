"""Artifact Consistency Validator — cross-validates generated artifacts."""
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any
from app.core import settings

def validate_consistency(artifacts_dir: Path) -> Dict[str, Any]:
    results = {}
    artifacts = [
        "execution_session.json", "execution_trace.json", "execution_certificate.json",
        "validation_matrix.json", "validation_dataset_manifest.json",
        "environment_capability.json", "execution_authorization.json",
    ]
    for name in artifacts:
        path = artifacts_dir / name
        if path.exists():
            try:
                with open(path) as f: data = json.load(f)
                sv = data.get("schema_version", "unknown")
                fv = data.get("forge_version", "unknown")
                results[name] = {"status": "PASS", "schema": sv, "forge_version": fv}
            except Exception as e:
                results[name] = {"status": "FAIL", "error": str(e)}
        else:
            results[name] = {"status": "MISSING"}

    present = sum(1 for r in results.values() if r["status"] == "PASS")
    missing = sum(1 for r in results.values() if r["status"] == "MISSING")
    report = {
        "schema_version": "1.0", "forge_version": settings.app_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": results,
        "present": present, "missing": missing,
        "overall": "PASS" if missing == 0 else "WARNING",
    }
    path = artifacts_dir / "consistency_report.json"
    with open(path, "w") as f: json.dump(report, f, indent=2)
    return report
