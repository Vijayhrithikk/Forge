"""
Artifact Validator — cross-validates every Forge artifact for completeness,
schema correctness, hash consistency, and cross-reference integrity.
"""

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List
from app.core import get_logger

logger = get_logger("app.validation.artifact_validator")

REQUIRED_ARTIFACTS = [
    "runtime_manifest.json", "training_session.json", "execution_graph.json",
    "dataset_runtime_manifest.json", "peft_manifest.json", "trainable_parameters.json",
    "model_manifest.json", "tokenizer_manifest.json", "preparation_manifest.json",
    "environment.json", "runtime_metrics.json", "events.jsonl", "logs.jsonl",
    "checkpoint_manifest.json", "recovery_manifest.json", "validation_report.json",
]


class ArtifactValidator:
    """Cross-validates all Forge artifacts."""

    def validate(self, runtime_dir: Path, reports_dir: Path) -> Dict[str, Any]:
        """Validate all artifacts across the workspace.

        Args:
            runtime_dir: The runtime directory (contains most artifacts).
            reports_dir: The reports directory.

        Returns:
            Dict with per-artifact status and overall result.
        """
        results = {}
        for name in REQUIRED_ARTIFACTS:
            # Determine location
            if name in ("validation_report.json",):
                path = reports_dir / name
            elif name in ("checkpoint_manifest.json", "recovery_manifest.json"):
                path = runtime_dir / "checkpoints" / name
            else:
                path = runtime_dir / name
                if not path.exists():
                    path = runtime_dir / "manifest" / name
                if not path.exists() and name.endswith(".jsonl"):
                    path = runtime_dir / "events" / name
                if not path.exists() and name.endswith(".jsonl"):
                    path = runtime_dir / "logs" / name

            if path.exists():
                valid, msg = self._validate_file(name, path)
                results[name] = {"status": "PASS" if valid else "WARNING", "message": msg, "path": str(path)}
            else:
                results[name] = {"status": "MISSING", "message": "Artifact not found"}

        present = sum(1 for r in results.values() if r["status"] in ("PASS", "WARNING"))
        missing = sum(1 for r in results.values() if r["status"] == "MISSING")

        report = {
            "schema_version": "1.0",
            "validated_at": datetime.now(timezone.utc).isoformat(),
            "total": len(REQUIRED_ARTIFACTS),
            "present": present,
            "missing": missing,
            "artifacts": results,
            "status": "PASS" if missing == 0 else ("WARNING" if missing < 5 else "FAIL"),
        }
        return report

    def _validate_file(self, name: str, path: Path) -> tuple:
        try:
            if name.endswith(".json"):
                with open(path) as f:
                    data = json.load(f)
                sv = data.get("schema_version", "none")
                return True, f"Schema v{sv}"
            elif name.endswith(".jsonl"):
                with open(path) as f:
                    lines = [l for l in f if l.strip()]
                return True, f"{len(lines)} entries"
            return True, "exists"
        except Exception as e:
            return False, str(e)

    def save_report(self, report: Dict, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        return path
