"""
Dataset Runtime — loads, validates, and manifests training datasets.
Reads datasets from Sprint 1. Validates against Training Plan.
"""

import json, hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from app.core import settings, get_logger
from app.training.exceptions import DatasetLoadError

logger = get_logger("app.training.dataset_runtime")


class DatasetRuntime:
    """Loads and validates the training dataset from Sprint 1 workspace."""
    def __init__(self, project_path: Path):
        self._project_path = project_path
        self._dataset_path = project_path / "dataset" / "original.jsonl"
        self._manifest: Dict[str, Any] = {}

    def load_and_validate(self, training_plan: Optional[Dict] = None) -> Dict[str, Any]:
        if not self._dataset_path.exists():
            raise DatasetLoadError("Dataset not found. Upload a JSONL dataset first.")
        records = []
        with open(self._dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line: records.append(json.loads(line))
        sample_count = len(records)
        if sample_count == 0:
            raise DatasetLoadError("Dataset is empty.")
        dataset_hash = hashlib.sha256(self._dataset_path.read_bytes()).hexdigest()
        meta_path = self._project_path / "metadata" / "metadata.json"
        avg_tokens, max_tokens = 0.0, 0
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            avg_tokens = meta.get("average_tokens", 0.0)
            max_tokens = meta.get("maximum_tokens", 0)
        self._manifest = {
            "schema_version": "1.0", "forge_version": settings.app_version,
            "dataset_hash": dataset_hash, "sample_count": sample_count,
            "average_tokens": avg_tokens, "maximum_tokens": max_tokens,
            "path": str(self._dataset_path),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        logger.info("dataset_loaded", samples=sample_count, hash=dataset_hash[:12])
        return {"records": records, "sample_count": sample_count, "hash": dataset_hash,
                "avg_tokens": avg_tokens, "max_tokens": max_tokens}

    def save_manifest(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "dataset_runtime_manifest.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._manifest, f, indent=2, ensure_ascii=False)
        return path
