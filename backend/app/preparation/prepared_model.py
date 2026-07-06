"""
PreparedModel — the immutable result of model acquisition + loading + preparation.

Mission C (Training Runtime) consumes PreparedModel. It must never
modify it directly. This guarantees deterministic, reproducible training.
"""

import json
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from app.core import settings, get_logger
from app.preparation.device import DeviceInfo
from app.preparation.precision import PrecisionConfig
from app.preparation.optimization import OptimizationConfig
from app.preparation.memory import MemoryInfo

logger = get_logger("app.preparation.prepared_model")

MANIFEST_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)  # Immutable
class PreparedModel:
    """Immutable container for a fully prepared model.

    Mission C takes this as input. No training code may modify it.
    All configuration decisions are finalized here.
    """

    # Model assets
    model: Any = None  # The loaded HuggingFace model
    tokenizer: Any = None  # The loaded tokenizer
    config: Any = None  # The model config

    # Device
    device: DeviceInfo = field(default_factory=lambda: DeviceInfo(name="cpu", device_type="cpu"))

    # Precision
    precision: PrecisionConfig = field(default_factory=lambda: PrecisionConfig(
        precision="fp32", dtype_name="float32", vram_factor=2.0, description="32-bit"))

    # Memory
    memory: MemoryInfo = field(default_factory=MemoryInfo)

    # Optimization
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)

    # Metadata
    model_id: str = ""
    runtime_id: str = ""
    prepared_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Validation
    validation_results: Dict[str, Any] = field(default_factory=dict)

    # Manifests
    model_manifest: Dict[str, Any] = field(default_factory=dict)
    tokenizer_manifest: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_ready(self) -> bool:
        """Check if the prepared model is ready for training."""
        return self.model is not None and self.tokenizer is not None

    def generate_preparation_manifest(self) -> Dict[str, Any]:
        """Generate the preparation_manifest.json."""
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "forge_version": settings.app_version,
            "runtime_id": self.runtime_id,
            "model_id": self.model_id,
            "prepared_at": self.prepared_at,
            "device": {
                "name": self.device.name,
                "type": self.device.device_type,
                "index": self.device.index,
            },
            "precision": {
                "precision": self.precision.precision,
                "dtype": self.precision.dtype_name,
                "bf16_supported": self.precision.bf16_supported,
                "fp16_supported": self.precision.fp16_supported,
            },
            "memory": {
                "total_gb": self.memory.total_gb,
                "free_gb": self.memory.free_gb,
                "projected_gb": self.memory.projected_gb,
                "status": self.memory.status,
            },
            "optimization": self.optimization.flags,
            "validation": self.validation_results,
        }

    def save_manifest(self, directory: Path) -> Path:
        """Persist the preparation manifest."""
        manifest = self.generate_preparation_manifest()
        path = directory / "preparation_manifest.json"
        directory.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        logger.info("preparation_manifest_saved", path=str(path))
        return path
