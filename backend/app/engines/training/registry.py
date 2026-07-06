"""
Model Registry — loads and queries the filesystem-based model registry.

Reads models/registry.json and provides typed access to model metadata,
LoRA defaults, GPU compatibility, optimizers, schedulers, and precision modes.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from app.core import get_logger

logger = get_logger("app.engines.training.registry")

# Path to registry relative to the backend directory
_REGISTRY_PATH = Path(__file__).parent.parent.parent.parent / "models" / "registry.json"


# ------------------------------------------------------------------
# Typed data structures
# ------------------------------------------------------------------

@dataclass
class ModelEntry:
    """Typed representation of a model from the registry."""
    id: str
    name: str
    huggingface_id: str
    architecture: str
    family: str
    parameters: int
    parameters_display: str
    context_length: int
    tokenizer: str
    precision: List[str]
    recommended_precision: str
    recommended_vram_gb: float
    recommended_gpu: str
    license: str
    instruction_tuned: bool
    supports_lora: bool
    lora_defaults: Dict[str, Any]
    training_notes: str
    tags: List[str]


@dataclass
class GPUEntry:
    """Typed representation of a GPU from the registry."""
    name: str
    vram_gb: float
    cuda_cores: int
    compatibility: str  # "compatible" | "limited" | "unsupported"
    notes: str


@dataclass
class OptimizerEntry:
    name: str
    description: str
    advantages: List[str]
    disadvantages: List[str]
    recommended_use: str


@dataclass
class SchedulerEntry:
    name: str
    description: str
    use_cases: List[str]
    recommendation: str


@dataclass
class PrecisionEntry:
    name: str
    description: str
    vram_factor: float
    compatibility: str
    notes: str


# ------------------------------------------------------------------
# Registry loader
# ------------------------------------------------------------------

class ModelRegistry:
    """Loads and queries the model registry from disk.

    The registry is a single JSON file (models/registry.json) that
    describes every supported model, GPU, optimizer, scheduler, and
    precision mode. No model information is hardcoded in Python.
    """

    def __init__(self, registry_path: Optional[Path] = None):
        self._path = registry_path or _REGISTRY_PATH
        self._data: Dict[str, Any] = {}
        self._models: Dict[str, ModelEntry] = {}
        self._gpus: Dict[str, GPUEntry] = {}
        self._optimizers: Dict[str, OptimizerEntry] = {}
        self._schedulers: Dict[str, SchedulerEntry] = {}
        self._precision_modes: Dict[str, PrecisionEntry] = {}
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def schema_version(self) -> str:
        return self._data.get("schema_version", "unknown")

    @property
    def model_ids(self) -> List[str]:
        return list(self._models.keys())

    def get_model(self, model_id: str) -> ModelEntry:
        """Return a model entry by ID.

        Raises KeyError if the model is not found.
        """
        if model_id not in self._models:
            raise KeyError(f"Model not found in registry: {model_id}")
        return self._models[model_id]

    def list_models(
        self,
        family: Optional[str] = None,
        tags: Optional[List[str]] = None,
        search: Optional[str] = None,
        max_parameters: Optional[int] = None,
        min_context: Optional[int] = None,
    ) -> List[ModelEntry]:
        """List models with optional filtering.

        Args:
            family: Filter by model family (llama, qwen2, mistral, gemma, phi).
            tags: Filter by tags (e.g. ["recommended", "lightweight"]).
            search: Free-text search across name and description.
            max_parameters: Maximum parameter count filter.
            min_context: Minimum context length filter.

        Returns:
            List of matching ModelEntry objects.
        """
        results = list(self._models.values())

        if family:
            results = [m for m in results if m.family == family]
        if tags:
            results = [m for m in results if any(t in m.tags for t in tags)]
        if search:
            q = search.lower()
            results = [
                m
                for m in results
                if q in m.name.lower()
                or q in m.architecture.lower()
                or q in m.family.lower()
                or any(q in t for t in m.tags)
            ]
        if max_parameters is not None:
            results = [m for m in results if m.parameters <= max_parameters]
        if min_context is not None:
            results = [m for m in results if m.context_length >= min_context]

        return results

    def list_gpus(self, min_vram: Optional[float] = None) -> List[GPUEntry]:
        """List GPUs with optional minimum VRAM filter."""
        gpus = list(self._gpus.values())
        if min_vram is not None:
            gpus = [g for g in gpus if g.vram_gb >= min_vram]
        return gpus

    def get_gpu(self, gpu_id: str) -> GPUEntry:
        return self._gpus[gpu_id]

    def get_optimizer(self, opt_id: str) -> OptimizerEntry:
        return self._optimizers[opt_id]

    def list_optimizers(self) -> List[OptimizerEntry]:
        return list(self._optimizers.values())

    def get_scheduler(self, sched_id: str) -> SchedulerEntry:
        return self._schedulers[sched_id]

    def list_schedulers(self) -> List[SchedulerEntry]:
        return list(self._schedulers.values())

    def get_precision(self, prec_id: str) -> PrecisionEntry:
        return self._precision_modes[prec_id]

    def list_precisions(self) -> List[PrecisionEntry]:
        return list(self._precision_modes.values())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load and parse the registry JSON file."""
        if not self._path.exists():
            logger.error("registry_not_found", path=str(self._path))
            raise FileNotFoundError(f"Model registry not found at {self._path}")

        with open(self._path, "r", encoding="utf-8") as f:
            self._data = json.load(f)

        # Parse models
        for m in self._data.get("models", []):
            entry = ModelEntry(
                id=m["id"],
                name=m["name"],
                huggingface_id=m["huggingface_id"],
                architecture=m["architecture"],
                family=m["family"],
                parameters=m["parameters"],
                parameters_display=m["parameters_display"],
                context_length=m["context_length"],
                tokenizer=m["tokenizer"],
                precision=m["precision"],
                recommended_precision=m["recommended_precision"],
                recommended_vram_gb=m["recommended_vram_gb"],
                recommended_gpu=m["recommended_gpu"],
                license=m["license"],
                instruction_tuned=m["instruction_tuned"],
                supports_lora=m["supports_lora"],
                lora_defaults=m["lora_defaults"],
                training_notes=m["training_notes"],
                tags=m.get("tags", []),
            )
            self._models[entry.id] = entry

        # Parse GPUs
        for gpu_id, g in self._data.get("gpu_compatibility", {}).items():
            self._gpus[gpu_id] = GPUEntry(
                name=g["name"],
                vram_gb=g["vram_gb"],
                cuda_cores=g["cuda_cores"],
                compatibility=g["compatibility"],
                notes=g["notes"],
            )

        # Parse optimizers
        for opt_id, o in self._data.get("optimizers", {}).items():
            self._optimizers[opt_id] = OptimizerEntry(
                name=o["name"],
                description=o["description"],
                advantages=o["advantages"],
                disadvantages=o["disadvantages"],
                recommended_use=o["recommended_use"],
            )

        # Parse schedulers
        for sched_id, s in self._data.get("schedulers", {}).items():
            self._schedulers[sched_id] = SchedulerEntry(
                name=s["name"],
                description=s["description"],
                use_cases=s["use_cases"],
                recommendation=s["recommendation"],
            )

        # Parse precision modes
        for prec_id, p in self._data.get("precision_modes", {}).items():
            self._precision_modes[prec_id] = PrecisionEntry(
                name=p["name"],
                description=p["description"],
                vram_factor=p["vram_factor"],
                compatibility=p["compatibility"],
                notes=p["notes"],
            )

        logger.info(
            "registry_loaded",
            path=str(self._path),
            models=len(self._models),
            gpus=len(self._gpus),
            optimizers=len(self._optimizers),
            schedulers=len(self._schedulers),
            precision_modes=len(self._precision_modes),
        )


# Singleton
model_registry = ModelRegistry()
