"""
Compatibility Engine — validates that dataset, model, LoRA config,
hyperparameters, and hardware are mutually compatible.

Returns PASS, WARNING, or FAIL with explanations — never just a boolean.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

from app.core import get_logger
from app.engines.training.registry import model_registry, ModelEntry
from app.engines.training.config import TrainingConfig
from app.engines.dataset.workspace import workspace_engine

logger = get_logger("app.engines.training.compatibility")


@dataclass
class CompatibilityResult:
    """A single compatibility check result."""
    check: str
    status: str  # PASS, WARNING, FAIL
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompatibilityReport:
    """Aggregated compatibility check results."""
    results: List[CompatibilityResult] = field(default_factory=list)

    @property
    def all_pass(self) -> bool:
        return all(r.status == "PASS" for r in self.results)

    @property
    def has_failures(self) -> bool:
        return any(r.status == "FAIL" for r in self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.status == "PASS")

    @property
    def warnings(self) -> int:
        return sum(1 for r in self.results if r.status == "WARNING")

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status == "FAIL")


class CompatibilityEngine:
    """Validates end-to-end compatibility of the training configuration."""

    def check(
        self,
        config: TrainingConfig,
        dataset_sample_count: int = 0,
        dataset_avg_tokens: float = 0,
    ) -> CompatibilityReport:
        """Run all compatibility checks.

        Args:
            config: The full training configuration.
            dataset_sample_count: Number of records in the dataset.
            dataset_avg_tokens: Average tokens per sample.

        Returns:
            CompatibilityReport with all check results.
        """
        results: List[CompatibilityResult] = []

        # 1. Model existence
        try:
            model = model_registry.get_model(config.model_id)
            results.append(CompatibilityResult(
                check="model_available",
                status="PASS",
                message=f"Model '{model.name}' found in registry.",
            ))
        except KeyError:
            results.append(CompatibilityResult(
                check="model_available",
                status="FAIL",
                message=f"Model '{config.model_id}' not found in registry.",
            ))
            # Can't continue without model
            return CompatibilityReport(results=results)

        # 2. Dataset availability
        try:
            ds_path = workspace_engine.get_dataset_path(config.project_id) / "original.jsonl"
            if ds_path.exists():
                results.append(CompatibilityResult(
                    check="dataset_available",
                    status="PASS",
                    message=f"Dataset found ({dataset_sample_count} samples).",
                    details={"path": str(ds_path), "samples": dataset_sample_count},
                ))
            else:
                results.append(CompatibilityResult(
                    check="dataset_available",
                    status="FAIL",
                    message="No dataset uploaded for this project. Upload a JSONL dataset first.",
                ))
        except FileNotFoundError:
            results.append(CompatibilityResult(
                check="dataset_available",
                status="FAIL",
                message="Project not found. Create a project first.",
            ))

        # 3. LoRA support
        if model.supports_lora:
            results.append(CompatibilityResult(
                check="lora_supported",
                status="PASS",
                message=f"Model supports LoRA fine-tuning.",
            ))
        else:
            results.append(CompatibilityResult(
                check="lora_supported",
                status="FAIL",
                message="This model does not support LoRA fine-tuning.",
            ))

        # 4. Precision compatibility
        precision = config.hyperparams.precision
        if precision in model.precision:
            results.append(CompatibilityResult(
                check="precision_compatible",
                status="PASS",
                message=f"Precision '{precision}' is supported by this model.",
            ))
        else:
            supported = ", ".join(model.precision)
            results.append(CompatibilityResult(
                check="precision_compatible",
                status="FAIL",
                message=f"Precision '{precision}' is not supported. Supported: {supported}.",
            ))

        # 5. Sequence length check
        if config.hyperparams.max_sequence_length > model.context_length:
            results.append(CompatibilityResult(
                check="sequence_length",
                status="FAIL",
                message=f"Max sequence length ({config.hyperparams.max_sequence_length}) exceeds model context ({model.context_length}).",
            ))
        elif config.hyperparams.max_sequence_length > model.context_length * 0.8:
            results.append(CompatibilityResult(
                check="sequence_length",
                status="WARNING",
                message=f"Sequence length ({config.hyperparams.max_sequence_length}) is near the model's context limit ({model.context_length}).",
            ))
        else:
            results.append(CompatibilityResult(
                check="sequence_length",
                status="PASS",
                message=f"Sequence length ({config.hyperparams.max_sequence_length}) within model context ({model.context_length}).",
            ))

        # 6. Dataset size vs epochs
        if dataset_sample_count > 0 and config.hyperparams.epochs > 5 and dataset_sample_count < 100:
            results.append(CompatibilityResult(
                check="dataset_size",
                status="WARNING",
                message=f"Small dataset ({dataset_sample_count} samples) with many epochs ({config.hyperparams.epochs}). Risk of overfitting.",
            ))
        else:
            results.append(CompatibilityResult(
                check="dataset_size",
                status="PASS",
                message=f"Dataset size ({dataset_sample_count} samples) is reasonable for {config.hyperparams.epochs} epochs.",
            ))

        # 7. Target modules
        if config.lora.target_modules:
            results.append(CompatibilityResult(
                check="target_modules",
                status="PASS",
                message=f"{len(config.lora.target_modules)} target modules configured.",
                details={"modules": config.lora.target_modules},
            ))
        else:
            recommended = model.lora_defaults.get("target_modules", [])
            results.append(CompatibilityResult(
                check="target_modules",
                status="WARNING",
                message="No target modules specified. Using model defaults.",
                details={"recommended": recommended},
            ))

        # 8. Optimizer compatibility
        opt = model_registry.get_optimizer(config.hyperparams.optimizer)
        results.append(CompatibilityResult(
            check="optimizer",
            status="PASS",
            message=f"Optimizer '{opt.name}' is supported.",
            details={"advantages": opt.advantages},
        ))

        return CompatibilityReport(results=results)


# Singleton
compatibility_engine = CompatibilityEngine()
