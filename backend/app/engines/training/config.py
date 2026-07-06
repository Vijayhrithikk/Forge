"""
Training Configuration Engines — LoRA and Hyperparameter management.

Validates, explains, and optimizes training configuration before
any GPU resources are committed.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from enum import Enum

from app.core import get_logger
from app.engines.training.registry import model_registry, ModelEntry

logger = get_logger("app.engines.training.config")


# ------------------------------------------------------------------
# Configuration data structures
# ------------------------------------------------------------------

@dataclass
class LoRAConfig:
    """LoRA (Low-Rank Adaptation) configuration for fine-tuning."""

    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"  # none, all, lora_only
    target_modules: List[str] = field(default_factory=list)
    task_type: str = "CAUSAL_LM"
    modules_to_save: List[str] = field(default_factory=list)

    def validate(self) -> List[Dict[str, Any]]:
        """Validate LoRA configuration and return warnings/errors."""
        issues: List[Dict[str, Any]] = []

        if self.rank < 1:
            issues.append({"field": "rank", "severity": "FAIL", "message": "LoRA rank must be >= 1."})
        elif self.rank > 256:
            issues.append({"field": "rank", "severity": "WARNING", "message": "Very high rank (>256). Diminishing returns expected."})

        if self.alpha < 1:
            issues.append({"field": "alpha", "severity": "FAIL", "message": "LoRA alpha must be >= 1."})
        elif self.alpha > 512:
            issues.append({"field": "alpha", "severity": "WARNING", "message": "Very high alpha (>512)."})

        if self.alpha < self.rank:
            issues.append({"field": "alpha", "severity": "WARNING", "message": "Alpha is less than rank. Typically alpha = 2 × rank is a good default."})

        if not 0 <= self.dropout <= 1:
            issues.append({"field": "dropout", "severity": "FAIL", "message": "Dropout must be between 0 and 1."})

        if self.bias not in ("none", "all", "lora_only"):
            issues.append({"field": "bias", "severity": "FAIL", "message": "Bias must be 'none', 'all', or 'lora_only'."})

        if not self.target_modules:
            issues.append({"field": "target_modules", "severity": "WARNING", "message": "No target modules specified. LoRA will not be applied to any layers."})

        return issues

    def get_parameter_descriptions(self) -> List[Dict[str, Any]]:
        """Return educational descriptions for every LoRA parameter."""
        return [
            {
                "name": "Rank (r)",
                "key": "rank",
                "description": "Higher rank increases learning capacity and adapter size. Lower rank is more efficient but may underfit. Common values: 8–64.",
                "recommended_range": "8–32",
                "current_value": self.rank,
                "recommendation": "16" if self.rank == 16 else f"Current: {self.rank} — {'Good balance' if 8 <= self.rank <= 32 else 'Consider adjusting to 8-32 range'}.",
            },
            {
                "name": "Alpha",
                "key": "alpha",
                "description": "Scaling factor for LoRA updates. Typically set to 2× rank. Higher alpha increases the influence of the adaptation.",
                "recommended_range": f"{max(1, self.rank)}–{self.rank * 4}",
                "current_value": self.alpha,
                "recommendation": f"Current: {self.alpha} — {'Good choice' if self.alpha >= self.rank else 'Consider alpha ≥ rank'}.",
            },
            {
                "name": "Dropout",
                "key": "dropout",
                "description": "Probability of dropping LoRA activations during training. Helps prevent overfitting on small datasets.",
                "recommended_range": "0.0–0.1",
                "current_value": self.dropout,
                "recommendation": f"Current: {self.dropout} — {'Good' if self.dropout <= 0.1 else 'High — may slow convergence'}.",
            },
            {
                "name": "Bias",
                "key": "bias",
                "description": "Whether to train bias parameters alongside LoRA weights. 'none' saves VRAM; 'lora_only' trains bias on LoRA layers only.",
                "recommended_range": "none | all | lora_only",
                "current_value": self.bias,
                "recommendation": f"Current: {self.bias} — {'Recommended default' if self.bias == 'none' else 'May increase VRAM usage'}.",
            },
        ]


@dataclass
class HyperparamsConfig:
    """Hyperparameter configuration for training."""

    epochs: int = 3
    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    optimizer: str = "adamw"
    scheduler: str = "linear"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_sequence_length: int = 2048
    seed: int = 42
    precision: str = "bf16"
    logging_steps: int = 10
    save_steps: int = 100

    def validate(self) -> List[Dict[str, Any]]:
        """Validate hyperparameters and return warnings/errors."""
        issues: List[Dict[str, Any]] = []

        if self.epochs < 1:
            issues.append({"field": "epochs", "severity": "FAIL", "message": "Epochs must be >= 1."})
        elif self.epochs > 10:
            issues.append({"field": "epochs", "severity": "WARNING", "message": f"High epoch count ({self.epochs}). Consider fewer epochs first."})

        if self.learning_rate <= 0:
            issues.append({"field": "learning_rate", "severity": "FAIL", "message": "Learning rate must be positive."})
        elif self.learning_rate > 1e-3:
            issues.append({"field": "learning_rate", "severity": "WARNING", "message": f"Very high learning rate ({self.learning_rate}). May cause instability."})
        elif self.learning_rate < 1e-6:
            issues.append({"field": "learning_rate", "severity": "WARNING", "message": f"Very low learning rate ({self.learning_rate}). Training may be slow."})

        if self.batch_size < 1:
            issues.append({"field": "batch_size", "severity": "FAIL", "message": "Batch size must be >= 1."})

        if self.gradient_accumulation_steps < 1:
            issues.append({"field": "gradient_accumulation_steps", "severity": "FAIL", "message": "Gradient accumulation steps must be >= 1."})
        elif self.gradient_accumulation_steps > 64:
            issues.append({"field": "gradient_accumulation_steps", "severity": "WARNING", "message": "Very high accumulation steps (>64). Effective batch size will be large."})

        if self.optimizer not in model_registry._optimizers:
            issues.append({"field": "optimizer", "severity": "FAIL", "message": f"Unsupported optimizer: {self.optimizer}."})

        if self.scheduler not in model_registry._schedulers:
            issues.append({"field": "scheduler", "severity": "FAIL", "message": f"Unsupported scheduler: {self.scheduler}."})

        if self.precision not in model_registry._precision_modes:
            issues.append({"field": "precision", "severity": "FAIL", "message": f"Unsupported precision: {self.precision}."})

        if not 0 <= self.warmup_ratio <= 1:
            issues.append({"field": "warmup_ratio", "severity": "FAIL", "message": "Warmup ratio must be between 0 and 1."})

        if self.weight_decay < 0:
            issues.append({"field": "weight_decay", "severity": "FAIL", "message": "Weight decay must be >= 0."})

        if self.max_sequence_length < 1:
            issues.append({"field": "max_sequence_length", "severity": "FAIL", "message": "Max sequence length must be >= 1."})

        return issues

    def effective_batch_size(self) -> int:
        """Calculate the effective batch size (batch_size × grad_accumulation)."""
        return self.batch_size * self.gradient_accumulation_steps


@dataclass
class TrainingConfig:
    """Complete training configuration combining model, LoRA, and hyperparameters."""

    project_id: str
    model_id: str
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    hyperparams: HyperparamsConfig = field(default_factory=HyperparamsConfig)
    output_dir: str = "./output"

    def validate_all(self) -> Dict[str, List[Dict[str, Any]]]:
        """Validate all components and return categorized issues."""
        issues: Dict[str, List[Dict[str, Any]]] = {
            "lora": self.lora.validate(),
            "hyperparams": self.hyperparams.validate(),
            "config": [],
        }

        # Cross-checks
        lora_has_fail = any(i["severity"] == "FAIL" for i in issues["lora"])
        hp_has_fail = any(i["severity"] == "FAIL" for i in issues["hyperparams"])

        if lora_has_fail or hp_has_fail:
            issues["config"].append({
                "field": "overall",
                "severity": "FAIL",
                "message": "Configuration has validation failures. Fix errors before proceeding.",
            })

        return issues

    @property
    def is_valid(self) -> bool:
        all_issues = self.validate_all()
        for cat in all_issues.values():
            if any(i["severity"] == "FAIL" for i in cat):
                return False
        return True
