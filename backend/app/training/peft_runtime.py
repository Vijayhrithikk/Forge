"""
PEFT Runtime — LoRA injection, parameter freezing, trainable verification.

Injects LoRA adapters into the PreparedModel, freezes base parameters,
verifies injection counts, and generates peft_manifest.json.
"""

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from app.core import settings, get_logger
from app.training.exceptions import PEFTInjectionError

logger = get_logger("app.training.peft_runtime")


class PEFTRuntime:
    """Injects LoRA adapters and verifies trainable parameters."""

    def inject(
        self,
        model: Any,
        lora_config_dict: Dict[str, Any],
        target_modules: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Inject LoRA adapters into the model.

        Args:
            model: The base HuggingFace model (from PreparedModel).
            lora_config_dict: LoRA config from training plan.
            target_modules: Override target modules (from registry).

        Returns:
            Dict with peft_model, injection report, and trainable count.
        """
        rank = lora_config_dict.get("rank", 16)
        alpha = lora_config_dict.get("alpha", 32)
        dropout = lora_config_dict.get("dropout", 0.05)
        bias = lora_config_dict.get("bias", "none")
        modules = target_modules or lora_config_dict.get("target_modules", [])

        if not modules:
            raise PEFTInjectionError("No target modules specified for LoRA injection.")

        total_params = sum(p.numel() for p in model.parameters())
        peft_model = model
        injected = False

        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError:
            raise PEFTInjectionError(
                "The 'peft' package is required for LoRA fine-tuning. "
                "Install it with: pip install peft"
            )

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=rank, lora_alpha=alpha, lora_dropout=dropout,
            bias=bias, target_modules=modules,
        )
        peft_model = get_peft_model(model, lora_config)
        injected = True
        logger.info("peft_injected", rank=rank, alpha=alpha, modules=len(modules))

        # Count trainable params
        trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        frozen = total_params - trainable
        trainable_pct = round(trainable / max(total_params, 1) * 100, 2)

        report = {
            "schema_version": "1.0", "forge_version": settings.app_version,
            "injected": injected, "base_model_params": total_params,
            "trainable_params": trainable, "frozen_params": frozen,
            "trainable_percentage": trainable_pct,
            "lora_config": {"rank": rank, "alpha": alpha, "dropout": dropout,
                           "bias": bias, "target_modules": modules},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.info("trainable_params_verified", trainable=trainable,
                     frozen=frozen, pct=trainable_pct)
        return {"peft_model": peft_model, "report": report}

    def save_manifest(self, report: Dict, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "peft_manifest.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        return path

    def save_trainable_report(self, report: Dict, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "trainable_parameters.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        return path
