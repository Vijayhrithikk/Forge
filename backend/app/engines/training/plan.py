"""
Training Plan Generator — produces a complete, versioned training_plan.json.

This is the output of Sprint 2. The training plan is the contract that
Sprint 3 (Training Engine) will execute. Nothing is trained here.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

from app.core import settings, get_logger
from app.engines.training.registry import model_registry
from app.engines.training.config import TrainingConfig
from app.engines.training.compatibility import CompatibilityReport
from app.engines.training.estimation import EstimationResult
from app.engines.training.scorer import ConfigurationScore
from app.engines.dataset.workspace import workspace_engine

logger = get_logger("app.engines.training.plan")


class TrainingPlanGenerator:
    """Generates a complete training plan from configuration and analysis.

    The plan is the single source of truth for Sprint 3 training execution.
    It combines model selection, LoRA config, hyperparameters, compatibility
    results, estimates, and quality scoring into a versioned JSON document.
    """

    def generate(
        self,
        config: TrainingConfig,
        compatibility: CompatibilityReport,
        estimation: EstimationResult,
        score: ConfigurationScore,
        dataset_stats: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate a complete training_plan.json document.

        Args:
            config: The validated training configuration.
            compatibility: Compatibility check results.
            estimation: VRAM/time/adapter estimates.
            score: Configuration quality score.
            dataset_stats: Optional dataset statistics from Sprint 1.

        Returns:
            Dict representing the training plan (to be saved as JSON).
        """
        model = model_registry.get_model(config.model_id)

        plan = {
            "schema_version": "1.0",
            "forge_version": settings.app_version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "ready": compatibility.all_pass and not compatibility.has_failures,

            "project": {
                "project_id": config.project_id,
            },

            "model": {
                "id": model.id,
                "name": model.name,
                "huggingface_id": model.huggingface_id,
                "architecture": model.architecture,
                "parameters": model.parameters,
                "parameters_display": model.parameters_display,
                "context_length": model.context_length,
                "tokenizer": model.tokenizer,
                "recommended_precision": model.recommended_precision,
                "recommended_gpu": model.recommended_gpu,
                "supports_lora": model.supports_lora,
            },

            "dataset": dataset_stats or {},

            "lora": {
                "rank": config.lora.rank,
                "alpha": config.lora.alpha,
                "dropout": config.lora.dropout,
                "bias": config.lora.bias,
                "target_modules": config.lora.target_modules or model.lora_defaults.get("target_modules", []),
                "task_type": config.lora.task_type,
            },

            "hyperparameters": {
                "epochs": config.hyperparams.epochs,
                "learning_rate": config.hyperparams.learning_rate,
                "batch_size": config.hyperparams.batch_size,
                "gradient_accumulation_steps": config.hyperparams.gradient_accumulation_steps,
                "effective_batch_size": config.hyperparams.effective_batch_size(),
                "optimizer": config.hyperparams.optimizer,
                "scheduler": config.hyperparams.scheduler,
                "warmup_ratio": config.hyperparams.warmup_ratio,
                "weight_decay": config.hyperparams.weight_decay,
                "max_sequence_length": config.hyperparams.max_sequence_length,
                "seed": config.hyperparams.seed,
                "precision": config.hyperparams.precision,
                "logging_steps": config.hyperparams.logging_steps,
                "save_steps": config.hyperparams.save_steps,
            },

            "compatibility": {
                "all_pass": compatibility.all_pass,
                "has_failures": compatibility.has_failures,
                "passed": compatibility.passed,
                "warnings": compatibility.warnings,
                "failed": compatibility.failed,
                "checks": [
                    {
                        "check": r.check,
                        "status": r.status,
                        "message": r.message,
                        "details": r.details,
                    }
                    for r in compatibility.results
                ],
            },

            "estimation": {
                "total_vram_gb": estimation.total_vram_gb,
                "model_memory_gb": estimation.model_memory_gb,
                "lora_memory_gb": estimation.lora_memory_gb,
                "optimizer_memory_gb": estimation.optimizer_memory_gb,
                "activation_memory_gb": estimation.activation_memory_gb,
                "safety_buffer_gb": estimation.safety_buffer_gb,
                "trainable_parameters": estimation.trainable_parameters,
                "adapter_size_mb": estimation.adapter_size_mb,
                "estimated_steps": estimation.estimated_steps,
                "estimated_duration_minutes": estimation.estimated_duration_minutes,
                "estimated_duration_display": estimation.estimated_duration_display,
                "assumptions": estimation.assumptions,
            },

            "quality": {
                "score": score.score,
                "grade": score.grade,
                "strengths": score.strengths,
                "warnings": score.warnings,
                "suggestions": score.suggestions,
                "deductions": score.deductions,
            },
        }

        return plan

    def save(
        self,
        plan: Dict[str, Any],
        project_id: str,
    ) -> Path:
        """Persist the training plan to the project's reports directory.

        Args:
            plan: The generated training plan dict.
            project_id: Target project identifier.

        Returns:
            Path to the saved training_plan.json file.
        """
        reports_dir = workspace_engine.get_reports_path(project_id)
        reports_dir.mkdir(parents=True, exist_ok=True)
        plan_path = reports_dir / "training_plan.json"

        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(plan, f, indent=2, ensure_ascii=False)

        logger.info("training_plan_saved", project_id=project_id, path=str(plan_path))
        return plan_path

    def export_markdown(self, plan: Dict[str, Any]) -> str:
        """Export the training plan as a human-readable Markdown report.

        Args:
            plan: The generated training plan dict.

        Returns:
            Markdown string suitable for display or download.
        """
        m = plan["model"]
        l = plan["lora"]
        h = plan["hyperparameters"]
        e = plan["estimation"]
        q = plan["quality"]
        c = plan["compatibility"]

        return f"""# Forge Training Plan

**Project:** {plan['project']['project_id']}
**Created:** {plan['created_at']}
**Status:** {'✅ Ready to Train' if plan['ready'] else '⚠️ Issues Detected'}

---

## Model

| Property | Value |
|----------|-------|
| Name | {m['name']} |
| HF Repository | `{m['huggingface_id']}` |
| Architecture | {m['architecture']} |
| Parameters | {m['parameters_display']} |
| Context Window | {m['context_length']:,} tokens |
| Tokenizer | {m['tokenizer']} |
| Recommended GPU | {m['recommended_gpu']} |

---

## LoRA Configuration

| Parameter | Value |
|-----------|-------|
| Rank (r) | {l['rank']} |
| Alpha | {l['alpha']} |
| Dropout | {l['dropout']} |
| Bias | {l['bias']} |
| Target Modules | {', '.join(l['target_modules'])} |

---

## Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Epochs | {h['epochs']} |
| Learning Rate | {h['learning_rate']} |
| Batch Size | {h['batch_size']} |
| Gradient Accumulation | {h['gradient_accumulation_steps']} |
| Effective Batch Size | {h['effective_batch_size']} |
| Optimizer | {h['optimizer']} |
| Scheduler | {h['scheduler']} |
| Warmup Ratio | {h['warmup_ratio']} |
| Weight Decay | {h['weight_decay']} |
| Max Sequence Length | {h['max_sequence_length']} |
| Precision | {h['precision']} |
| Seed | {h['seed']} |

---

## Estimates

| Metric | Value |
|--------|-------|
| Total VRAM | {e['total_vram_gb']:.1f} GB |
| Model Memory | {e['model_memory_gb']:.1f} GB |
| LoRA Memory | {e['lora_memory_gb']:.1f} GB |
| Optimizer Memory | {e['optimizer_memory_gb']:.1f} GB |
| Adapter Size | {e['adapter_size_mb']:.1f} MB |
| Trainable Params | {e['trainable_parameters']:,} |
| Estimated Duration | {e['estimated_duration_display']} |

### Assumptions

{chr(10).join('- ' + a for a in e['assumptions'])}

---

## Compatibility

| Check | Status |
|-------|--------|
{chr(10).join(f"| {r['check']} | {r['status']} |" for r in c['checks'])}

---

## Configuration Score

**{q['score']} / 100 — {q['grade']}**

### Strengths
{chr(10).join('- ' + s for s in q['strengths'])}

### Warnings
{chr(10).join('- ' + w for w in q['warnings'])}

### Suggestions
{chr(10).join('- ' + s for s in q['suggestions'])}

---

*Generated by Forge v{plan['forge_version']} — Plan Schema v{plan['schema_version']}*
"""


# Singleton
plan_generator = TrainingPlanGenerator()
