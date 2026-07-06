"""
Configuration Scoring Engine — evaluates training configuration quality (0–100).

Scores based on safe defaults, model compatibility, dataset fit,
parameter balance, and hardware suitability. Every deduction is explained.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any

from app.engines.training.config import TrainingConfig
from app.engines.training.compatibility import CompatibilityReport
from app.engines.training.estimation import EstimationResult
from app.engines.training.registry import model_registry


@dataclass
class ConfigurationScore:
    """Quality score with explanations for every deduction."""

    score: int                                         # 0–100
    grade: str                                         # Excellent / Good / Needs Improvement / Poor
    strengths: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    deductions: List[Dict[str, Any]] = field(default_factory=list)


class ConfigurationScorer:
    """Scores a training configuration on a 0–100 scale.

    Scoring methodology (documented):
    - Safe defaults:           25 points
    - Model compatibility:     25 points
    - Dataset fit:             20 points
    - Parameter balance:       15 points
    - Hardware suitability:    15 points
    """

    MAX_SCORE = 100

    def score(
        self,
        config: TrainingConfig,
        compatibility: CompatibilityReport,
        estimation: EstimationResult,
    ) -> ConfigurationScore:
        """Score the configuration and explain every deduction.

        Args:
            config: The training configuration.
            compatibility: Compatibility check results.
            estimation: VRAM/time/adapter estimates.

        Returns:
            ConfigurationScore with detailed explanations.
        """
        strengths: List[str] = []
        warnings_list: List[str] = []
        suggestions: List[str] = []
        deductions: List[Dict[str, Any]] = []
        total = self.MAX_SCORE

        # ---- 1. Safe Defaults (25 pts) ----
        hp = config.hyperparams
        defaults_score = 25

        if hp.epochs > 5:
            deductions.append({"factor": "Safe Defaults", "deduction": -3, "reason": f"High epoch count ({hp.epochs}). Start with fewer epochs."})
            defaults_score -= 3
        if hp.learning_rate > 5e-4:
            deductions.append({"factor": "Safe Defaults", "deduction": -3, "reason": f"High learning rate ({hp.learning_rate}). May cause instability."})
            defaults_score -= 3
        if hp.learning_rate < 1e-5:
            deductions.append({"factor": "Safe Defaults", "deduction": -2, "reason": f"Very low learning rate ({hp.learning_rate}). Training may be slow."})
            defaults_score -= 2
        if hp.precision == "fp32":
            deductions.append({"factor": "Safe Defaults", "deduction": -4, "reason": "FP32 uses 2× memory with minimal quality benefit for LoRA."})
            defaults_score -= 4
        if hp.batch_size * hp.gradient_accumulation_steps > 128:
            deductions.append({"factor": "Safe Defaults", "deduction": -2, "reason": f"Large effective batch size ({hp.effective_batch_size()})."})
            defaults_score -= 2
        if config.lora.rank > 64:
            deductions.append({"factor": "Safe Defaults", "deduction": -2, "reason": f"High LoRA rank ({config.lora.rank}). Diminishing returns above 64."})
            defaults_score -= 2

        total -= (25 - defaults_score)
        if defaults_score >= 23:
            strengths.append("Configuration uses safe, recommended defaults.")
        elif defaults_score < 15:
            warnings_list.append("Several parameters deviate significantly from recommended defaults.")

        # ---- 2. Model Compatibility (25 pts) ----
        compat_score = 25
        if compatibility.has_failures:
            deductions.append({"factor": "Model Compatibility", "deduction": -25, "reason": "Compatibility failures detected. Configuration cannot proceed."})
            compat_score = 0
            warnings_list.append("Compatibility failures must be resolved before training.")
        elif compatibility.warnings > 2:
            deductions.append({"factor": "Model Compatibility", "deduction": -5, "reason": f"{compatibility.warnings} compatibility warnings."})
            compat_score = 20
        elif compatibility.all_pass:
            strengths.append("All compatibility checks passed.")

        total -= (25 - compat_score)

        # ---- 3. Dataset Fit (20 pts) ----
        dataset_score = 20
        if config.lora.alpha < config.lora.rank:
            deductions.append({"factor": "Dataset Fit", "deduction": -3, "reason": "LoRA alpha < rank. Alpha should typically be ≥ rank."})
            dataset_score -= 3
        if hp.max_sequence_length > 4096:
            deductions.append({"factor": "Dataset Fit", "deduction": -2, "reason": f"Long max sequence ({hp.max_sequence_length}). Ensure dataset needs this."})
            dataset_score -= 2

        total -= (20 - dataset_score)
        if dataset_score >= 18:
            strengths.append("Dataset and sequence configuration look appropriate.")

        # ---- 4. Parameter Balance (15 pts) ----
        balance_score = 15
        if config.lora.dropout > 0.2:
            deductions.append({"factor": "Parameter Balance", "deduction": -3, "reason": f"High dropout ({config.lora.dropout}). May degrade training."})
            balance_score -= 3
        if hp.warmup_ratio > 0.2:
            deductions.append({"factor": "Parameter Balance", "deduction": -2, "reason": f"High warmup ratio ({hp.warmup_ratio}). Typically 0.03-0.1 is sufficient."})
            balance_score -= 2
        if hp.weight_decay > 0.1:
            deductions.append({"factor": "Parameter Balance", "deduction": -2, "reason": f"High weight decay ({hp.weight_decay}). May limit learning."})
            balance_score -= 2

        total -= (15 - balance_score)

        # ---- 5. Hardware Suitability (15 pts) ----
        hardware_score = 15
        try:
            model = model_registry.get_model(config.model_id)
            if estimation.total_vram_gb > model.recommended_vram_gb * 1.5:
                deductions.append({"factor": "Hardware Suitability", "deduction": -5, "reason": f"Estimated VRAM ({estimation.total_vram_gb:.1f}GB) exceeds recommended ({model.recommended_vram_gb}GB)."})
                hardware_score -= 5
            elif estimation.total_vram_gb > model.recommended_vram_gb:
                deductions.append({"factor": "Hardware Suitability", "deduction": -2, "reason": "Estimated VRAM is slightly above the recommended level."})
                hardware_score -= 2
            else:
                strengths.append(f"VRAM estimate ({estimation.total_vram_gb:.1f}GB) fits within recommended range.")
        except KeyError:
            pass

        total -= (15 - hardware_score)

        # Clamp
        total = max(0, min(100, total))

        # Grade
        if total >= 90:
            grade = "Excellent"
        elif total >= 75:
            grade = "Good"
        elif total >= 50:
            grade = "Needs Improvement"
        else:
            grade = "Poor"

        # Generate suggestions from deductions
        for d in deductions:
            suggestions.append(f"{d['factor']}: {d['reason']}")

        return ConfigurationScore(
            score=total,
            grade=grade,
            strengths=strengths,
            warnings=warnings_list,
            suggestions=suggestions,
            deductions=deductions,
        )


# Singleton
config_scorer = ConfigurationScorer()
