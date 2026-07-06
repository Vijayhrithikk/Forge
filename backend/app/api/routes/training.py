"""
Training Configuration API — validate, estimate, plan, and export.

All configuration, no training. Sprint 2 produces training_plan.json.
Sprint 3 will execute it.
"""

import json
from typing import Optional, List
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse

from app.core import get_logger
from app.schemas.responses import SuccessResponse
from app.engines.training.registry import model_registry
from app.engines.training.config import TrainingConfig, LoRAConfig, HyperparamsConfig
from app.engines.training.compatibility import compatibility_engine
from app.engines.training.estimation import estimation_engine
from app.engines.training.scorer import config_scorer
from app.engines.training.plan import plan_generator
from app.engines.dataset.workspace import workspace_engine

router = APIRouter(prefix="/training", tags=["training"])
logger = get_logger("app.api.training")


@router.post("/config/validate", response_model=SuccessResponse)
async def validate_config(
    project_id: str = Query(...),
    model_id: str = Query(...),
    # LoRA params
    lora_rank: int = Query(16, ge=1, le=256),
    lora_alpha: int = Query(32, ge=1, le=512),
    lora_dropout: float = Query(0.05, ge=0, le=1),
    lora_bias: str = Query("none"),
    target_modules: Optional[str] = Query(None, description="Comma-separated target modules"),
    # Hyperparams
    epochs: int = Query(3, ge=1),
    learning_rate: float = Query(2e-4, gt=0),
    batch_size: int = Query(4, ge=1),
    gradient_accumulation: int = Query(1, ge=1),
    optimizer: str = Query("adamw"),
    scheduler: str = Query("linear"),
    warmup_ratio: float = Query(0.03, ge=0, le=1),
    weight_decay: float = Query(0.01, ge=0),
    max_sequence_length: int = Query(2048, ge=1),
    seed: int = Query(42),
    precision: str = Query("bf16"),
):
    """Validate a complete training configuration.

    Checks all parameters, cross-checks with model and dataset,
    and returns validation issues, estimates, and quality score.
    """
    # Build config
    lora = LoRAConfig(
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
        bias=lora_bias,
        target_modules=[m.strip() for m in target_modules.split(",")] if target_modules else [],
    )

    hyperparams = HyperparamsConfig(
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        optimizer=optimizer,
        scheduler=scheduler,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        max_sequence_length=max_sequence_length,
        seed=seed,
        precision=precision,
    )

    config = TrainingConfig(
        project_id=project_id,
        model_id=model_id,
        lora=lora,
        hyperparams=hyperparams,
    )

    # Validate
    validation_issues = config.validate_all()

    # Load dataset stats if available
    dataset_stats = None
    ds_sample_count = 0
    ds_avg_tokens = 0.0
    try:
        meta_path = workspace_engine.get_metadata_path(project_id) / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                dataset_stats = json.load(f)
            ds_sample_count = dataset_stats.get("records", 0)
            ds_avg_tokens = dataset_stats.get("average_tokens", 0.0)
    except Exception:
        pass

    # Compatibility
    compat = compatibility_engine.check(config, ds_sample_count, ds_avg_tokens)

    # Estimation
    estimation = estimation_engine.estimate(config, ds_sample_count, ds_avg_tokens)

    # Score
    score = config_scorer.score(config, compat, estimation)

    return SuccessResponse(
        message=f"Configuration validated. Score: {score.score}/100 ({score.grade}).",
        data={
            "validation": {
                "lora_issues": validation_issues["lora"],
                "hyperparams_issues": validation_issues["hyperparams"],
                "config_issues": validation_issues["config"],
                "is_valid": config.is_valid,
            },
            "compatibility": {
                "all_pass": compat.all_pass,
                "checks": [
                    {"check": r.check, "status": r.status, "message": r.message, "details": r.details}
                    for r in compat.results
                ],
            },
            "estimation": {
                "model_memory_gb": estimation.model_memory_gb,
                "lora_memory_gb": estimation.lora_memory_gb,
                "optimizer_memory_gb": estimation.optimizer_memory_gb,
                "activation_memory_gb": estimation.activation_memory_gb,
                "safety_buffer_gb": estimation.safety_buffer_gb,
                "total_vram_gb": estimation.total_vram_gb,
                "trainable_parameters": estimation.trainable_parameters,
                "adapter_size_mb": estimation.adapter_size_mb,
                "estimated_steps": estimation.estimated_steps,
                "estimated_duration_minutes": estimation.estimated_duration_minutes,
                "estimated_duration_display": estimation.estimated_duration_display,
                "assumptions": estimation.assumptions,
            },
            "score": {
                "score": score.score,
                "grade": score.grade,
                "strengths": score.strengths,
                "warnings": score.warnings,
                "suggestions": score.suggestions,
                "deductions": score.deductions,
            },
            "compatible_gpus": estimation_engine.compatible_gpus(estimation.total_vram_gb),
        },
    )


@router.post("/config/plan", response_model=SuccessResponse)
async def generate_plan(
    project_id: str = Query(...),
    model_id: str = Query(...),
    # LoRA
    lora_rank: int = Query(16), lora_alpha: int = Query(32),
    lora_dropout: float = Query(0.05), lora_bias: str = Query("none"),
    target_modules: Optional[str] = Query(None),
    # Hyperparams
    epochs: int = Query(3), learning_rate: float = Query(2e-4),
    batch_size: int = Query(4), gradient_accumulation: int = Query(1),
    optimizer: str = Query("adamw"), scheduler: str = Query("linear"),
    warmup_ratio: float = Query(0.03), weight_decay: float = Query(0.01),
    max_sequence_length: int = Query(2048), seed: int = Query(42),
    precision: str = Query("bf16"),
):
    """Generate and save a complete training plan.

    Combines model selection, LoRA config, hyperparameters, compatibility,
    estimates, and quality score into a versioned training_plan.json.
    """
    lora = LoRAConfig(
        rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout, bias=lora_bias,
        target_modules=[m.strip() for m in target_modules.split(",")] if target_modules else [],
    )
    hyperparams = HyperparamsConfig(
        epochs=epochs, learning_rate=learning_rate, batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        optimizer=optimizer, scheduler=scheduler,
        warmup_ratio=warmup_ratio, weight_decay=weight_decay,
        max_sequence_length=max_sequence_length, seed=seed, precision=precision,
    )
    config = TrainingConfig(project_id=project_id, model_id=model_id, lora=lora, hyperparams=hyperparams)

    # Dataset stats
    dataset_stats = None
    ds_sample_count = 0
    ds_avg_tokens = 0.0
    try:
        meta_path = workspace_engine.get_metadata_path(project_id) / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                dataset_stats = json.load(f)
            ds_sample_count = dataset_stats.get("records", 0)
            ds_avg_tokens = dataset_stats.get("average_tokens", 0.0)
    except Exception:
        pass

    compat = compatibility_engine.check(config, ds_sample_count, ds_avg_tokens)
    estimation = estimation_engine.estimate(config, ds_sample_count, ds_avg_tokens)
    score = config_scorer.score(config, compat, estimation)
    plan = plan_generator.generate(config, compat, estimation, score, dataset_stats)
    saved_path = plan_generator.save(plan, project_id)

    logger.info("training_plan_generated", project_id=project_id, model=model_id)

    return SuccessResponse(
        message=f"Training plan generated and saved to {saved_path}.",
        data={"plan": plan, "path": str(saved_path)},
    )


@router.get("/config/plan", response_model=SuccessResponse)
async def get_plan(project_id: str = Query(...)):
    """Retrieve a previously saved training plan."""
    plan_path = workspace_engine.get_reports_path(project_id) / "training_plan.json"
    if not plan_path.exists():
        raise HTTPException(status_code=404, detail="No training plan found. Generate one first.")

    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)

    return SuccessResponse(message="Training plan loaded.", data={"plan": plan})


@router.get("/config/plan/export", response_class=PlainTextResponse)
async def export_plan_markdown(project_id: str = Query(...)):
    """Export the training plan as a human-readable Markdown report."""
    plan_path = workspace_engine.get_reports_path(project_id) / "training_plan.json"
    if not plan_path.exists():
        raise HTTPException(status_code=404, detail="No training plan found.")

    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)

    md = plan_generator.export_markdown(plan)
    return PlainTextResponse(content=md, media_type="text/markdown")


@router.get("/config/plan/export/json", response_model=SuccessResponse)
async def export_plan_json(project_id: str = Query(...)):
    """Download the training plan as JSON."""
    plan_path = workspace_engine.get_reports_path(project_id) / "training_plan.json"
    if not plan_path.exists():
        raise HTTPException(status_code=404, detail="No training plan found.")

    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)

    return SuccessResponse(message="Training plan ready for download.", data=plan)
