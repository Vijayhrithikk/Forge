"""
Model Browser API — list, search, filter, and inspect supported models.

Reads from the filesystem model registry. No model downloading.
"""

from typing import Optional, List

from fastapi import APIRouter, Query, HTTPException

from app.core import get_logger
from app.schemas.responses import SuccessResponse
from app.engines.training.registry import model_registry

router = APIRouter(prefix="/models", tags=["models"])
logger = get_logger("app.api.models")


@router.get("", response_model=SuccessResponse)
async def list_models(
    family: Optional[str] = Query(None, description="Filter by model family (llama, qwen2, mistral, gemma, phi)"),
    tags: Optional[str] = Query(None, description="Comma-separated tags (e.g. 'recommended,lightweight')"),
    search: Optional[str] = Query(None, description="Free-text search across name and architecture"),
    max_parameters: Optional[int] = Query(None, description="Maximum parameter count filter"),
    min_context: Optional[int] = Query(None, description="Minimum context length filter"),
):
    """List supported models with optional filtering.

    Returns all models from the registry matching the given criteria.
    """
    tag_list = [t.strip() for t in tags.split(",")] if tags else None

    models = model_registry.list_models(
        family=family,
        tags=tag_list,
        search=search,
        max_parameters=max_parameters,
        min_context=min_context,
    )

    return SuccessResponse(
        message=f"{len(models)} model(s) found.",
        data={
            "models": [
                {
                    "id": m.id,
                    "name": m.name,
                    "huggingface_id": m.huggingface_id,
                    "architecture": m.architecture,
                    "family": m.family,
                    "parameters": m.parameters,
                    "parameters_display": m.parameters_display,
                    "context_length": m.context_length,
                    "tokenizer": m.tokenizer,
                    "precision": m.precision,
                    "recommended_precision": m.recommended_precision,
                    "recommended_vram_gb": m.recommended_vram_gb,
                    "recommended_gpu": m.recommended_gpu,
                    "license": m.license,
                    "instruction_tuned": m.instruction_tuned,
                    "supports_lora": m.supports_lora,
                    "lora_defaults": m.lora_defaults,
                    "training_notes": m.training_notes,
                    "tags": m.tags,
                }
                for m in models
            ],
        },
    )


@router.get("/{model_id}", response_model=SuccessResponse)
async def get_model(model_id: str):
    """Get detailed information about a specific model.

    Returns full metadata including LoRA defaults, training notes,
    and recommended configuration.
    """
    try:
        model = model_registry.get_model(model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

    return SuccessResponse(
        message=f"Model: {model.name}",
        data={
            "id": model.id,
            "name": model.name,
            "huggingface_id": model.huggingface_id,
            "architecture": model.architecture,
            "family": model.family,
            "parameters": model.parameters,
            "parameters_display": model.parameters_display,
            "context_length": model.context_length,
            "tokenizer": model.tokenizer,
            "precision": model.precision,
            "recommended_precision": model.recommended_precision,
            "recommended_vram_gb": model.recommended_vram_gb,
            "recommended_gpu": model.recommended_gpu,
            "license": model.license,
            "instruction_tuned": model.instruction_tuned,
            "supports_lora": model.supports_lora,
            "lora_defaults": model.lora_defaults,
            "training_notes": model.training_notes,
            "tags": model.tags,
        },
    )


@router.get("/meta/optimizers", response_model=SuccessResponse)
async def list_optimizers():
    """List supported optimizers with descriptions."""
    optimizers = [
        {
            "id": opt_id,
            "name": o.name,
            "description": o.description,
            "advantages": o.advantages,
            "disadvantages": o.disadvantages,
            "recommended_use": o.recommended_use,
        }
        for opt_id, o in model_registry._optimizers.items()
    ]
    return SuccessResponse(message="Optimizers loaded.", data={"optimizers": optimizers})


@router.get("/meta/schedulers", response_model=SuccessResponse)
async def list_schedulers():
    """List supported schedulers with descriptions."""
    schedulers = [
        {
            "id": sched_id,
            "name": s.name,
            "description": s.description,
            "use_cases": s.use_cases,
            "recommendation": s.recommendation,
        }
        for sched_id, s in model_registry._schedulers.items()
    ]
    return SuccessResponse(message="Schedulers loaded.", data={"schedulers": schedulers})


@router.get("/meta/precisions", response_model=SuccessResponse)
async def list_precisions():
    """List supported precision modes."""
    precisions = [
        {
            "id": prec_id,
            "name": p.name,
            "description": p.description,
            "vram_factor": p.vram_factor,
            "compatibility": p.compatibility,
            "notes": p.notes,
        }
        for prec_id, p in model_registry._precision_modes.items()
    ]
    return SuccessResponse(message="Precision modes loaded.", data={"precisions": precisions})


@router.get("/meta/gpus", response_model=SuccessResponse)
async def list_gpus():
    """List GPU compatibility information."""
    gpus = [
        {
            "id": gpu_id,
            "name": g.name,
            "vram_gb": g.vram_gb,
            "cuda_cores": g.cuda_cores,
            "compatibility": g.compatibility,
            "notes": g.notes,
        }
        for gpu_id, g in model_registry._gpus.items()
    ]
    return SuccessResponse(message="GPU info loaded.", data={"gpus": gpus})
