"""
Model Runtime API — acquire, load, and prepare models for training.

Mission B endpoints. Integrates Acquisition + Loader + Preparation
into the Runtime Coordinator lifecycle.
"""

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.core import get_logger
from app.schemas.responses import SuccessResponse
from app.engines.dataset.workspace import workspace_engine
from app.acquisition.resolver import registry_resolver
from app.acquisition.downloader import create_download_manager
from app.acquisition.verifier import integrity_verifier
from app.acquisition.cache import cache_manager
from app.acquisition.loader import model_loader
from app.acquisition.manifest import (
    generate_model_manifest, generate_tokenizer_manifest, save_manifest,
)
from app.preparation.device import device_engine
from app.preparation.precision import precision_engine
from app.preparation.memory import memory_engine
from app.preparation.optimization import optimization_engine
from app.preparation.prepared_model import PreparedModel

router = APIRouter(prefix="/runtime", tags=["model-runtime"])
logger = get_logger("app.api.model_runtime")

# In-memory store of prepared models (keyed by project_id)
_prepared_models: dict[str, PreparedModel] = {}


@router.post("/acquire", response_model=SuccessResponse)
async def acquire_model(
    project_id: str = Query(...),
    model_id: Optional[str] = Query(None, description="Override model from training plan"),
    revision: Optional[str] = Query("main"),
):
    """Acquire model assets: resolve -> download -> verify -> cache.

    Downloads model weights, config, and tokenizer from HuggingFace
    into the local cache. Resumable and retryable.
    """
    if not model_id:
        # Try to read from training plan
        try:
            plan_path = workspace_engine.get_reports_path(project_id) / "training_plan.json"
            if plan_path.exists():
                import json
                with open(plan_path) as f:
                    plan = json.load(f)
                model_id = plan.get("model", {}).get("id")
        except Exception:
            pass

    if not model_id:
        raise HTTPException(status_code=400, detail="No model specified. Provide model_id or generate a training plan first.")

    # 1. Resolve
    asset = registry_resolver.resolve(model_id, revision)

    # 2. Download
    dl_manager = create_download_manager()
    cache_dir = dl_manager.download(asset)

    # 3. Verify
    hashes = integrity_verifier.verify(cache_dir, asset)

    # 4. Cache manifest
    asset_meta = {
        "model_id": asset.model_id,
        "huggingface_id": asset.huggingface_id,
        "revision": asset.revision,
        "architecture": asset.architecture,
    }
    cache_manager.write_cache_manifest(model_id, revision or "main", hashes, asset_meta)

    # 5. Generate manifests
    model_manifest = generate_model_manifest(asset, cache_dir, hashes)
    tokenizer_manifest = generate_tokenizer_manifest(asset, cache_dir)
    save_manifest(model_manifest, cache_dir / "metadata" / "model_manifest.json")
    save_manifest(tokenizer_manifest, cache_dir / "metadata" / "tokenizer_manifest.json")

    return SuccessResponse(
        message=f"Model '{model_id}' acquired and verified.",
        data={
            "model_id": model_id,
            "revision": revision,
            "cache_path": str(cache_dir),
            "files_verified": len(hashes),
        },
    )


@router.post("/load", response_model=SuccessResponse)
async def load_model(
    project_id: str = Query(...),
    model_id: Optional[str] = Query(None),
    device: str = Query("auto"),
    precision: str = Query("bf16"),
):
    """Load a model and tokenizer from the verified cache.

    Requires the model to be acquired first (POST /runtime/acquire).
    """
    if not model_id:
        try:
            plan_path = workspace_engine.get_reports_path(project_id) / "training_plan.json"
            if plan_path.exists():
                import json
                with open(plan_path) as f:
                    plan = json.load(f)
                model_id = plan.get("model", {}).get("id")
        except Exception:
            pass

    if not model_id:
        raise HTTPException(status_code=400, detail="No model specified.")

    if not cache_manager.is_cached(model_id):
        raise HTTPException(status_code=400, detail=f"Model '{model_id}' not cached. Run /runtime/acquire first.")

    asset = registry_resolver.resolve(model_id)
    result = model_loader.load(asset, device=device, precision=precision)

    # Prepare the model
    dev = device_engine.select_device(device)
    prec = precision_engine.select_precision(result.model_manifest.get("model", {}).get("recommended_precision", precision) if isinstance(result.model_manifest.get("model"), dict) else precision, dev, precision)
    mem = memory_engine.validate(dev, 4.0)  # Conservative estimate
    opt = optimization_engine.configure(dev)

    prepared = PreparedModel(
        model=result.model,
        tokenizer=result.tokenizer,
        config=result.config,
        device=dev,
        precision=prec,
        memory=mem,
        optimization=opt,
        model_id=model_id,
        runtime_id="",
        validation_results=result.validation_results,
        model_manifest=result.model_manifest,
        tokenizer_manifest=result.tokenizer_manifest,
    )

    # Save preparation manifest
    runtime_dir = workspace_engine.get_project_path(project_id) / "runtime"
    prepared.save_manifest(runtime_dir / "manifest")

    _prepared_models[project_id] = prepared

    return SuccessResponse(
        message=f"Model '{model_id}' loaded and prepared.",
        data={
            "model_id": model_id,
            "device": dev.name,
            "precision": prec.precision,
            "validation": result.validation_results,
            "load_duration": result.load_duration,
        },
    )


@router.post("/prepare-model", response_model=SuccessResponse)
async def prepare_model(
    project_id: str = Query(...),
):
    """Validate preparation — device, precision, memory, optimization.

    Does not load the model. Assumes the model is already loaded.
    """
    prepared = _prepared_models.get(project_id)
    if not prepared:
        raise HTTPException(status_code=404, detail="No loaded model. Run /runtime/load first.")

    dev = prepared.device
    prec = prepared.precision

    supported_precisions = precision_engine.list_supported(dev)
    return SuccessResponse(
        message="Model preparation validated.",
        data={
            "device": {"name": dev.name, "type": dev.device_type},
            "precision": {"selected": prec.precision, "supported": supported_precisions},
            "memory": {"total_gb": prepared.memory.total_gb, "free_gb": prepared.memory.free_gb, "status": prepared.memory.status},
            "optimization": prepared.optimization.flags,
            "optimization_notes": prepared.optimization.notes,
            "ready": prepared.is_ready,
        },
    )


@router.get("/model", response_model=SuccessResponse)
async def get_model_status(project_id: str = Query(...)):
    """Get the current model runtime status."""
    prepared = _prepared_models.get(project_id)
    if not prepared:
        return SuccessResponse(message="No model loaded.", data={"status": "not_loaded"})

    return SuccessResponse(
        message="Model status.",
        data={
            "status": "ready" if prepared.is_ready else "loading",
            "model_id": prepared.model_id,
            "device": prepared.device.name,
            "precision": prepared.precision.precision,
            "prepared_at": prepared.prepared_at,
            "manifest": prepared.generate_preparation_manifest(),
        },
    )


@router.get("/cache", response_model=SuccessResponse)
async def get_cache_status():
    """Get the current model cache status."""
    cached = cache_manager.list_cached_models()
    return SuccessResponse(
        message=f"{len(cached)} model(s) cached.",
        data={
            "cached_models": cached,
            "cache_root": str(cache_manager.root),
            "total_size_bytes": cache_manager.total_cache_size(),
        },
    )
