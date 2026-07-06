"""
Runtime API — Typed, validated endpoints for the Execution Runtime.

No training happens here. These endpoints manage the Runtime lifecycle:
create, prepare, cancel, and query status/environment/events/metrics.
"""

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.core import get_logger
from app.schemas.responses import SuccessResponse
from app.engines.dataset.workspace import workspace_engine
from app.runtime.runtime import RuntimeCoordinator

router = APIRouter(prefix="/runtime", tags=["runtime"])
logger = get_logger("app.api.runtime")

# In-memory store of active runtimes (stateless API, runtime is filesystem-backed)
_active_runtimes: dict[str, RuntimeCoordinator] = {}


def _get_or_create_runtime(project_id: str) -> RuntimeCoordinator:
    """Get an existing runtime or create a new one for the project.

    In production, this would be managed by a proper runtime manager.
    For Version 1, we track active runtimes in memory.
    """
    if project_id in _active_runtimes:
        return _active_runtimes[project_id]

    try:
        project_path = workspace_engine.get_project_path(project_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")

    coordinator = RuntimeCoordinator(project_path, project_id)
    _active_runtimes[project_id] = coordinator
    return coordinator


# ------------------------------------------------------------------
# Lifecycle endpoints
# ------------------------------------------------------------------

@router.post("/create", response_model=SuccessResponse)
async def create_runtime(project_id: str = Query(...)):
    """Create a new Runtime instance for the project.

    Initializes state machine, manifest, and emits RUNTIME_CREATED.
    This does NOT start preparation — call /prepare for that.
    """
    coordinator = _get_or_create_runtime(project_id)
    result = coordinator.create()
    return SuccessResponse(message="Runtime created.", data=result)


@router.post("/prepare", response_model=SuccessResponse)
async def prepare_runtime(project_id: str = Query(...)):
    """Execute the full preparation pipeline.

    Acquires lock → prepares workspace → validates environment →
    discovers GPU → validates GPU → generates reports → READY.
    """
    coordinator = _get_or_create_runtime(project_id)
    result = coordinator.prepare()
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("message", "Preparation failed."))
    return SuccessResponse(message="Runtime ready.", data=result)


@router.post("/cancel", response_model=SuccessResponse)
async def cancel_runtime(project_id: str = Query(...)):
    """Cancel the active Runtime.

    Releases the execution lock, cleans temporary files,
    and transitions to CANCELLED state.
    """
    coordinator = _get_or_create_runtime(project_id)
    result = coordinator.cancel()
    _active_runtimes.pop(project_id, None)
    return SuccessResponse(message="Runtime cancelled.", data=result)


@router.post("/validate", response_model=SuccessResponse)
async def validate_runtime(project_id: str = Query(...)):
    """Validate the runtime environment without preparing.

    Runs environment and GPU checks as a pre-flight validation.
    """
    coordinator = _get_or_create_runtime(project_id)
    # Run environment + GPU discovery without full preparation
    from app.runtime.environment import environment_validator
    from app.runtime.gpu import gpu_discovery

    env_report = environment_validator.validate()
    gpu_report = gpu_discovery.discover()

    return SuccessResponse(
        message="Runtime validation complete.",
        data={
            "environment": env_report,
            "gpu": gpu_report,
            "runtime_id": coordinator.runtime_id,
        },
    )


# ------------------------------------------------------------------
# Query endpoints
# ------------------------------------------------------------------

@router.get("/status", response_model=SuccessResponse)
async def runtime_status(project_id: str = Query(...)):
    """Get the current Runtime status."""
    coordinator = _get_or_create_runtime(project_id)
    return SuccessResponse(message="Runtime status.", data=coordinator.status())


@router.get("/environment", response_model=SuccessResponse)
async def runtime_environment(project_id: str = Query(...)):
    """Get the environment validation report."""
    coordinator = _get_or_create_runtime(project_id)
    env = coordinator.get_environment()
    if not env:
        raise HTTPException(status_code=404, detail="Environment report not yet generated.")
    return SuccessResponse(message="Environment report.", data=env)


@router.get("/events", response_model=SuccessResponse)
async def runtime_events(project_id: str = Query(...), limit: int = Query(50, ge=1, le=500)):
    """Get recent Runtime events."""
    coordinator = _get_or_create_runtime(project_id)
    events = coordinator.get_events(limit)
    return SuccessResponse(message=f"{len(events)} event(s).", data={"events": events})


@router.get("/manifest", response_model=SuccessResponse)
async def runtime_manifest(project_id: str = Query(...)):
    """Get the Runtime manifest."""
    coordinator = _get_or_create_runtime(project_id)
    return SuccessResponse(message="Runtime manifest.", data=coordinator.get_manifest())


@router.get("/metrics", response_model=SuccessResponse)
async def runtime_metrics(project_id: str = Query(...)):
    """Get current Runtime metrics."""
    coordinator = _get_or_create_runtime(project_id)
    return SuccessResponse(message="Runtime metrics.", data=coordinator.get_metrics())


@router.get("/audit", response_model=SuccessResponse)
async def runtime_audit(project_id: str = Query(...)):
    """Get the Runtime audit report."""
    coordinator = _get_or_create_runtime(project_id)
    audit = coordinator.get_audit()
    if not audit:
        raise HTTPException(status_code=404, detail="Audit report not yet generated.")
    return SuccessResponse(message="Runtime audit.", data=audit)
