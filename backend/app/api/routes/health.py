"""
Health check endpoint.

Provides rich application status for monitoring, load balancers,
and the frontend connectivity indicator.
"""

import sys
import time
from datetime import datetime, timezone

from fastapi import APIRouter

from app.core import settings, get_logger
from app.schemas.responses import SuccessResponse, HealthResponse
from app.utils.git import get_git_commit

router = APIRouter(tags=["health"])
logger = get_logger("app.api.health")

# Capture startup time for uptime calculation
_START_TIME = time.time()

# Supported models for Version 1 (LoRA-capable, open-weight models)
_SUPPORTED_MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "microsoft/phi-2",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
]


@router.get("/health", response_model=SuccessResponse)
async def health_check():
    """Return comprehensive application health status.

    Returns:
        SuccessResponse with HealthData including version, git commit,
        Python runtime, supported models, environment, and uptime.
    """
    uptime = time.time() - _START_TIME
    timestamp = datetime.now(timezone.utc).isoformat()

    health_data = HealthResponse(
        status="healthy",
        application=settings.app_name,
        version=settings.app_version,
        build=timestamp[:10],  # build date as YYYY-MM-DD
        git_commit=get_git_commit(),
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        supported_models=_SUPPORTED_MODELS,
        environment=settings.app_env,
        uptime=round(uptime, 2),
        timestamp=timestamp,
    )

    logger.info("health_check", uptime=round(uptime, 2))

    return SuccessResponse(
        message="Application is healthy",
        data=health_data.model_dump(),
    )
