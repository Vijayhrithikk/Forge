"""
Project management endpoints.

Create and manage training projects.
Each project has an isolated filesystem workspace.
"""

from fastapi import APIRouter, HTTPException

from app.core import get_logger
from app.schemas.responses import SuccessResponse, ErrorDetail, ErrorResponse
from app.engines.dataset.workspace import workspace_engine

router = APIRouter(prefix="/projects", tags=["projects"])
logger = get_logger("app.api.projects")


@router.post("", response_model=SuccessResponse, status_code=201)
async def create_project(name: str):
    """Create a new training project with a filesystem workspace.

    Args:
        name: Human-readable project name (must be unique).

    Returns:
        SuccessResponse with project details including ID and directory paths.
    """
    if not name or not name.strip():
        logger.warning("project_create_failed", reason="empty_name")
        raise HTTPException(status_code=400, detail="Project name is required.")

    try:
        project = workspace_engine.create_project(name.strip())
        logger.info("project_created_via_api", project_id=project["project_id"])
        return SuccessResponse(
            message=f"Project '{name}' created successfully.",
            data=project,
        )
    except FileExistsError:
        logger.warning("project_create_failed", reason="name_exists", name=name)
        raise HTTPException(
            status_code=409,
            detail=f"A project with name '{name}' already exists.",
        )
