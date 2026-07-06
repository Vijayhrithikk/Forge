"""
Dataset upload endpoint.

Accepts JSONL file uploads, validates them, and stores them
in the project workspace.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request

from app.core import get_logger
from app.schemas.responses import SuccessResponse
from app.engines.dataset.upload import upload_engine, UploadError

router = APIRouter(prefix="/datasets", tags=["datasets"])
logger = get_logger("app.api.upload")


@router.post("/upload", response_model=SuccessResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    project_id: str = Form(...),
):
    """Upload a JSONL dataset file for a training project.

    The file is streamed to disk, validated for encoding and structure,
    and stored in the project's dataset directory.

    Args:
        file: The JSONL file to upload.
        project_id: Target project identifier.

    Returns:
        SuccessResponse with stored file metadata.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    try:
        result = await upload_engine.upload(file, project_id)
        logger.info(
            "dataset_uploaded_via_api",
            project_id=project_id,
            filename=result["filename"],
            records=result["record_count"],
            size=result["size"],
        )
        return SuccessResponse(
            message=f"Dataset uploaded successfully. {result['record_count']} records detected.",
            data=result,
        )
    except UploadError as exc:
        logger.warning(
            "upload_rejected",
            project_id=project_id,
            filename=file.filename,
            reason=exc.message,
        )
        raise HTTPException(
            status_code=400 if not exc.recoverable else 422,
            detail=exc.message,
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Project '{project_id}' not found.",
        )
