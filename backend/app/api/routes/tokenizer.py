"""
Tokenizer preview endpoint.

Allows users to explore how tokenization works
by previewing individual samples from their dataset.
"""

from fastapi import APIRouter, Query, HTTPException

from app.core import get_logger
from app.schemas.responses import SuccessResponse
from app.engines.dataset.tokenizer import tokenizer_preview
from app.engines.dataset.workspace import workspace_engine

import json

router = APIRouter(prefix="/tokenizer", tags=["tokenizer"])
logger = get_logger("app.api.tokenizer")


@router.get("/preview", response_model=SuccessResponse)
async def preview_tokenizer(
    project_id: str = Query(...),
    sample: int = Query(1, ge=1),
):
    """Preview tokenization for a specific sample in a dataset.

    Loads the specified sample from the project's dataset and
    shows how the Tokenizer Preview Engine would tokenize it.

    Args:
        project_id: Target project.
        sample: Sample number (1-indexed) to preview.

    Returns:
        SuccessResponse with original text, tokens, counts, and cost estimates.
    """
    try:
        dataset_dir = workspace_engine.get_dataset_path(project_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")

    dataset_path = dataset_dir / "original.jsonl"
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="No dataset uploaded for this project.")

    # Load the requested sample
    with open(dataset_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i + 1 == sample:
                record = json.loads(line.strip())
                instruction = record.get("instruction", "")
                inp = record.get("input", "")
                output = record.get("output", "")

                prompt_text = f"{instruction} {inp}".strip()
                prompt_preview = tokenizer_preview.preview_sample(prompt_text)
                response_preview = tokenizer_preview.preview_sample(output)

                return SuccessResponse(
                    message=f"Tokenizer preview for sample {sample}.",
                    data={
                        "sample": sample,
                        "instruction": instruction,
                        "input": inp,
                        "output": output,
                        "prompt": prompt_preview,
                        "response": response_preview,
                    },
                )

    raise HTTPException(
        status_code=404,
        detail=f"Sample {sample} not found in dataset.",
    )
