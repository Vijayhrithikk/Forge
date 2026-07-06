"""
Dataset validation, analysis, and inspection endpoints.

Triggers the full pipeline: validate → analyze → score → report.
Supports SSE streaming for live progress during validation.
"""

import json
import asyncio
from pathlib import Path
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.core import get_logger
from app.schemas.responses import SuccessResponse
from app.engines.dataset.workspace import workspace_engine
from app.engines.dataset.validation import validation_pipeline
from app.engines.dataset.analysis import analysis_engine

router = APIRouter(prefix="/datasets", tags=["datasets"])
logger = get_logger("app.api.datasets")


@router.post("/validate", response_model=SuccessResponse)
async def validate_dataset(project_id: str = Query(...)):
    """Run the full validation pipeline on a project's dataset.

    Returns validation results, statistics, and quality score.
    """
    try:
        dataset_dir = workspace_engine.get_dataset_path(project_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")

    dataset_path = dataset_dir / "original.jsonl"
    if not dataset_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No dataset uploaded for this project. Upload a JSONL file first.",
        )

    try:
        # 1. Validate
        validation_report = validation_pipeline.validate(dataset_path, project_id)

        # 2. Compute statistics
        stats = analysis_engine.compute_statistics(dataset_path, validation_report)

        # 3. Compute quality score
        quality = analysis_engine.compute_quality(stats, validation_report)

        # 4. Persist reports
        reports_dir = workspace_engine.get_reports_path(project_id)
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Write validation report
        with open(reports_dir / "validation.json", "w", encoding="utf-8") as f:
            json.dump({
                "project_id": project_id,
                "total_records": validation_report.total_records,
                "passed": validation_report.passed_count,
                "warnings": validation_report.warning_count,
                "failed": validation_report.failed_count,
                "is_valid": validation_report.is_valid,
                "results": [
                    {
                        "validator": r.validator,
                        "status": r.status.value,
                        "message": r.message,
                        "details": r.details,
                    }
                    for r in validation_report.results
                ],
                "schema_version": "1.0",
            }, f, indent=2, ensure_ascii=False)

        # Write statistics
        with open(reports_dir / "statistics.json", "w", encoding="utf-8") as f:
            json.dump({
                "project_id": project_id,
                "sample_count": stats.sample_count,
                "avg_prompt_chars": stats.avg_prompt_chars,
                "avg_response_chars": stats.avg_response_chars,
                "avg_prompt_tokens": stats.avg_prompt_tokens,
                "avg_response_tokens": stats.avg_response_tokens,
                "min_tokens": stats.min_tokens,
                "max_tokens": stats.max_tokens,
                "median_tokens": stats.median_tokens,
                "p95_tokens": stats.p95_tokens,
                "duplicate_prompt_count": stats.duplicate_prompt_count,
                "duplicate_exact_count": stats.duplicate_exact_count,
                "empty_prompt_count": stats.empty_prompt_count,
                "empty_response_count": stats.empty_response_count,
                "estimated_training_tokens": stats.estimated_training_tokens,
                "estimated_adapter_size_mb": stats.estimated_adapter_size_mb,
                "estimated_training_minutes": stats.estimated_training_minutes,
                "response_char_lengths": stats.response_char_lengths,
                "token_counts": stats.token_counts,
                "schema_version": "1.0",
            }, f, indent=2, ensure_ascii=False)

        # Write quality report
        with open(reports_dir / "quality.json", "w", encoding="utf-8") as f:
            json.dump({
                "project_id": project_id,
                "score": quality.score,
                "grade": quality.grade,
                "strengths": quality.strengths,
                "warnings": quality.warnings,
                "recommendations": quality.recommendations,
                "factor_scores": quality.factor_scores,
                "schema_version": "1.0",
            }, f, indent=2, ensure_ascii=False)

        # Write metadata
        dataset_file = dataset_path
        metadata = {
            "project_id": project_id,
            "dataset_name": "original.jsonl",
            "created_at": dataset_path.stat().st_ctime,
            "encoding": "utf-8",
            "records": stats.sample_count,
            "average_tokens": round((stats.avg_prompt_tokens + stats.avg_response_tokens), 1),
            "maximum_tokens": stats.max_tokens,
            "minimum_tokens": stats.min_tokens,
            "duplicates": stats.duplicate_exact_count,
            "empty_prompts": stats.empty_prompt_count,
            "empty_responses": stats.empty_response_count,
            "schema_version": "1.0",
            "quality_score": quality.score,
            "estimated_training_tokens": stats.estimated_training_tokens,
            "estimated_adapter_size_mb": stats.estimated_adapter_size_mb,
            "estimated_training_minutes": stats.estimated_training_minutes,
        }
        meta_dir = workspace_engine.get_metadata_path(project_id)
        with open(meta_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(
            "dataset_validated_and_analyzed",
            project_id=project_id,
            quality_score=quality.score,
            grade=quality.grade,
        )

        return SuccessResponse(
            message=f"Dataset analysis complete. Quality: {quality.score}/100 ({quality.grade}).",
            data={
                "validation": {
                    "total_records": validation_report.total_records,
                    "passed": validation_report.passed_count,
                    "warnings": validation_report.warning_count,
                    "failed": validation_report.failed_count,
                    "is_valid": validation_report.is_valid,
                    "results": [
                        {
                            "validator": r.validator,
                            "status": r.status.value,
                            "message": r.message,
                            "details": r.details,
                        }
                        for r in validation_report.results
                    ],
                },
                "statistics": {
                    "sample_count": stats.sample_count,
                    "avg_prompt_tokens": stats.avg_prompt_tokens,
                    "avg_response_tokens": stats.avg_response_tokens,
                    "min_tokens": stats.min_tokens,
                    "max_tokens": stats.max_tokens,
                    "median_tokens": stats.median_tokens,
                    "p95_tokens": stats.p95_tokens,
                    "duplicate_prompt_count": stats.duplicate_prompt_count,
                    "duplicate_exact_count": stats.duplicate_exact_count,
                    "empty_prompt_count": stats.empty_prompt_count,
                    "empty_response_count": stats.empty_response_count,
                    "estimated_training_tokens": stats.estimated_training_tokens,
                    "estimated_adapter_size_mb": stats.estimated_adapter_size_mb,
                    "estimated_training_minutes": stats.estimated_training_minutes,
                    "response_char_lengths": stats.response_char_lengths,
                    "token_counts": stats.token_counts,
                },
                "quality": {
                    "score": quality.score,
                    "grade": quality.grade,
                    "strengths": quality.strengths,
                    "warnings": quality.warnings,
                    "recommendations": quality.recommendations,
                    "factor_scores": quality.factor_scores,
                },
            },
        )
    except Exception as exc:
        logger.error("validation_failed", project_id=project_id, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/validate/stream")
async def validate_dataset_stream(project_id: str = Query(...)):
    """Stream dataset validation progress via Server-Sent Events.

    The client receives real-time updates as each validator runs.
    """
    try:
        dataset_dir = workspace_engine.get_dataset_path(project_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")

    dataset_path = dataset_dir / "original.jsonl"
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="No dataset uploaded for this project.")

    async def event_stream() -> AsyncGenerator[str, None]:
        queue: asyncio.Queue = asyncio.Queue()

        async def on_progress(event):
            await queue.put(event)

        # Run validation in a thread to not block the event loop
        loop = asyncio.get_event_loop()

        try:
            yield f"data: {json.dumps({'stage': 'validation_started', 'status': 'running'})}\n\n"

            validation_report = await loop.run_in_executor(
                None,
                lambda: validation_pipeline.validate(dataset_path, project_id, on_progress=None)
            )

            yield f"data: {json.dumps({'stage': 'validation_complete', 'passed': validation_report.passed_count, 'warnings': validation_report.warning_count, 'failed': validation_report.failed_count})}\n\n"

            yield f"data: {json.dumps({'stage': 'analysis_started'})}\n\n"

            stats = await loop.run_in_executor(
                None,
                lambda: analysis_engine.compute_statistics(dataset_path, validation_report)
            )

            yield f"data: {json.dumps({'stage': 'statistics_complete', 'sample_count': stats.sample_count})}\n\n"

            quality = analysis_engine.compute_quality(stats, validation_report)

            yield f"data: {json.dumps({'stage': 'quality_complete', 'score': quality.score, 'grade': quality.grade})}\n\n"

            yield f"data: {json.dumps({'stage': 'done', 'score': quality.score, 'grade': quality.grade, 'strengths': quality.strengths, 'warnings': quality.warnings, 'recommendations': quality.recommendations})}\n\n"

        except Exception as exc:
            yield f"data: {json.dumps({'stage': 'error', 'message': str(exc)})}\n\n"

        # Consume progress queue (runs in parallel but we stream after for simplicity)
        while not queue.empty():
            event = await queue.get()
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/inspect", response_model=SuccessResponse)
async def inspect_dataset(
    project_id: str = Query(...),
    sample: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
):
    """Inspect dataset samples with navigation.

    Args:
        project_id: Target project.
        sample: Starting sample number (1-indexed).
        limit: Number of samples to return.

    Returns:
        SuccessResponse with sample data and pagination info.
    """
    try:
        dataset_dir = workspace_engine.get_dataset_path(project_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")

    dataset_path = dataset_dir / "original.jsonl"
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="No dataset uploaded for this project.")

    total_records = 0
    samples = []

    with open(dataset_path, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
        total_records = sum(1 for line in all_lines if line.strip())

        start_idx = sample - 1
        end_idx = min(start_idx + limit, len(all_lines))

        for i in range(start_idx, end_idx):
            if i >= len(all_lines):
                break
            line = all_lines[i].strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            instruction = record.get("instruction", "")
            inp = record.get("input", "")
            output = record.get("output", "")

            from app.engines.dataset.analysis import TokenEstimator
            prompt_tokens = TokenEstimator.estimate(f"{instruction} {inp}".strip())
            response_tokens = TokenEstimator.estimate(output)

            samples.append({
                "line": i + 1,
                "instruction": instruction,
                "input": inp,
                "output": output,
                "char_count": len(instruction) + len(inp) + len(output),
                "prompt_tokens": prompt_tokens,
                "response_tokens": response_tokens,
                "total_tokens": prompt_tokens + response_tokens,
            })

    return SuccessResponse(
        message=f"Samples {sample}-{sample + len(samples) - 1} of {total_records}.",
        data={
            "total_records": total_records,
            "start": sample,
            "limit": limit,
            "samples": samples,
            "has_prev": sample > 1,
            "has_next": sample + limit <= total_records,
        },
    )
