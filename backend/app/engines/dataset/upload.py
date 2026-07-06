"""
Upload Engine — receives, validates, and stores dataset files.

Supports JSONL only. Validates encoding, sanitizes filenames,
streams to disk, and triggers post-upload validation pipeline.
"""

import os
import re
import uuid
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, BinaryIO

import aiofiles
from fastapi import UploadFile

from app.core import settings, get_logger
from app.engines.dataset.workspace import workspace_engine

logger = get_logger("app.engines.dataset.upload")

# Maximum upload size (default 500MB)
MAX_UPLOAD_SIZE = 500 * 1024 * 1024

# Allowed extensions
ALLOWED_EXTENSIONS = {".jsonl"}

# Unsafe filename characters (beyond path separators and null)
_UNSAFE_FILENAME_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


class UploadError(Exception):
    """Raised when an upload fails validation before storage."""

    def __init__(self, message: str, recoverable: bool = False):
        self.message = message
        self.recoverable = recoverable
        super().__init__(message)


class UploadEngine:
    """Handles dataset file upload with validation and storage.

    Responsibilities:
    - Accept multipart file uploads via FastAPI.
    - Validate file type, size, encoding, and filename safety.
    - Stage upload in a temp location, then commit to project workspace.
    - Generate upload progress events.
    """

    def __init__(self, max_size: int = MAX_UPLOAD_SIZE):
        self._max_size = max_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def upload(
        self,
        file: UploadFile,
        project_id: str,
        on_progress=None,
    ) -> Dict[str, Any]:
        """Receive and store a dataset file for a project.

        The upload pipeline:
        1. Validate filename safety
        2. Validate file extension
        3. Read into temporary location (streaming, size-checked)
        4. Detect encoding (must be UTF-8)
        5. Validate JSONL structure
        6. Copy to project dataset directory
        7. Generate metadata

        Args:
            file: The uploaded file from FastAPI.
            project_id: Target project identifier.
            on_progress: Optional async callback(bytes_read, total).

        Returns:
            Dict with stored path, size, encoding, and record count.

        Raises:
            UploadError: For any validation failure.
            FileNotFoundError: If the project does not exist.
        """
        self._emit_progress(on_progress, "upload_started", 0, 0)

        # 1. Validate filename
        sanitized_name = self._sanitize_filename(file.filename or "dataset.jsonl")
        self._emit_progress(on_progress, "filename_validated", 0, 0)

        # 2. Validate extension
        ext = os.path.splitext(sanitized_name)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise UploadError(
                f"Unsupported file type '{ext}'. Only .jsonl files are accepted.",
                recoverable=False,
            )
        self._emit_progress(on_progress, "extension_validated", 0, 0)

        # 3. Ensure project exists
        project_path = workspace_engine.get_project_path(project_id)
        uploads_dir = project_path / "uploads"

        # 4. Stream to temporary file with size enforcement
        tmp_path = uploads_dir / f"tmp_{uuid.uuid4().hex}.upload"
        stored_path = await self._stream_to_disk(
            file, tmp_path, on_progress
        )

        # 5. Detect encoding
        encoding = self._detect_encoding(stored_path)
        if encoding.lower() not in ("utf-8", "ascii"):
            stored_path.unlink(missing_ok=True)
            raise UploadError(
                f"Dataset must be UTF-8 encoded. Detected: {encoding}.",
                recoverable=False,
            )
        self._emit_progress(on_progress, "encoding_validated", stored_path.stat().st_size, 0)

        # 6. Validate JSONL structure (quick scan)
        record_count = self._count_jsonl_records(stored_path)
        if record_count == 0:
            stored_path.unlink(missing_ok=True)
            raise UploadError(
                "Dataset contains no valid JSON records. Upload a non-empty JSONL file.",
                recoverable=False,
            )
        self._emit_progress(on_progress, "jsonl_validated", stored_path.stat().st_size, record_count)

        # 7. Commit to dataset directory
        dataset_dir = project_path / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        final_path = dataset_dir / "original.jsonl"

        # If a dataset already exists, archive it with a timestamp prefix
        if final_path.exists():
            archived = dataset_dir / f"original_{uuid.uuid4().hex[:8]}.jsonl"
            shutil.move(str(final_path), str(archived))
            logger.info("previous_dataset_archived", project_id=project_id, archived=str(archived))

        shutil.move(str(stored_path), str(final_path))
        self._emit_progress(on_progress, "upload_completed", final_path.stat().st_size, record_count)

        size = final_path.stat().st_size
        logger.info(
            "upload_completed",
            project_id=project_id,
            filename=sanitized_name,
            size=size,
            records=record_count,
            encoding=encoding,
        )

        return {
            "path": str(final_path),
            "filename": sanitized_name,
            "size": size,
            "encoding": encoding,
            "record_count": record_count,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _stream_to_disk(
        self,
        file: UploadFile,
        dest: Path,
        on_progress=None,
    ) -> Path:
        """Stream the uploaded file to disk with size enforcement."""
        total = 0
        chunk_size = 64 * 1024  # 64KB chunks

        async with aiofiles.open(dest, "wb") as out:
            while chunk := await file.read(chunk_size):
                total += len(chunk)
                if total > self._max_size:
                    dest.unlink(missing_ok=True)
                    max_mb = self._max_size / (1024 * 1024)
                    raise UploadError(
                        f"File exceeds maximum size of {max_mb:.0f} MB.",
                        recoverable=False,
                    )
                await out.write(chunk)
                self._emit_progress(on_progress, "uploading", total, 0)

        return dest

    def _sanitize_filename(self, filename: str) -> str:
        """Replace unsafe characters and strip path separators.

        Returns a safe basename — no directory traversal possible.
        """
        # Strip any path components
        name = os.path.basename(filename)
        # Replace unsafe characters with underscores
        name = _UNSAFE_FILENAME_RE.sub("_", name)
        # Remove leading dots/hyphens (hiding files on Unix)
        name = name.lstrip(".-")
        # Fallback for empty/invalid names
        if not name:
            name = "dataset.jsonl"
        # Ensure .jsonl extension
        if not name.lower().endswith(".jsonl"):
            name = f"{name}.jsonl" if "." not in name else name
        return name

    def _detect_encoding(self, path: Path) -> str:
        """Detect file encoding using chardet/cchardet or a basic heuristic.

        Falls back to 'utf-8' detection via a read attempt.
        """
        import json

        try:
            with open(path, "r", encoding="utf-8") as f:
                # Read the first few lines to confirm valid UTF-8
                for _ in range(min(10, 100)):
                    line = f.readline()
                    if not line:
                        break
                    json.loads(line.strip())
            return "utf-8"
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass

        # Basic detection: try utf-8, then latin-1, then cp1252
        for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
            try:
                with open(path, "r", encoding=enc) as f:
                    f.read(4096)
                return enc
            except (UnicodeDecodeError, UnicodeError):
                continue

        return "unknown"

    def _count_jsonl_records(self, path: Path, sample_lines: int = 500) -> int:
        """Quickly count valid JSON records in a JSONL file.

        Only samples the first N lines for speed — full validation
        is handled by the validation pipeline.
        """
        import json

        count = 0
        try:
            with open(path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        json.loads(line)
                        count += 1
                    except json.JSONDecodeError:
                        raise UploadError(
                            f"Invalid JSON on line {i + 1}. "
                            f"Ensure every line is a valid JSON object.",
                            recoverable=False,
                        )
        except UnicodeDecodeError:
            raise UploadError(
                "File is not valid UTF-8. Re-save the file with UTF-8 encoding.",
                recoverable=False,
            )

        return count

    @staticmethod
    def _emit_progress(on_progress, stage: str, bytes_done: int, records: int) -> None:
        """Emit a progress event if a callback is registered."""
        if on_progress:
            try:
                # Fire-and-forget — don't block the upload on progress
                import asyncio
                asyncio.ensure_future(
                    on_progress({
                        "stage": stage,
                        "bytes": bytes_done,
                        "records": records,
                    })
                )
            except Exception:
                pass  # Progress callbacks must never break the pipeline


# Singleton instance
upload_engine = UploadEngine()
