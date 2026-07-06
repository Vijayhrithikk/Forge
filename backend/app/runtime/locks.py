"""
Execution Lock — prevents concurrent training in one workspace.

Creates, holds, and releases a filesystem lock. The lock survives
process crashes and enables recovery detection.
"""

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from app.core import get_logger
from app.runtime.exceptions import LockAcquisitionError

logger = get_logger("app.runtime.locks")

LOCK_FILENAME = "execution.lock"


class ExecutionLock:
    """Filesystem-based execution lock.

    Prevents concurrent Runtime instances from operating on the
    same project workspace. The lock contains metadata about the
    owning Runtime so recovery can make informed decisions.
    """

    def __init__(self, lock_directory: Path):
        self._path = lock_directory / LOCK_FILENAME
        self._acquired = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_acquired(self) -> bool:
        return self._acquired

    @property
    def path(self) -> Path:
        return self._path

    def acquire(self, runtime_id: str) -> None:
        """Acquire the execution lock.

        Creates the lock file with runtime metadata. If the lock
        already exists, checks whether it's stale and raises an
        appropriate error.

        Args:
            runtime_id: The ID of the Runtime acquiring the lock.

        Raises:
            LockAcquisitionError: If another active Runtime holds the lock.
        """
        if self._path.exists():
            existing = self._read_lock()
            if existing:
                existing_state = existing.get("state", "unknown")
                raise LockAcquisitionError(
                    lock_path=str(self._path),
                    existing_state=existing_state,
                )

        self._path.parent.mkdir(parents=True, exist_ok=True)
        lock_data = {
            "runtime_id": runtime_id,
            "acquired_at": datetime.now(timezone.utc).isoformat(),
            "state": "ACTIVE",
            "forge_version": "0.1.0",
        }
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(lock_data, f, indent=2)

        self._acquired = True
        logger.info("lock_acquired", runtime_id=runtime_id, path=str(self._path))

    def release(self) -> None:
        """Release the execution lock.

        Removes the lock file from disk. Safe to call even if
        the lock was never acquired.
        """
        if self._path.exists():
            self._path.unlink()
            logger.info("lock_released", path=str(self._path))
        self._acquired = False

    def read_state(self) -> Optional[dict]:
        """Read the current lock state without acquiring."""
        return self._read_lock()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _read_lock(self) -> Optional[dict]:
        """Read lock file contents, returning None if missing."""
        if not self._path.exists():
            return None
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            logger.warning("lock_file_corrupt", path=str(self._path))
            return None
