"""
Runtime Workspace — prepares and validates the execution directory structure.

Every Runtime instance gets its own workspace with standard subdirectories
for manifest, logs, events, metrics, checkpoints, temporary files, and exports.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List

from app.core import get_logger
from app.runtime.exceptions import MissingWorkspace, DiskPressure

logger = get_logger("app.runtime.workspace")

# Minimum disk space required (1 GB)
MIN_DISK_SPACE_GB = 1.0

# Standard runtime subdirectories
RUNTIME_SUBDIRS = [
    "manifest",
    "logs",
    "events",
    "metrics",
    "checkpoints",
    "temporary",
    "exports",
]


class RuntimeWorkspace:
    """Manages the execution workspace for a Runtime instance.

    Responsibilities:
    - Create standard directory layout
    - Validate required artifacts exist
    - Check disk space
    - Clean up temporary files on cancellation
    """

    def __init__(self, project_path: Path):
        self._project_path = project_path
        self._runtime_path = project_path / "runtime"

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def runtime_path(self) -> Path:
        return self._runtime_path

    @property
    def manifest_path(self) -> Path:
        return self._runtime_path / "manifest"

    @property
    def logs_path(self) -> Path:
        return self._runtime_path / "logs"

    @property
    def events_path(self) -> Path:
        return self._runtime_path / "events"

    @property
    def metrics_path(self) -> Path:
        return self._runtime_path / "metrics"

    @property
    def checkpoints_path(self) -> Path:
        return self._runtime_path / "checkpoints"

    @property
    def temporary_path(self) -> Path:
        return self._runtime_path / "temporary"

    @property
    def exports_path(self) -> Path:
        return self._runtime_path / "exports"

    @property
    def lock_path(self) -> Path:
        return self._runtime_path

    def all_paths(self) -> Dict[str, Path]:
        return {
            "runtime": self._runtime_path,
            "manifest": self.manifest_path,
            "logs": self.logs_path,
            "events": self.events_path,
            "metrics": self.metrics_path,
            "checkpoints": self.checkpoints_path,
            "temporary": self.temporary_path,
            "exports": self.exports_path,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare(self) -> Dict[str, Path]:
        """Create the standard runtime directory layout.

        Returns:
            Dict mapping directory names to their paths.

        Raises:
            MissingWorkspace: If the project path doesn't exist.
            DiskPressure: If insufficient disk space is available.
        """
        if not self._project_path.is_dir():
            raise MissingWorkspace(path=str(self._project_path))

        self._check_disk_space()

        paths = {}
        for subdir in RUNTIME_SUBDIRS:
            dir_path = self._runtime_path / subdir
            dir_path.mkdir(parents=True, exist_ok=True)
            paths[subdir] = dir_path

        logger.info(
            "workspace_prepared",
            runtime_path=str(self._runtime_path),
            subdirs=RUNTIME_SUBDIRS,
        )

        return paths

    def validate_required_artifacts(self, project_id: str) -> List[str]:
        """Check that required project artifacts exist before execution.

        Args:
            project_id: The project to validate.

        Returns:
            List of missing artifact descriptions (empty = all present).
        """
        missing = []

        # Training plan
        plan_path = self._project_path / "reports" / "training_plan.json"
        if not plan_path.exists():
            missing.append("training_plan.json (generate from Configuration Studio)")

        # Dataset
        dataset_path = self._project_path / "dataset" / "original.jsonl"
        if not dataset_path.exists():
            missing.append("dataset/original.jsonl (upload a JSONL dataset)")

        # Project metadata
        meta_path = self._project_path / "metadata" / "metadata.json"
        if not meta_path.exists():
            missing.append("metadata.json (run dataset validation first)")

        return missing

    def cleanup_temporary(self) -> None:
        """Remove temporary files. Safe to call at any point."""
        if self.temporary_path.exists():
            shutil.rmtree(self.temporary_path, ignore_errors=True)
            self.temporary_path.mkdir(parents=True, exist_ok=True)
            logger.info("temporary_cleaned", path=str(self.temporary_path))

    def cleanup_all(self) -> None:
        """Remove the entire runtime workspace. Irreversible."""
        if self._runtime_path.exists():
            shutil.rmtree(self._runtime_path, ignore_errors=True)
            logger.info("workspace_removed", path=str(self._runtime_path))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_disk_space(self) -> None:
        """Verify sufficient disk space is available."""
        try:
            stat = shutil.disk_usage(self._project_path)
            free_gb = stat.free / (1024 ** 3)
            if free_gb < MIN_DISK_SPACE_GB:
                raise DiskPressure(
                    path=str(self._project_path),
                    available_gb=free_gb,
                    required_gb=MIN_DISK_SPACE_GB,
                )
        except OSError:
            # Can't check disk — proceed cautiously
            logger.warning("disk_check_failed", path=str(self._project_path))
