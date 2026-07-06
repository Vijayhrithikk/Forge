"""
Workspace Engine — filesystem-based project management.

Every dataset belongs to a project. Projects are isolated directories
under the configured workspace root. No database, no ORM, no migrations.

Directory structure per project:
    workspace/{project_id}/
        dataset/       # original uploaded file
        uploads/       # temporary upload staging
        analysis/      # analysis results
        reports/       # generated reports (JSON)
        logs/          # per-stage logs
        metadata/      # project metadata
"""

import os
import uuid
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

from app.core import settings, get_logger

logger = get_logger("app.engines.dataset.workspace")


class WorkspaceEngine:
    """Manages the filesystem workspace for Forge projects.

    Responsibilities:
    - Create project directories with standard scaffolding.
    - Provide paths for dataset storage, reports, logs, and metadata.
    - Enforce project-level isolation.
    - Never overwrite existing data.
    """

    def __init__(self, root: Optional[str] = None):
        """Initialize the workspace engine.

        Args:
            root: Workspace root directory. Defaults to FORGE_DATA_DIR from settings.
        """
        self._root = Path(root or settings.forge_data_dir).resolve()
        self._ensure_root()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def root(self) -> Path:
        """Absolute path to the workspace root directory."""
        return self._root

    def create_project(self, name: str) -> Dict[str, Any]:
        """Create a new project with a unique ID and standard directory layout.

        Args:
            name: Human-readable project name.

        Returns:
            Dict with project_id, name, path, created_at, and subdirectory paths.
        """
        project_id = self._generate_project_id()

        # Validate uniqueness
        project_path = self._root / project_id
        if project_path.exists():
            logger.warning(
                "project_id_collision",
                project_id=project_id,
                message="Regenerating project ID due to collision",
            )
            project_id = self._generate_project_id()
            project_path = self._root / project_id

        # Create standard directory layout
        dirs = {
            "project": project_path,
            "dataset": project_path / "dataset",
            "uploads": project_path / "uploads",
            "analysis": project_path / "analysis",
            "reports": project_path / "reports",
            "logs": project_path / "logs",
            "metadata": project_path / "metadata",
        }

        for dir_path in dirs.values():
            dir_path.mkdir(parents=True, exist_ok=False)

        created_at = datetime.now(timezone.utc).isoformat()

        # Write initial metadata
        meta = {
            "project_id": project_id,
            "name": name,
            "created_at": created_at,
            "schema_version": "1.0",
        }
        self._write_json(dirs["metadata"] / "project.json", meta)

        logger.info(
            "project_created",
            project_id=project_id,
            name=name,
            path=str(project_path),
        )

        return {
            "project_id": project_id,
            "name": name,
            "created_at": created_at,
            "path": str(project_path),
            "directories": {k: str(v) for k, v in dirs.items()},
        }

    def get_project_path(self, project_id: str) -> Path:
        """Return the root path for a project.

        Args:
            project_id: The project identifier.

        Returns:
            Path to the project directory.

        Raises:
            FileNotFoundError: If the project does not exist.
        """
        path = self._root / project_id
        if not path.is_dir():
            raise FileNotFoundError(f"Project not found: {project_id}")
        return path

    def get_dataset_path(self, project_id: str) -> Path:
        """Return the dataset directory path for a project."""
        return self.get_project_path(project_id) / "dataset"

    def get_reports_path(self, project_id: str) -> Path:
        """Return the reports directory path for a project."""
        return self.get_project_path(project_id) / "reports"

    def get_logs_path(self, project_id: str) -> Path:
        """Return the logs directory path for a project."""
        return self.get_project_path(project_id) / "logs"

    def get_metadata_path(self, project_id: str) -> Path:
        """Return the metadata directory path for a project."""
        return self.get_project_path(project_id) / "metadata"

    def project_exists(self, project_id: str) -> bool:
        """Check whether a project directory exists."""
        return (self._root / project_id).is_dir()

    def delete_project(self, project_id: str) -> None:
        """Permanently delete a project and all its data.

        Args:
            project_id: The project to delete.

        Raises:
            FileNotFoundError: If the project does not exist.
        """
        path = self.get_project_path(project_id)
        shutil.rmtree(path)
        logger.info("project_deleted", project_id=project_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_root(self) -> None:
        """Create the workspace root directory if it doesn't exist."""
        self._root.mkdir(parents=True, exist_ok=True)
        logger.info("workspace_initialized", root=str(self._root))

    @staticmethod
    def _generate_project_id() -> str:
        """Generate a unique, URL-safe project identifier."""
        return uuid.uuid4().hex[:12]

    @staticmethod
    def _write_json(path: Path, data: Dict[str, Any]) -> None:
        """Write a dictionary as pretty-printed JSON."""
        import json

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


# Singleton instance
workspace_engine = WorkspaceEngine()
