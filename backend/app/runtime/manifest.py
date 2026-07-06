"""
Runtime Manifest — the source of truth for every execution.

The manifest is a JSON document that tracks every aspect of
a Runtime instance. It survives crashes and enables recovery.
"""

import json
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from app.core import settings, get_logger
from app.runtime.state_machine import RuntimeState

logger = get_logger("app.runtime.manifest")

MANIFEST_SCHEMA_VERSION = "1.0"


class RuntimeManifest:
    """Manages the runtime_manifest.json for a single execution.

    The manifest is written to disk on every update so it
    survives process crashes. Frontend and recovery logic
    read the manifest to understand current state.
    """

    def __init__(
        self,
        runtime_id: str,
        project_id: str,
        workspace_path: Path,
        plan_hash: str = "",
    ):
        self.runtime_id = runtime_id
        self.project_id = project_id
        self.workspace_path = workspace_path
        self._data: Dict[str, Any] = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "runtime_id": runtime_id,
            "project_id": project_id,
            "plan_hash": plan_hash,
            "plan_version": "1.0",
            "forge_version": settings.app_version,
            "registry_version": "1.0",
            "state": RuntimeState.CREATED.value,
            "workspace": str(workspace_path),
            "dataset": "",
            "model": "",
            "gpu": "",
            "cuda": "",
            "torch": "",
            "python": "",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint": {},
            "elapsed": 0.0,
            "status": {"warnings": 0, "errors": 0},
        }
        self._dirty = True

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> str:
        return self._data["state"]

    @state.setter
    def state(self, value: str):
        self._data["state"] = value
        self._touch()

    @property
    def data(self) -> Dict[str, Any]:
        return dict(self._data)

    # ------------------------------------------------------------------
    # Update methods
    # ------------------------------------------------------------------

    def update_state(self, state: RuntimeState) -> None:
        self._data["state"] = state.value
        self._touch()

    def update_environment(self, python_ver: str, torch_ver: str, cuda_ver: str) -> None:
        self._data["python"] = python_ver
        self._data["torch"] = torch_ver
        self._data["cuda"] = cuda_ver
        self._touch()

    def update_gpu(self, gpu_name: str, vram_gb: float, cuda_capability: str) -> None:
        self._data["gpu"] = gpu_name
        self._data["cuda"] = cuda_capability
        self._touch()

    def update_dataset(self, path: str, records: int) -> None:
        self._data["dataset"] = path
        self._touch()

    def update_model(self, model_id: str) -> None:
        self._data["model"] = model_id
        self._touch()

    def add_warning(self) -> None:
        self._data["status"]["warnings"] += 1
        self._touch()

    def add_error(self) -> None:
        self._data["status"]["errors"] += 1
        self._touch()

    def update_elapsed(self, seconds: float) -> None:
        self._data["elapsed"] = round(seconds, 1)
        self._touch()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: Path) -> Path:
        """Write the manifest to disk.

        Args:
            directory: The manifest directory (typically workspace/runtime/manifest/).

        Returns:
            Path to the saved manifest file.
        """
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "runtime_manifest.json"

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

        self._dirty = False
        return path

    @classmethod
    def load(cls, path: Path) -> "RuntimeManifest":
        """Load a manifest from disk.

        Args:
            path: Path to the runtime_manifest.json file.

        Returns:
            A RuntimeManifest instance populated from disk.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        manifest = cls(
            runtime_id=data["runtime_id"],
            project_id=data["project_id"],
            workspace_path=Path(data["workspace"]),
            plan_hash=data.get("plan_hash", ""),
        )
        manifest._data = data
        manifest._dirty = False

        logger.info("manifest_loaded", runtime_id=manifest.runtime_id, state=manifest.state)
        return manifest

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _touch(self) -> None:
        self._data["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._dirty = True


def generate_runtime_id() -> str:
    """Generate a unique runtime identifier."""
    return f"run_{uuid.uuid4().hex[:12]}"
