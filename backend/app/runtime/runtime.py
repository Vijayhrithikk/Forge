"""
Runtime Coordinator — orchestrates every execution stage.

The Coordinator owns the complete execution lifecycle. Individual
modules never call one another directly. Everything flows through
the Coordinator. This creates a deterministic execution lifecycle.
"""

import json
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from app.core import settings, get_logger
from app.runtime.state_machine import StateMachine, RuntimeState, InvalidTransitionError
from app.runtime.manifest import RuntimeManifest, generate_runtime_id
from app.runtime.locks import ExecutionLock
from app.runtime.workspace import RuntimeWorkspace
from app.runtime.events import (
    RuntimeEvent, RuntimeEventType, EventSeverity, EventStore,
)
from app.runtime.logger import RuntimeLogger
from app.runtime.environment import environment_validator
from app.runtime.gpu import gpu_discovery
from app.runtime.metrics import RuntimeMetrics
from app.runtime.exceptions import RuntimeError as ForgeRuntimeError

logger = get_logger("app.runtime.coordinator")


class RuntimeCoordinator:
    """Orchestrates the full Runtime lifecycle.

    The Runtime follows this exact sequence:
    1. Create Runtime → 2. Acquire Lock → 3. Prepare Workspace →
    4. Validate Workspace → 5. Validate Environment → 6. Discover GPU →
    7. Validate GPU → 8. Generate Reports → 9. READY

    Every stage emits events, logs, and updates the manifest.
    The Coordinator never implements business logic — it orchestrates
    existing modules.
    """

    def __init__(self, project_path: Path, project_id: str):
        self._runtime_id = generate_runtime_id()
        self._project_id = project_id
        self._project_path = project_path

        # Core components
        self._state = StateMachine()
        self._workspace = RuntimeWorkspace(project_path)
        self._lock = ExecutionLock(self._workspace.lock_path)
        self._manifest = RuntimeManifest(
            runtime_id=self._runtime_id,
            project_id=project_id,
            workspace_path=self._workspace.runtime_path,
        )

        # Observability
        self._events = EventStore(self._workspace.events_path, self._runtime_id)
        self._log = RuntimeLogger(self._workspace.logs_path, self._runtime_id, project_id)
        self._metrics = RuntimeMetrics(self._runtime_id)

        # Results cache
        self._env_report: Optional[Dict] = None
        self._gpu_report: Optional[Dict] = None
        self._gpu_validation: Optional[list] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def runtime_id(self) -> str:
        return self._runtime_id

    @property
    def current_state(self) -> RuntimeState:
        return self._state.current_state

    @property
    def is_ready(self) -> bool:
        return self._state.current_state == RuntimeState.READY

    # ------------------------------------------------------------------
    # Lifecycle API
    # ------------------------------------------------------------------

    def create(self) -> Dict[str, Any]:
        """Step 1: Create the Runtime instance.

        Initializes state machine, emits RUNTIME_CREATED event,
        and saves the initial manifest.
        """
        self._emit_event(RuntimeEventType.RUNTIME_CREATED, "Runtime created.")
        self._log.info("CREATED", "coordinator", "Runtime instance created",
                       runtime_id=self._runtime_id, project_id=self._project_id)
        self._manifest.update_state(RuntimeState.CREATED)
        self._manifest.save(self._workspace.manifest_path)

        return self._status_response("Runtime created.")

    def prepare(self) -> Dict[str, Any]:
        """Steps 2-9: Execute full preparation pipeline."""
        try:
            # Step 2: Acquire lock
            self._begin_stage("lock_acquisition")
            self._state.transition(RuntimeState.VALIDATING)
            self._lock.acquire(self._runtime_id)
            self._emit_event(RuntimeEventType.LOCK_ACQUIRED, "Execution lock acquired.")
            self._complete_stage("lock_acquisition")

            # Step 3: Prepare workspace
            self._begin_stage("workspace_preparation")
            self._state.transition(RuntimeState.PREPARING)
            self._manifest.update_state(RuntimeState.PREPARING)
            self._emit_event(RuntimeEventType.WORKSPACE_PREPARATION_STARTED, "Preparing workspace.")
            self._workspace.prepare()
            self._emit_event(RuntimeEventType.WORKSPACE_PREPARATION_COMPLETED, "Workspace ready.")
            self._complete_stage("workspace_preparation")

            # Step 4: Validate workspace artifacts
            self._begin_stage("workspace_validation")
            missing = self._workspace.validate_required_artifacts(self._project_id)
            if missing:
                self._log.warning("PREPARING", "workspace",
                                  f"Missing artifacts: {missing}")
                self._manifest.add_warning()
            self._complete_stage("workspace_validation")

            # Step 5: Validate environment
            self._begin_stage("environment_validation")
            self._emit_event(RuntimeEventType.ENVIRONMENT_VALIDATION_STARTED, "Validating environment.")
            self._env_report = environment_validator.validate(self._log)
            self._manifest.update_environment(
                python_ver=sys_version(),
                torch_ver=_get_torch_version(),
                cuda_ver=_get_cuda_version(),
            )
            env_event_msg = f"Environment: {self._env_report['status']}"
            self._emit_event(RuntimeEventType.ENVIRONMENT_VALIDATION_COMPLETED,
                             env_event_msg)
            env_status = self._env_report["status"]
            if env_status == "FAIL":
                self._manifest.add_error()
            elif env_status == "WARNING":
                self._manifest.add_warning()
            self._complete_stage("environment_validation")

            # Save environment report
            env_path = self._workspace.runtime_path / "environment.json"
            with open(env_path, "w", encoding="utf-8") as f:
                json.dump(self._env_report, f, indent=2, ensure_ascii=False)

            # Steps 6-7: GPU discovery + validation
            self._begin_stage("gpu_discovery")
            self._emit_event(RuntimeEventType.GPU_DISCOVERY_STARTED, "Discovering GPU hardware.")
            self._gpu_report = gpu_discovery.discover(self._log)
            if self._gpu_report.get("gpus"):
                gpu = self._gpu_report["gpus"][0]
                self._manifest.update_gpu(
                    gpu_name=gpu["name"],
                    vram_gb=gpu["total_memory_gb"],
                    cuda_capability=gpu["compute_capability"],
                )
            self._emit_event(RuntimeEventType.GPU_DISCOVERY_COMPLETED,
                             f"Found {self._gpu_report['device_count']} GPU(s)")
            self._complete_stage("gpu_discovery")

            self._begin_stage("gpu_validation")
            self._emit_event(RuntimeEventType.GPU_VALIDATION_STARTED, "Validating GPU compatibility.")
            self._gpu_validation = gpu_discovery.validate_compatibility(
                estimated_vram_gb=6.0,  # Conservative default; real value from training plan
                runtime_log=self._log,
            )
            compat_count = sum(1 for g in self._gpu_validation if g["status"] == "compatible")
            if compat_count == 0:
                self._manifest.add_warning()
                self._log.warning("PREPARING", "gpu", "No fully compatible GPU found")
            self._emit_event(RuntimeEventType.GPU_VALIDATION_COMPLETED,
                             f"{compat_count} compatible GPU(s)")
            self._complete_stage("gpu_validation")

            # Step 8: Finalize
            self._begin_stage("manifest_creation")
            self._manifest.update_state(RuntimeState.READY)
            self._manifest.update_elapsed(self._metrics.elapsed_seconds)
            self._manifest.save(self._workspace.manifest_path)
            self._complete_stage("manifest_creation")

            # Step 9: READY
            self._state.transition(RuntimeState.READY)
            self._emit_event(RuntimeEventType.RUNTIME_READY, "Runtime ready for training.")
            self._log.success("READY", "coordinator", "Runtime preparation complete",
                              elapsed=self._metrics.elapsed_seconds)

            # Save metrics
            self._metrics.save(self._workspace.metrics_path)

            # Save audit
            self._save_audit()

            return self._status_response("Runtime ready.")

        except ForgeRuntimeError as exc:
            self._handle_failure(exc)
            return self._status_response(f"Preparation failed: {exc.description}", is_error=True)
        except Exception as exc:
            self._handle_failure(exc)
            return self._status_response(f"Unexpected error: {str(exc)}", is_error=True)

    def cancel(self) -> Dict[str, Any]:
        """Cancel the Runtime safely.

        Releases the execution lock, cleans temporary files,
        and transitions to CANCELLED state.
        """
        try:
            self._state.transition(RuntimeState.CANCELLED)
        except InvalidTransitionError:
            # Already terminal
            pass

        self._manifest.update_state(RuntimeState.CANCELLED)
        self._manifest.save(self._workspace.manifest_path)
        self._lock.release()
        self._workspace.cleanup_temporary()
        self._emit_event(RuntimeEventType.RUNTIME_CANCELLED, "Runtime cancelled.")
        self._log.info("CANCELLED", "coordinator", "Runtime cancelled")

        return {"status": "success", "message": "Runtime cancelled.", "runtime_id": self._runtime_id}

    def status(self) -> Dict[str, Any]:
        """Return current Runtime status (for API)."""
        return self._status_response("Runtime status retrieved.")

    def get_metrics(self) -> Dict[str, Any]:
        return self._metrics.to_dict()

    def get_environment(self) -> Dict:
        return self._env_report or {}

    def get_gpu_report(self) -> Dict:
        return self._gpu_report or {}

    def get_manifest(self) -> Dict:
        return self._manifest.data

    def get_events(self, limit: int = 50) -> list:
        return self._events.read_recent(limit)

    def get_audit(self) -> Dict:
        audit_path = self._workspace.runtime_path / "runtime_audit.json"
        if audit_path.exists():
            with open(audit_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _begin_stage(self, stage: str) -> None:
        self._metrics.begin_stage(stage)

    def _complete_stage(self, stage: str) -> None:
        self._metrics.complete_stage(stage)
        self._manifest.update_elapsed(self._metrics.elapsed_seconds)
        self._manifest.save(self._workspace.manifest_path)
        self._metrics.save(self._workspace.metrics_path)

    def _emit_event(self, event_type: RuntimeEventType, message: str,
                    severity: EventSeverity = EventSeverity.INFO,
                    metadata: Optional[Dict] = None) -> None:
        event = RuntimeEvent(
            event_type=event_type,
            runtime_id=self._runtime_id,
            severity=severity,
            message=message,
            metadata=metadata,
        )
        self._events.append(event)

    def _handle_failure(self, exc: Exception) -> None:
        """Handle a failure gracefully — log, emit, update manifest."""
        try:
            self._state.transition(RuntimeState.FAILED)
        except InvalidTransitionError:
            pass

        if isinstance(exc, ForgeRuntimeError):
            self._manifest.add_error()
            self._emit_event(RuntimeEventType.RUNTIME_FAILED, exc.description,
                             EventSeverity.ERROR, {"error_code": exc.error_code})
            self._log.error("FAILED", "coordinator", exc.description,
                            error_code=exc.error_code)
        else:
            self._manifest.add_error()
            self._emit_event(RuntimeEventType.RUNTIME_FAILED, str(exc),
                             EventSeverity.ERROR)
            self._log.error("FAILED", "coordinator", str(exc))

        self._manifest.update_state(RuntimeState.FAILED)
        self._manifest.update_elapsed(self._metrics.elapsed_seconds)
        self._manifest.save(self._workspace.manifest_path)
        self._metrics.save(self._workspace.metrics_path)

    def _status_response(self, message: str, is_error: bool = False) -> Dict[str, Any]:
        return {
            "status": "error" if is_error else "success",
            "runtime_id": self._runtime_id,
            "current_state": self._state.current_state.value,
            "progress": self._metrics.progress,
            "warnings": self._metrics._warnings,
            "errors": self._metrics._errors,
            "message": message,
            "environment_report": self._env_report,
            "gpu_report": self._gpu_report,
            "gpu_validation": self._gpu_validation,
            "manifest_location": str(self._workspace.manifest_path / "runtime_manifest.json"),
            "metrics_location": str(self._workspace.metrics_path / "runtime_metrics.json"),
        }

    def _save_audit(self) -> None:
        """Generate the execution audit report."""
        audit = {
            "runtime_id": self._runtime_id,
            "project_id": self._project_id,
            "forge_version": settings.app_version,
            "state": self._state.current_state.value,
            "validation_summary": {
                "environment": self._env_report["status"] if self._env_report else "unknown",
                "gpu": self._gpu_report["status"] if self._gpu_report else "unknown",
            },
            "warnings": self._metrics._warnings,
            "errors": self._metrics._errors,
            "elapsed_seconds": self._metrics.elapsed_seconds,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        audit_path = self._workspace.runtime_path / "runtime_audit.json"
        with open(audit_path, "w", encoding="utf-8") as f:
            json.dump(audit, f, indent=2, ensure_ascii=False)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def sys_version() -> str:
    import sys
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def _get_torch_version() -> str:
    try:
        import torch
        return torch.__version__
    except ImportError:
        return "not installed"


def _get_cuda_version() -> str:
    try:
        import torch
        return torch.version.cuda or "N/A"
    except Exception:
        return "N/A"
