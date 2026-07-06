"""
Runtime Exception Taxonomy — typed, recoverable-aware exception hierarchy.

Every exception exposes: error_code, title, description, recoverable, recommendation.
Never use bare Exception. Never expose raw stack traces.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class RuntimeError(Exception):
    """Base exception for all Runtime errors.

    Every RuntimeError must include enough information for the
    Coordinator to decide: retry, fail gracefully, or abort.
    """

    error_code: str
    title: str
    description: str
    recommendation: str
    recoverable: bool = False
    details: Optional[Dict[str, Any]] = None

    def __init__(self):
        super().__init__(self.description)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_code": self.error_code,
            "title": self.title,
            "description": self.description,
            "recommendation": self.recommendation,
            "recoverable": self.recoverable,
            "details": self.details or {},
        }


# ------------------------------------------------------------------
# Recoverable errors
# ------------------------------------------------------------------

class DownloadError(RuntimeError):
    def __init__(self, resource: str, details: Optional[Dict] = None):
        self.error_code = "RUNTIME_DOWNLOAD_ERROR"
        self.title = "Download Failed"
        self.description = f"Failed to download: {resource}"
        self.recommendation = "Check your network connection and retry. The Runtime will resume from where it stopped."
        self.recoverable = True
        self.details = details
        super().__init__()


class NetworkTimeout(RuntimeError):
    def __init__(self, operation: str):
        self.error_code = "RUNTIME_NETWORK_TIMEOUT"
        self.title = "Network Timeout"
        self.description = f"Network operation timed out: {operation}"
        self.recommendation = "Check your connection. The Runtime can retry this operation."
        self.recoverable = True
        self.details = {"operation": operation}
        super().__init__()


class GPUBusy(RuntimeError):
    def __init__(self, gpu_name: str, free_mb: int, required_mb: int):
        self.error_code = "RUNTIME_GPU_BUSY"
        self.title = "GPU Busy"
        self.description = f"GPU '{gpu_name}' has {free_mb}MB free but needs {required_mb}MB."
        self.recommendation = "Free GPU memory or select a different GPU. Retry when memory is available."
        self.recoverable = True
        self.details = {"gpu": gpu_name, "free_mb": free_mb, "required_mb": required_mb}
        super().__init__()


class DiskPressure(RuntimeError):
    def __init__(self, path: str, available_gb: float, required_gb: float):
        self.error_code = "RUNTIME_DISK_PRESSURE"
        self.title = "Insufficient Disk Space"
        self.description = f"Disk at '{path}' has {available_gb:.1f}GB free, needs {required_gb:.1f}GB."
        self.recommendation = "Free disk space or change the workspace location."
        self.recoverable = True
        self.details = {"path": path, "available_gb": available_gb, "required_gb": required_gb}
        super().__init__()


# ------------------------------------------------------------------
# Non-recoverable errors
# ------------------------------------------------------------------

class InvalidTrainingPlan(RuntimeError):
    def __init__(self, reason: str):
        self.error_code = "RUNTIME_INVALID_PLAN"
        self.title = "Invalid Training Plan"
        self.description = f"The Training Plan is invalid: {reason}"
        self.recommendation = "Regenerate the Training Plan from the Configuration Studio."
        self.recoverable = False
        self.details = {"reason": reason}
        super().__init__()


class InvalidDataset(RuntimeError):
    def __init__(self, reason: str):
        self.error_code = "RUNTIME_INVALID_DATASET"
        self.title = "Invalid Dataset"
        self.description = f"The dataset cannot be used for training: {reason}"
        self.recommendation = "Re-upload a valid JSONL dataset and run validation."
        self.recoverable = False
        self.details = {"reason": reason}
        super().__init__()


class MissingWorkspace(RuntimeError):
    def __init__(self, path: str):
        self.error_code = "RUNTIME_MISSING_WORKSPACE"
        self.title = "Missing Workspace"
        self.description = f"Required workspace not found: {path}"
        self.recommendation = "Create a project and upload a dataset first."
        self.recoverable = False
        self.details = {"path": path}
        super().__init__()


class UnsupportedGPU(RuntimeError):
    def __init__(self, gpu_name: str, reason: str):
        self.error_code = "RUNTIME_UNSUPPORTED_GPU"
        self.title = "Unsupported GPU"
        self.description = f"GPU '{gpu_name}' is not supported: {reason}"
        self.recommendation = "Use a compatible GPU (see registry for supported hardware)."
        self.recoverable = False
        self.details = {"gpu": gpu_name, "reason": reason}
        super().__init__()


class LockAcquisitionError(RuntimeError):
    def __init__(self, lock_path: str, existing_state: str):
        self.error_code = "RUNTIME_LOCK_ACQUISITION"
        self.title = "Execution Lock Active"
        self.description = f"Another runtime is active at '{lock_path}'. State: {existing_state}"
        self.recommendation = "Wait for the active execution to complete, or cancel it first."
        self.recoverable = True
        self.details = {"lock_path": lock_path, "existing_state": existing_state}
        super().__init__()
