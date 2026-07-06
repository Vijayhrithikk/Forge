"""Simulation Provider — validates execution without GPU hardware."""
import json, time, os
from pathlib import Path
from typing import Dict, Any, Callable, Optional
from app.core import get_logger
from app.providers.base import BaseProvider, ProviderCapabilities

logger = get_logger("app.providers.simulation")


class SimulationProvider(BaseProvider):
    """Lightweight provider that simulates execution for validation.

    Never claims that training occurred. All certificates are clearly
    marked SIMULATION, not REMOTE_VALIDATED.
    """

    def name(self) -> str: return "simulation"
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            name="simulation", version="1.0",
            gpu_types=["simulated"], cuda_versions=["simulated"],
            max_disk_gb=10, persistent_storage=False,
            streaming_support=False, checkpoint_support=False, artifact_download=True,
        )

    def health(self) -> Dict[str, Any]:
        return {"provider": "simulation", "status": "READY",
                "checks": {"simulation": True, "note": "Simulation always ready for validation."}}

    def prepare(self, workspace: Dict = None) -> Dict[str, Any]:
        ws = (workspace or {}).get("root", "data/simulation")
        Path(ws).mkdir(parents=True, exist_ok=True)
        return {"provider": "simulation", "workspace": ws, "status": "prepared", "simulation": True}

    def upload(self, paths: Dict) -> Dict[str, Any]:
        return {"provider": "simulation", "status": "simulated_upload", "simulation": True}

    def execute(self, package: Dict, on_event: Optional[Callable] = None) -> Dict[str, Any]:
        if on_event: on_event("SIMULATION_STARTED", "Simulation execution started")
        return {"provider": "simulation", "status": "simulated_execution",
                "job_id": f"sim-{int(time.time())}", "simulation": True}

    def download(self, remote_paths: list) -> Dict[str, Any]:
        return {"provider": "simulation", "status": "simulated_download",
                "artifacts": remote_paths, "simulation": True}

    def cleanup(self) -> Dict[str, Any]:
        return {"provider": "simulation", "status": "simulated_cleanup", "simulation": True}
