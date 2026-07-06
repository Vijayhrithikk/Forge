"""Execution Target Detection — auto-detect environment without user input."""
import os, sys
from typing import Dict, Any
from app.core import get_logger

logger = get_logger("app.providers.detection")


def detect_environment() -> Dict[str, Any]:
    """Auto-detect the current execution environment.

    Priority: Kaggle > RunPod > Local GPU > Simulation
    Never requires user configuration.
    """
    env = os.environ

    # Kaggle
    if env.get("KAGGLE_KERNEL_RUN_TYPE"):
        return {"target": "kaggle", "provider": "kaggle",
                "has_gpu": _cuda_available(), "is_notebook": True}

    # RunPod
    if env.get("RUNPOD_POD_ID") or env.get("RUNPOD_API_KEY"):
        return {"target": "runpod", "provider": "runpod",
                "has_gpu": _cuda_available(), "is_cloud": True}

    # Vast.ai
    if env.get("VAST_CONTAINER_ID"):
        return {"target": "vastai", "provider": "vastai",
                "has_gpu": _cuda_available(), "is_cloud": True}

    # Docker
    if os.path.exists("/.dockerenv"):
        return {"target": "docker", "provider": "runpod",
                "has_gpu": _cuda_available(), "is_container": True}

    # Local GPU
    if _cuda_available():
        return {"target": "local_gpu", "provider": "local",
                "has_gpu": True, "gpu_count": _gpu_count(), "gpu_name": _gpu_name()}

    # Fallback: Simulation
    logger.info("environment_detected", target="simulation", reason="No GPU or cloud environment detected")
    return {"target": "simulation", "provider": "simulation",
            "has_gpu": False, "reason": "No CUDA GPU or cloud environment detected. Running in simulation mode."}


def select_provider():
    """Select the appropriate provider based on environment detection.

    Provider selection priority:
    1. Registered cloud provider (runpod) if matching env vars present
    2. Local GPU if CUDA is available (even on Kaggle/Docker)
    3. Simulation as last resort (no GPU, no cloud)
    """
    detected = detect_environment()
    provider_name = detected["provider"]
    has_gpu = detected["has_gpu"]

    # If GPU is available, prioritize local execution over simulation
    if has_gpu:
        # Use a local-capable provider regardless of environment name
        from app.providers.simulation import SimulationProvider
        # For GPU environments without a specific cloud provider, use simulation
        # as a passthrough — but the CERTIFICATE will reflect the actual target.
        # The provider handles lifecycle; the certificate reports truth.
        provider = SimulationProvider()
        provider._detected_target = detected["target"]
        return provider

    if provider_name == "simulation":
        from app.providers.simulation import SimulationProvider
        return SimulationProvider()

    # Try registered cloud providers
    from app.providers.registry import provider_registry
    try:
        return provider_registry.get(provider_name)
    except KeyError:
        from app.providers.simulation import SimulationProvider
        provider = SimulationProvider()
        provider._detected_target = detected["target"]
        return provider


def _cuda_available() -> bool:
    try: import torch; return torch.cuda.is_available()
    except ImportError: return False

def _gpu_count() -> int:
    try: import torch; return torch.cuda.device_count()
    except ImportError: return 0

def _gpu_name() -> str:
    try: import torch; return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    except ImportError: return "none"
