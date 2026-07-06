"""
GPU Engine — GPU discovery, validation, and compatibility checks.

Detects available GPUs, their VRAM, CUDA capability, and checks
compatibility against the Model Registry recommendations.
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from app.core import get_logger
from app.runtime.logger import RuntimeLogger

logger = get_logger("app.runtime.gpu")


class GPUDiscovery:
    """Discovers and validates GPU hardware for training."""

    def __init__(self):
        self._gpus: List[Dict[str, Any]] = []

    def discover(self, runtime_log: Optional[RuntimeLogger] = None) -> Dict[str, Any]:
        """Discover available GPUs and their properties.

        Returns:
            Dict with gpus list, count, and CUDA availability.
        """
        self._gpus = []

        try:
            import torch
        except ImportError:
            report = {
                "cuda_available": False,
                "device_count": 0,
                "gpus": [],
                "status": "FAIL",
                "message": "PyTorch not installed — cannot discover GPUs.",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            if runtime_log:
                runtime_log.error("PREPARING", "gpu", "PyTorch not installed")
            return report

        cuda_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count() if cuda_available else 0

        gpus = []
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            total_mem = props.total_mem / (1024 ** 3)
            try:
                free_mem = torch.cuda.mem_get_info(i)[0] / (1024 ** 3)
            except Exception:
                free_mem = total_mem

            gpus.append({
                "index": i,
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory_gb": round(total_mem, 1),
                "free_memory_gb": round(free_mem, 1),
                "multi_gpu_bridge": hasattr(props, "multi_gpu_board_group_id"),
            })

        report = {
            "cuda_available": cuda_available,
            "cuda_version": torch.version.cuda if cuda_available else "N/A",
            "device_count": device_count,
            "gpus": gpus,
            "status": "PASS" if cuda_available and device_count > 0 else "WARNING",
            "message": f"Found {device_count} GPU(s)." if device_count > 0 else "No GPUs detected. Training requires CUDA-capable GPU.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self._gpus = gpus

        if runtime_log:
            runtime_log.info("PREPARING", "gpu",
                             f"GPU discovery: {device_count} device(s), CUDA={cuda_available}")

        return report

    def validate_compatibility(
        self,
        estimated_vram_gb: float,
        registry_gpus: Optional[Dict[str, Any]] = None,
        runtime_log: Optional[RuntimeLogger] = None,
    ) -> List[Dict[str, Any]]:
        """Check discovered GPUs against training requirements.

        Args:
            estimated_vram_gb: VRAM required by the training config.
            registry_gpus: GPU compatibility data from the model registry.

        Returns:
            List of GPU compatibility results.
        """
        results = []

        for gpu in self._gpus:
            free = gpu["free_memory_gb"]
            if free >= estimated_vram_gb:
                status = "compatible"
                msg = f"{gpu['name']}: {free:.1f}GB free ≥ {estimated_vram_gb:.1f}GB required"
            elif free >= estimated_vram_gb * 0.8:
                status = "limited"
                msg = f"{gpu['name']}: {free:.1f}GB free (close to {estimated_vram_gb:.1f}GB required)"
            else:
                status = "unsupported"
                msg = f"{gpu['name']}: {free:.1f}GB free < {estimated_vram_gb:.1f}GB required"

            results.append({
                "index": gpu["index"],
                "name": gpu["name"],
                "total_memory_gb": gpu["total_memory_gb"],
                "free_memory_gb": free,
                "compute_capability": gpu["compute_capability"],
                "status": status,
                "message": msg,
            })

        if not self._gpus:
            results.append({
                "index": -1,
                "name": "No GPU detected",
                "total_memory_gb": 0,
                "free_memory_gb": 0,
                "compute_capability": "N/A",
                "status": "FAIL",
                "message": "No CUDA-capable GPU detected. Training requires a GPU.",
            })

        if runtime_log:
            compatible = sum(1 for r in results if r["status"] == "compatible")
            runtime_log.info("PREPARING", "gpu",
                             f"GPU validation: {compatible}/{len(results)} compatible")

        return results

    def supports_bf16(self) -> bool:
        """Check if any discovered GPU supports bfloat16."""
        for gpu in self._gpus:
            cap = gpu.get("compute_capability", "0.0")
            major = int(cap.split(".")[0]) if "." in cap else 0
            if major >= 8:  # Ampere+ supports BF16
                return True
        return False

    def supports_fp16(self) -> bool:
        """Check if any discovered GPU supports half precision."""
        for gpu in self._gpus:
            cap = gpu.get("compute_capability", "0.0")
            major = int(cap.split(".")[0]) if "." in cap else 0
            if major >= 7:  # Volta+ supports FP16 well
                return True
        return False


# Singleton
gpu_discovery = GPUDiscovery()
