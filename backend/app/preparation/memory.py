"""
Memory Engine — validates available memory for training.

Estimates current and projected memory usage. Warns before OOM.
Never waits until the Trainer crashes.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any

from app.core import get_logger
from app.preparation.device import DeviceInfo
from app.preparation.exceptions import MemoryInsufficient

logger = get_logger("app.preparation.memory")


@dataclass
class MemoryInfo:
    total_gb: float = 0.0
    free_gb: float = 0.0
    used_gb: float = 0.0
    projected_gb: float = 0.0
    safety_margin_gb: float = 0.0
    peak_gb: float = 0.0
    status: str = "UNKNOWN"  # PASS, WARNING, FAIL


class MemoryEngine:
    """Validates and tracks memory for training."""

    SAFETY_MARGIN_FACTOR = 0.15  # 15% safety margin

    def validate(
        self,
        device: DeviceInfo,
        model_size_gb: float,
        lora_overhead_gb: float = 0.5,
        optimizer_overhead_gb: float = 0.5,
        activation_overhead_gb: float = 1.0,
    ) -> MemoryInfo:
        """Validate that sufficient memory is available.

        Args:
            device: The selected device.
            model_size_gb: Estimated model memory footprint.
            lora_overhead_gb: LoRA adapter memory.
            optimizer_overhead_gb: Optimizer state memory.
            activation_overhead_gb: Activation memory.

        Returns:
            MemoryInfo with validation results.

        Raises:
            MemoryInsufficient: If memory is critically insufficient.
        """
        info = MemoryInfo()

        if device.device_type == "cpu":
            # CPU memory: use system RAM approximation
            try:
                import psutil
                mem = psutil.virtual_memory()
                info.total_gb = mem.total / (1024 ** 3)
                info.free_gb = mem.available / (1024 ** 3)
            except ImportError:
                info.total_gb = 16.0  # Assume 16GB
                info.free_gb = 8.0
        elif device.device_type == "cuda":
            try:
                import torch
                info.total_gb = torch.cuda.get_device_properties(device.index).total_mem / (1024 ** 3)
                free_bytes, _ = torch.cuda.mem_get_info(device.index)
                info.free_gb = free_bytes / (1024 ** 3)
            except ImportError:
                info.total_gb = 8.0
                info.free_gb = 4.0
        else:
            info.total_gb = 8.0
            info.free_gb = 4.0

        info.used_gb = info.total_gb - info.free_gb
        required = model_size_gb + lora_overhead_gb + optimizer_overhead_gb + activation_overhead_gb
        info.safety_margin_gb = required * self.SAFETY_MARGIN_FACTOR
        info.projected_gb = required + info.safety_margin_gb
        info.peak_gb = info.projected_gb

        if info.free_gb >= info.projected_gb:
            info.status = "PASS"
        elif info.free_gb >= required:
            info.status = "WARNING"
        else:
            info.status = "FAIL"

        logger.info("memory_validated",
                     total_gb=round(info.total_gb, 1),
                     free_gb=round(info.free_gb, 1),
                     projected_gb=round(info.projected_gb, 1),
                     status=info.status)

        if info.status == "FAIL":
            raise MemoryInsufficient(info.projected_gb, info.free_gb)

        return info


# Singleton
memory_engine = MemoryEngine()
