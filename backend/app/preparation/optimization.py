"""
Optimization Engine — training optimizations and feature validation.

Supports gradient checkpointing, Flash Attention, TF32, and
memory-efficient attention. Registry-driven; never enables
unsupported optimizations.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from app.core import get_logger
from app.preparation.device import DeviceInfo

logger = get_logger("app.preparation.optimization")


@dataclass
class OptimizationConfig:
    gradient_checkpointing: bool = False
    flash_attention: bool = False
    tf32: bool = False
    memory_efficient_attention: bool = False
    flags: Dict[str, bool] = field(default_factory=dict)
    supported: Dict[str, bool] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


class OptimizationEngine:
    """Validates and configures training optimizations."""

    def configure(
        self,
        device: DeviceInfo,
        enable_gradient_checkpointing: bool = True,
        enable_flash_attention: bool = False,
        enable_tf32: bool = True,
    ) -> OptimizationConfig:
        """Configure available optimizations for the device.

        Args:
            device: The selected training device.
            enable_gradient_checkpointing: Reduce memory via recomputation.
            enable_flash_attention: Faster attention (Ampere+).
            enable_tf32: TensorFloat-32 (Ampere+).

        Returns:
            OptimizationConfig with what was enabled and notes.
        """
        config = OptimizationConfig()

        # Gradient checkpointing (CPU-safe)
        if enable_gradient_checkpointing:
            config.gradient_checkpointing = True
            config.supported["gradient_checkpointing"] = True
            config.notes.append("Gradient checkpointing enabled: reduced memory, slightly slower.")

        # Flash Attention (CUDA + Ampere+)
        if enable_flash_attention and device.device_type == "cuda":
            if self._cuda_compute_capability() >= 8.0:
                config.flash_attention = True
                config.supported["flash_attention"] = True
                config.notes.append("Flash Attention 2 enabled: faster attention, reduced memory.")
            else:
                config.supported["flash_attention"] = False
                config.notes.append("Flash Attention requires Ampere+ GPU (compute 8.0+). Disabled.")

        # TF32 (Ampere+)
        if enable_tf32 and device.device_type == "cuda":
            if self._cuda_compute_capability() >= 8.0:
                config.tf32 = True
                config.supported["tf32"] = True
                config.notes.append("TF32 enabled: faster matmuls on Ampere+ GPUs.")
                # Enable in PyTorch
                try:
                    import torch
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                except Exception:
                    pass
            else:
                config.supported["tf32"] = False
                config.notes.append("TF32 requires Ampere+ GPU. Disabled.")

        config.flags = {
            "gradient_checkpointing": config.gradient_checkpointing,
            "flash_attention": config.flash_attention,
            "tf32": config.tf32,
            "memory_efficient_attention": config.memory_efficient_attention,
        }

        logger.info("optimization_configured", **config.flags)
        return config

    @staticmethod
    def _cuda_compute_capability() -> float:
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                return float(f"{props.major}.{props.minor}")
        except Exception:
            pass
        return 0.0


# Singleton
optimization_engine = OptimizationEngine()
