"""
Precision Engine — selects and validates training precision.

Reads recommendations from registry and training plan.
Validates GPU compatibility. Never silently downgrades.
"""

from dataclasses import dataclass
from typing import Optional, List

from app.core import get_logger
from app.preparation.exceptions import UnsupportedPrecision
from app.preparation.device import DeviceInfo

logger = get_logger("app.preparation.precision")


@dataclass
class PrecisionConfig:
    precision: str  # fp32, fp16, bf16
    dtype_name: str
    vram_factor: float
    description: str
    bf16_supported: bool = False
    fp16_supported: bool = False


class PrecisionEngine:
    """Selects and validates training precision."""

    PRECISION_MAP = {
        "fp32": PrecisionConfig("fp32", "float32", 2.0, "32-bit full precision"),
        "fp16": PrecisionConfig("fp16", "float16", 1.0, "16-bit half precision"),
        "bf16": PrecisionConfig("bf16", "bfloat16", 1.0, "16-bit brain floating point"),
    }

    def select_precision(
        self,
        recommended: str,
        device: DeviceInfo,
        preference: Optional[str] = None,
    ) -> PrecisionConfig:
        """Select the best precision for the device.

        Args:
            recommended: Recommended precision from the model registry.
            device: The selected execution device.
            preference: Optional user override.

        Returns:
            PrecisionConfig for the selected precision.

        Raises:
            UnsupportedPrecision: If the selected precision is unsupported.
        """
        selected = preference or recommended

        if selected not in self.PRECISION_MAP:
            raise UnsupportedPrecision(selected, list(self.PRECISION_MAP.keys()))

        config = self.PRECISION_MAP[selected]

        # Check device compatibility
        if selected == "bf16" and device.device_type == "cuda":
            config.bf16_supported = self._cuda_supports_bf16()
            if not config.bf16_supported:
                logger.warning("bf16_not_supported", fallback="fp16")
                return self.PRECISION_MAP["fp16"]

        if selected == "fp16" and device.device_type == "cuda":
            config.fp16_supported = self._cuda_supports_fp16()

        if device.device_type == "cpu" and selected != "fp32":
            logger.warning("cpu_precision_limited", selected=selected,
                           message="CPU only supports fp32 efficiently")

        logger.info("precision_selected", precision=config.precision,
                     bf16=config.bf16_supported, fp16=config.fp16_supported)
        return config

    def list_supported(self, device: DeviceInfo) -> List[str]:
        """List all precision modes supported by the device."""
        supported = ["fp32"]
        if device.device_type == "cuda":
            if self._cuda_supports_fp16():
                supported.append("fp16")
            if self._cuda_supports_bf16():
                supported.append("bf16")
        return supported

    @staticmethod
    def _cuda_supports_bf16() -> bool:
        try:
            import torch
            return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        except ImportError:
            return False

    @staticmethod
    def _cuda_supports_fp16() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False


# Singleton
precision_engine = PrecisionEngine()
