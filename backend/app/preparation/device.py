"""
Device Engine — selects and validates the execution device.

Supports CPU, CUDA, and future MPS/Multi-GPU. Never hardcodes CUDA.
"""

from dataclasses import dataclass
from typing import Optional

from app.core import get_logger
from app.preparation.exceptions import DeviceNotAvailable

logger = get_logger("app.preparation.device")


@dataclass
class DeviceInfo:
    name: str
    device_type: str  # cpu, cuda, mps
    index: int = 0
    available: bool = True


class DeviceEngine:
    """Selects and validates the execution device."""

    SUPPORTED_DEVICES = ["cpu", "cuda", "mps"]

    def select_device(self, preference: str = "auto") -> DeviceInfo:
        """Select the best available device.

        Priority: CUDA > MPS > CPU

        Args:
            preference: 'auto', 'cpu', 'cuda', 'cuda:0', etc.

        Returns:
            DeviceInfo for the selected device.

        Raises:
            DeviceNotAvailable: If the requested device is unavailable.
        """
        if preference != "auto":
            return self._resolve_explicit(preference)

        # Auto-detect
        if self._cuda_available():
            return DeviceInfo(name="cuda", device_type="cuda", index=0)
        if self._mps_available():
            return DeviceInfo(name="mps", device_type="mps", index=0)

        logger.info("device_selected", name="cpu", method="auto")
        return DeviceInfo(name="cpu", device_type="cpu")

    def _resolve_explicit(self, device: str) -> DeviceInfo:
        """Resolve an explicit device string."""
        if device == "cpu":
            return DeviceInfo(name="cpu", device_type="cpu")

        if device.startswith("cuda"):
            if not self._cuda_available():
                raise DeviceNotAvailable(device, "CUDA not available")
            idx = int(device.split(":")[1]) if ":" in device else 0
            if idx >= self._cuda_device_count():
                raise DeviceNotAvailable(device, f"CUDA device {idx} not available")
            return DeviceInfo(name=device, device_type="cuda", index=idx)

        if device == "mps":
            if not self._mps_available():
                raise DeviceNotAvailable(device, "MPS not available on this platform")
            return DeviceInfo(name="mps", device_type="mps")

        raise DeviceNotAvailable(device, "Unknown device")

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    @staticmethod
    def _cuda_device_count() -> int:
        try:
            import torch
            return torch.cuda.device_count()
        except ImportError:
            return 0

    @staticmethod
    def _mps_available() -> bool:
        try:
            import torch
            return torch.backends.mps.is_available()
        except Exception:
            return False


# Singleton
device_engine = DeviceEngine()
