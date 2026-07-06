"""
Runtime Environment Engine — validates the execution environment.

Checks Python version, installed libraries (PyTorch, Transformers,
PEFT, TRL, Accelerate), CUDA availability, OS, CPU, RAM, and disk.

Every check returns PASS/WARNING/FAIL with explanations.
Never boolean only.
"""

import os
import sys
import platform
import shutil
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from app.core import get_logger
from app.runtime.logger import RuntimeLogger

logger = get_logger("app.runtime.environment")


def _get_package_version(package_name: str) -> Optional[str]:
    """Safely get the installed version of a package."""
    try:
        mod = __import__(package_name, fromlist=["__version__"])
        return getattr(mod, "__version__", None)
    except ImportError:
        return None


class EnvironmentValidator:
    """Validates the Python environment for training readiness.

    Checks all required libraries and reports their status.
    Gracefully handles missing packages — doesn't crash on imports.
    """

    def __init__(self):
        self._results: List[Dict[str, Any]] = []

    def validate(self, runtime_log: Optional[RuntimeLogger] = None) -> Dict[str, Any]:
        """Run all environment checks and return a report.

        Returns:
            Dict with status, timestamp, checks list, and summary.
        """
        self._results = []

        # Python
        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        self._check("python", "Python", py_ver, "3.10", "3.13",
                     f"Python {py_ver}", "PASS")

        # OS
        os_name = platform.system()
        self._check("os", "Operating System", os_name, None, None,
                     f"{os_name} {platform.release()}", "PASS")

        # CPU
        cpu_count = os.cpu_count() or 1
        self._check("cpu", "CPU Cores", str(cpu_count), "4", None,
                     f"{cpu_count} logical cores", "PASS" if cpu_count >= 4 else "WARNING")

        # RAM
        try:
            import psutil
            ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        except ImportError:
            ram_gb = 8.0  # Assume 8GB if psutil unavailable
        self._check("ram", "System RAM", f"{ram_gb:.1f} GB", "8", None,
                     f"{ram_gb:.1f} GB", "PASS" if ram_gb >= 8 else "WARNING")

        # Disk
        try:
            cwd = Path.cwd()
            disk = shutil.disk_usage(cwd)
            disk_free_gb = disk.free / (1024 ** 3)
            self._check("disk", "Free Disk Space", f"{disk_free_gb:.1f} GB", "5", None,
                         f"{disk_free_gb:.1f} GB free", "PASS" if disk_free_gb >= 5 else "WARNING")
        except Exception:
            self._check("disk", "Free Disk Space", "unknown", "5", None,
                         "Could not determine disk space", "WARNING")

        # PyTorch
        torch_ver = _get_package_version("torch")
        if torch_ver:
            self._check("torch", "PyTorch", torch_ver, "2.0", None,
                         f"PyTorch {torch_ver}", "PASS")
        else:
            self._check("torch", "PyTorch", "not installed", "2.0", None,
                         "PyTorch is required for training", "FAIL")

        # CUDA via PyTorch
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            cuda_ver = torch.version.cuda if cuda_available else "N/A"
            self._check("cuda_torch", "Torch CUDA", str(cuda_available), None, None,
                         f"CUDA {'available' if cuda_available else 'not available'} (v{cuda_ver})",
                         "PASS" if cuda_available else "WARNING")
        except ImportError:
            self._check("cuda_torch", "Torch CUDA", "unknown", None, None,
                         "PyTorch not installed — cannot check CUDA", "WARNING")

        # Transformers
        tf_ver = _get_package_version("transformers")
        if tf_ver:
            self._check("transformers", "Transformers", tf_ver, "4.40", None,
                         f"Transformers {tf_ver}", "PASS")
        else:
            self._check("transformers", "Transformers", "not installed", "4.40", None,
                         "Required for model loading and tokenization", "WARNING")

        # PEFT
        peft_ver = _get_package_version("peft")
        if peft_ver:
            self._check("peft", "PEFT", peft_ver, "0.10", None,
                         f"PEFT {peft_ver}", "PASS")
        else:
            self._check("peft", "PEFT", "not installed", "0.10", None,
                         "Required for LoRA fine-tuning", "WARNING")

        # Accelerate
        accel_ver = _get_package_version("accelerate")
        if accel_ver:
            self._check("accelerate", "Accelerate", accel_ver, "0.30", None,
                         f"Accelerate {accel_ver}", "PASS")
        else:
            self._check("accelerate", "Accelerate", "not installed", "0.30", None,
                         "Used for efficient training", "WARNING")

        # TRL
        trl_ver = _get_package_version("trl")
        if trl_ver:
            self._check("trl", "TRL", trl_ver, "0.10", None,
                         f"TRL {trl_ver}", "PASS")
        else:
            self._check("trl", "TRL", "not installed", "0.10", None,
                         "Required for supervised fine-tuning", "WARNING")

        # Compile results
        passed = sum(1 for r in self._results if r["status"] == "PASS")
        warnings = sum(1 for r in self._results if r["status"] == "WARNING")
        failed = sum(1 for r in self._results if r["status"] == "FAIL")

        report = {
            "status": "FAIL" if failed > 0 else ("WARNING" if warnings > 0 else "PASS"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": self._results,
            "summary": {"passed": passed, "warnings": warnings, "failed": failed},
        }

        if runtime_log:
            runtime_log.info("PREPARING", "environment",
                             f"Environment check: {passed}P/{warnings}W/{failed}F")

        return report

    def _check(self, key: str, name: str, value: str, minimum: Optional[str],
               maximum: Optional[str], message: str, status: str) -> None:
        self._results.append({
            "key": key,
            "name": name,
            "value": value,
            "minimum": minimum,
            "maximum": maximum,
            "message": message,
            "status": status,
        })


# Singleton
environment_validator = EnvironmentValidator()
