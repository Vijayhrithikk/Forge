"""Hardware scanner — CPU, RAM, GPU, OS, Python."""
import os, sys, platform
from typing import Dict, Any

def scan_hardware() -> Dict[str, Any]:
    info = {
        "os": platform.system(), "os_release": platform.release(),
        "architecture": platform.machine(),
        "cpu_count": os.cpu_count() or 1,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }
    try:
        import psutil
        mem = psutil.virtual_memory()
        info["ram_total_gb"] = round(mem.total / (1024**3), 1)
        info["ram_available_gb"] = round(mem.available / (1024**3), 1)
    except ImportError:
        info["ram_total_gb"] = 8.0
        info["ram_available_gb"] = 4.0
    info["gpus"] = _scan_gpus()
    return info

def _scan_gpus() -> list:
    gpus = []
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpus.append({
                    "index": i, "name": props.name,
                    "vram_gb": round((props.total_memory if hasattr(props, 'total_memory') else props.total_mem) / (1024**3), 1),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "driver": torch._C._cuda_getDriverVersion() if hasattr(torch._C, '_cuda_getDriverVersion') else "unknown",
                })
        return gpus
    except ImportError:
        return []
