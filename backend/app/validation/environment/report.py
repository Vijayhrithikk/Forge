"""Environment capability report generator."""
import json
from pathlib import Path
from typing import Dict, Any

def generate_capability_report(scan: Dict) -> Dict[str, Any]:
    hw = scan.get("hardware", {})
    cu = scan.get("cuda", {})
    deps = scan.get("dependencies", {})
    has_gpu = cu.get("cuda_available", False) and len(hw.get("gpus", [])) > 0
    return {
        "capable": has_gpu,
        "gpu_name": hw.get("gpus", [{}])[0].get("name", "none") if has_gpu else "none",
        "vram_gb": hw.get("gpus", [{}])[0].get("vram_gb", 0) if has_gpu else 0,
        "ram_gb": hw.get("ram_total_gb", 0),
        "recommended_model": "qwen2.5-1.5b-instruct" if has_gpu else None,
        "can_train": has_gpu,
        "limitations": [],
        "recommendations": [],
    }
