"""Dependency validator — checks all required Python packages."""
from typing import Dict, Any
from importlib.metadata import version, PackageNotFoundError

REQUIRED = {
    "torch": {"min": "2.0.0", "required": True},
    "transformers": {"min": "4.40.0", "required": True},
    "datasets": {"min": "2.14.0", "required": True},
    "accelerate": {"min": "0.30.0", "required": False},
    "trl": {"min": "0.10.0", "required": False},
    "peft": {"min": "0.10.0", "required": False},
    "huggingface_hub": {"min": "0.20.0", "required": True},
    "bitsandbytes": {"min": "0.41.0", "required": False},
    "safetensors": {"min": "0.4.0", "required": True},
    "sentencepiece": {"min": "0.1.99", "required": False},
    "tokenizers": {"min": "0.19.0", "required": True},
}

def validate_dependencies() -> Dict[str, Any]:
    results = {}
    for name, req in REQUIRED.items():
        try:
            v = version(name)
            compatible = _version_ge(v, req["min"])
            status = "PASS" if compatible else "FAIL"
            results[name] = {"status": status, "installed": v, "required": req["min"],
                            "required_pkg": req["required"], "message": f"{name}=={v}"}
        except PackageNotFoundError:
            status = "FAIL" if req["required"] else "WARNING"
            results[name] = {"status": status, "installed": None, "required": req["min"],
                            "required_pkg": req["required"], "message": f"{name} not installed"}
    return results

def _version_ge(installed: str, required: str) -> bool:
    try:
        from packaging.version import Version
        return Version(installed) >= Version(required)
    except ImportError:
        parts_i = [int(x) for x in installed.split(".")[:3]]
        parts_r = [int(x) for x in required.split(".")[:3]]
        return parts_i >= parts_r
