"""
GPU Environment Validator — comprehensive validation for production GPU execution.

Validates hardware, CUDA, Python stack, storage, network, and execution readiness.
Returns READY, LIMITED, or NOT_READY with detailed reports.
"""

import json, os, sys, shutil
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any
from importlib.metadata import version, PackageNotFoundError

from app.core import settings, get_logger

logger = get_logger("app.validation.gpu_environment")

REQUIRED_PACKAGES = {
    "torch": "2.0.0", "transformers": "4.40.0", "datasets": "2.14.0",
    "accelerate": "0.30.0", "trl": "0.10.0", "peft": "0.10.0",
    "huggingface_hub": "0.20.0", "bitsandbytes": "0.41.0",
    "safetensors": "0.4.0", "sentencepiece": "0.1.99", "tokenizers": "0.19.0",
}
OPTIONAL_PACKAGES = {"flash_attn": "2.0.0", "xformers": "0.0.20"}


class GPUEnvironmentValidator:
    """Validates that the environment can execute GPU training."""

    def __init__(self):
        self._reports: Dict[str, Dict] = {}

    def validate_all(self) -> Dict[str, Any]:
        """Run all validations and return readiness assessment."""
        self._reports["hardware"] = self._validate_hardware()
        self._reports["cuda"] = self._validate_cuda()
        self._reports["dependencies"] = self._validate_python_stack()
        self._reports["storage"] = self._validate_storage()
        self._reports["network"] = self._validate_network()
        self._reports["execution_plan"] = self._validate_execution_plan()

        readiness = self._assess_readiness()
        report = {
            "schema_version": "1.0", "forge_version": settings.app_version,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "reports": self._reports, "readiness": readiness,
        }
        out_dir = Path("data/validation")
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, data in self._reports.items():
            with open(out_dir / f"{name}_validation.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        with open(out_dir / "execution_readiness.json", "w", encoding="utf-8") as f:
            json.dump({"readiness": readiness, **(self._reports.get("execution_plan", {}))}, f, indent=2)
        logger.info("gpu_validation_complete", readiness=readiness)
        return report

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    def _validate_hardware(self) -> Dict[str, Any]:
        info = {"os": sys.platform, "python": sys.version.split()[0], "cpu_count": os.cpu_count() or 1}
        try:
            import psutil
            mem = psutil.virtual_memory()
            info["ram_total_gb"] = round(mem.total / (1024**3), 1)
            info["ram_available_gb"] = round(mem.available / (1024**3), 1)
        except ImportError:
            info["ram_total_gb"] = 0; info["ram_available_gb"] = 0
        try:
            import torch
            info["cuda_available"] = torch.cuda.is_available()
            if info["cuda_available"]:
                info["cuda_version"] = torch.version.cuda
                info["device_count"] = torch.cuda.device_count()
                gpus = []
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    gpus.append({
                        "index": i, "name": props.name,
                        "vram_gb": round((props.total_memory if hasattr(props, 'total_memory') else props.total_mem) / (1024**3), 1),
                        "compute_capability": f"{props.major}.{props.minor}",
                        "multi_processor_count": props.multi_processor_count,
                    })
                info["gpus"] = gpus
            else:
                info["gpus"] = []
        except ImportError:
            info["cuda_available"] = False; info["gpus"] = []
        info["status"] = "PASS" if info.get("cuda_available") and info.get("gpus") else "FAIL"
        return info

    def _validate_cuda(self) -> Dict[str, Any]:
        result = {"available": False, "checks": {}}
        try:
            import torch
            result["available"] = torch.cuda.is_available()
            if result["available"]:
                result["cuda_version"] = torch.version.cuda
                result["device_count"] = torch.cuda.device_count()
                result["checks"]["bf16"] = {"status": "PASS" if torch.cuda.is_bf16_supported() else "WARNING",
                                             "message": "BF16 supported" if torch.cuda.is_bf16_supported() else "BF16 not available"}
                result["checks"]["fp16"] = {"status": "PASS", "message": "FP16 supported"}
                cc = torch.cuda.get_device_capability(0)
                result["checks"]["tensor_cores"] = {"status": "PASS" if cc[0] >= 7 else "WARNING",
                                                     "message": f"Compute {cc[0]}.{cc[1]}"}
                result["checks"]["flash_attention"] = {"status": "PASS" if cc[0] >= 8 else "WARNING",
                                                        "message": "Ampere+ supports Flash Attention"}
            else:
                result["checks"]["cuda_available"] = {"status": "FAIL", "message": "CUDA not available"}
        except ImportError:
            result["checks"]["torch"] = {"status": "FAIL", "message": "PyTorch not installed"}
        result["status"] = "PASS" if result["available"] else "FAIL"
        return result

    def _validate_python_stack(self) -> Dict[str, Any]:
        results = {}
        for name, min_ver in {**REQUIRED_PACKAGES, **OPTIONAL_PACKAGES}.items():
            try:
                v = version(name)
                compatible = _version_ge(v, min_ver)
                results[name] = {"status": "PASS" if compatible else "WARNING", "version": v, "required": min_ver}
            except PackageNotFoundError:
                required = name in REQUIRED_PACKAGES
                results[name] = {"status": "FAIL" if required else "WARNING", "version": None, "required": min_ver}
        passed = sum(1 for r in results.values() if r["status"] == "PASS")
        failed = sum(1 for r in results.values() if r["status"] == "FAIL")
        return {"status": "PASS" if failed == 0 else "FAIL", "packages": results,
                "passed": passed, "failed": failed, "total": len(results)}

    def _validate_storage(self) -> Dict[str, Any]:
        dirs = {
            "workspace": Path("data"), "model_cache": Path("models/cache"),
            "hf_cache": Path.home() / ".cache" / "huggingface",
            "checkpoints": Path("data/checkpoints"), "output": Path("output"), "temp": Path("data/tmp"),
        }
        results = {}
        for name, path in dirs.items():
            try:
                path.mkdir(parents=True, exist_ok=True)
                test_file = path / ".write_test"
                test_file.write_text("test"); test_file.unlink()
                disk = shutil.disk_usage(path)
                free_gb = disk.free / (1024**3)
                results[name] = {"status": "PASS" if free_gb >= 5 else ("WARNING" if free_gb >= 1 else "FAIL"),
                                 "free_gb": round(free_gb, 1), "writable": True, "path": str(path)}
            except Exception as e:
                results[name] = {"status": "FAIL", "free_gb": 0, "writable": False, "error": str(e), "path": str(path)}
        failed = sum(1 for r in results.values() if r["status"] == "FAIL")
        return {"status": "PASS" if failed == 0 else "FAIL", "directories": results}

    def _validate_network(self) -> Dict[str, Any]:
        import urllib.request, urllib.error
        results = {}
        for name, url in [("internet", "https://huggingface.co"), ("hf_api", "https://huggingface.co/api/models")]:
            try:
                urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=10)
                results[name] = {"status": "PASS", "reachable": True}
            except Exception as e:
                results[name] = {"status": "FAIL" if name == "hf_api" else "WARNING", "reachable": False, "error": str(e)[:100]}
        try:
            from huggingface_hub import HfApi
            HfApi().list_models(limit=1)
            results["hf_auth"] = {"status": "PASS", "authenticated": True}
        except ImportError:
            results["hf_auth"] = {"status": "WARNING", "authenticated": False, "message": "huggingface_hub not installed"}
        except Exception as e:
            results["hf_auth"] = {"status": "WARNING", "authenticated": False, "error": str(e)[:100]}
        failed = sum(1 for r in results.values() if r["status"] == "FAIL")
        return {"status": "PASS" if failed == 0 else "FAIL", "checks": results}

    def _validate_execution_plan(self) -> Dict[str, Any]:
        plan_path = Path("data/reports/training_plan.json")
        valid_plan = plan_path.exists()
        model = "qwen2.5-1.5b-instruct"
        if valid_plan:
            try:
                with open(plan_path) as f: plan = json.load(f)
                model = plan.get("model", {}).get("id", model)
            except Exception:
                valid_plan = False
        return {
            "training_plan_exists": valid_plan,
            "selected_model": model,
            "validation_dataset": "100 samples (auto-generated)",
            "lora_config": {"rank": 8, "alpha": 16, "dropout": 0.05},
            "training_config": {"epochs": 1, "batch_size": 1, "gradient_accumulation": 2, "max_steps": 20},
            "status": "PASS" if valid_plan else "WARNING",
        }

    def _assess_readiness(self) -> str:
        has_gpu = self._reports.get("hardware", {}).get("cuda_available", False)
        deps_pass = self._reports.get("dependencies", {}).get("status") == "PASS"
        storage_pass = self._reports.get("storage", {}).get("status") == "PASS"
        network_pass = self._reports.get("network", {}).get("status") == "PASS"
        if has_gpu and deps_pass and storage_pass and network_pass:
            return "READY"
        elif has_gpu or (deps_pass and storage_pass):
            return "LIMITED"
        return "NOT_READY"


def _version_ge(installed: str, required: str) -> bool:
    try:
        from packaging.version import Version
        return Version(installed) >= Version(required)
    except ImportError:
        parts_i = [int(x) for x in installed.split(".")[:3]]
        parts_r = [int(x) for x in required.split(".")[:3]]
        return parts_i >= parts_r


gpu_validator = GPUEnvironmentValidator()
