"""CUDA validator — GPU access, precision, compute capability."""
from typing import Dict, Any

def validate_cuda() -> Dict[str, Any]:
    result = {"cuda_available": False, "checks": {}}
    try:
        import torch
        result["cuda_available"] = torch.cuda.is_available()
        if result["cuda_available"]:
            result["cuda_version"] = torch.version.cuda or "unknown"
            result["device_count"] = torch.cuda.device_count()
            result["checks"]["cuda_runtime"] = {"status": "PASS", "message": f"CUDA {result['cuda_version']}"}
            result["checks"]["bf16"] = {"status": "PASS" if torch.cuda.is_bf16_supported() else "WARNING",
                                         "message": "BF16 supported" if torch.cuda.is_bf16_supported() else "BF16 not supported"}
            result["checks"]["fp16"] = {"status": "PASS", "message": "FP16 supported"}
            result["checks"]["tensor_cores"] = {"status": "PASS" if _has_tensor_cores() else "WARNING",
                                                  "message": "Tensor Cores available" if _has_tensor_cores() else "No Tensor Cores"}
        else:
            result["checks"]["cuda_available"] = {"status": "FAIL", "message": "CUDA not available"}
    except ImportError:
        result["checks"]["torch"] = {"status": "FAIL", "message": "PyTorch not installed"}
    return result

def _has_tensor_cores() -> bool:
    try:
        import torch
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            return cap[0] >= 7
    except Exception:
        pass
    return False
