"""Network validator — internet connectivity + HuggingFace access."""
import urllib.request
import urllib.error
from typing import Dict, Any

def validate_network(timeout: int = 5) -> Dict[str, Any]:
    result = {}
    result["internet"] = _check_url("https://huggingface.co", timeout)
    result["huggingface_api"] = _check_url("https://huggingface.co/api/models", timeout)
    result["hf_authentication"] = _check_hf_auth()
    return result

def _check_url(url: str, timeout: int) -> Dict[str, Any]:
    try:
        req = urllib.request.Request(url, method="HEAD")
        urllib.request.urlopen(req, timeout=timeout)
        return {"status": "PASS", "message": f"Reachable: {url}"}
    except Exception as e:
        return {"status": "FAIL" if "huggingface" in url else "WARNING", "message": str(e)[:120]}

def _check_hf_auth() -> Dict[str, Any]:
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.list_models(limit=1)
        return {"status": "PASS", "message": "HF API accessible"}
    except ImportError:
        return {"status": "WARNING", "message": "huggingface_hub not installed"}
    except Exception as e:
        return {"status": "WARNING", "message": str(e)[:120]}
