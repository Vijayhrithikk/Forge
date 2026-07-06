"""Provider Capability Discovery — reports provider capabilities."""
from typing import Dict, Any
from app.providers.registry import provider_registry

class CapabilityDiscovery:
    def discover(self, provider_name: str) -> Dict[str, Any]:
        caps = provider_registry.capabilities(provider_name)
        return {"name": caps.name, "version": caps.version, "gpu_types": caps.gpu_types,
                "cuda_versions": caps.cuda_versions, "max_disk_gb": caps.max_disk_gb,
                "streaming": caps.streaming_support, "checkpoint": caps.checkpoint_support,
                "artifact_download": caps.artifact_download}
    def all(self) -> Dict: return provider_registry.all_capabilities()
