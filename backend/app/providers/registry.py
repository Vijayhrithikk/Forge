"""Provider Registry — discovers and selects execution providers."""
from typing import Dict, List, Optional
from app.providers.base import BaseProvider, ProviderCapabilities
from app.providers.runpod import RunPodProvider

class ProviderRegistry:
    def __init__(self):
        self._providers: Dict[str, BaseProvider] = {}
        self._register_builtins()

    def _register_builtins(self):
        from app.providers.runpod import RunPodProvider
        from app.providers.simulation import SimulationProvider
        self._providers["runpod"] = RunPodProvider()
        self._providers["simulation"] = SimulationProvider()

    def list_providers(self) -> List[str]: return list(self._providers.keys())
    def get(self, name: str) -> BaseProvider:
        if name not in self._providers: raise KeyError(f"Provider '{name}' not registered")
        return self._providers[name]
    def capabilities(self, name: str) -> ProviderCapabilities: return self.get(name).capabilities()
    def all_capabilities(self) -> Dict[str, ProviderCapabilities]:
        return {n: p.capabilities() for n, p in self._providers.items()}

provider_registry = ProviderRegistry()
