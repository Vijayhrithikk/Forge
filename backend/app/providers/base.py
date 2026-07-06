"""Execution Provider Interface — abstracts remote execution lifecycle."""
from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, Optional
from dataclasses import dataclass

@dataclass
class ProviderCapabilities:
    name: str; version: str = "1.0"
    gpu_types: list = None; cuda_versions: list = None
    max_disk_gb: float = 100; persistent_storage: bool = False
    streaming_support: bool = True; checkpoint_support: bool = True
    artifact_download: bool = True; internet: bool = True

class BaseProvider(ABC):
    """Abstract provider — owns environment lifecycle, never training logic."""
    @abstractmethod
    def name(self) -> str: ...
    @abstractmethod
    def capabilities(self) -> ProviderCapabilities: ...
    @abstractmethod
    def health(self) -> Dict[str, Any]: ...
    @abstractmethod
    def prepare(self, workspace: Dict) -> Dict[str, Any]: ...
    @abstractmethod
    def upload(self, paths: Dict) -> Dict[str, Any]: ...
    @abstractmethod
    def execute(self, package: Dict, on_event: Optional[Callable] = None) -> Dict[str, Any]: ...
    @abstractmethod
    def download(self, remote_paths: list) -> Dict[str, Any]: ...
    @abstractmethod
    def cleanup(self) -> Dict[str, Any]: ...
