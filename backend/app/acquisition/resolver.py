"""
Registry Resolver — resolves model requirements from registry metadata.

Never hardcodes HuggingFace repositories. Everything comes from registry.json.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from pathlib import Path

from app.core import get_logger
from app.engines.training.registry import model_registry, ModelEntry
from app.acquisition.exceptions import RegistryResolutionError

logger = get_logger("app.acquisition.resolver")


@dataclass
class ResolvedAsset:
    """A fully resolved model asset ready for acquisition."""
    model_id: str
    huggingface_id: str
    revision: str
    architecture: str
    tokenizer_name: str
    recommended_precision: str
    target_modules: List[str]
    context_length: int
    files_to_acquire: List[str]  # Relative file paths to download
    metadata: Dict[str, Any]


class RegistryResolver:
    """Resolves a model ID into a complete acquisition plan.

    Reads from the model registry and produces a ResolvedAsset
    containing everything the Downloader needs to acquire the model.
    """

    # Default revision for all models
    DEFAULT_REVISION = "main"

    # Standard files to acquire for any HuggingFace model
    STANDARD_FILES = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
    ]

    def resolve(self, model_id: str, revision: Optional[str] = None) -> ResolvedAsset:
        """Resolve a model ID into a complete acquisition plan.

        Args:
            model_id: Registry model ID (e.g., 'qwen2.5-1.5b-instruct').
            revision: Optional specific revision. Defaults to 'main'.

        Returns:
            ResolvedAsset with all information needed for download.

        Raises:
            RegistryResolutionError: If the model is not found or resolution fails.
        """
        entry = model_registry.get_model(model_id)
        rev = revision or self.DEFAULT_REVISION

        # Determine weight file pattern based on architecture
        weight_files = self._weight_files_for_family(entry.family)

        files = self.STANDARD_FILES + weight_files

        resolved = ResolvedAsset(
            model_id=entry.id,
            huggingface_id=entry.huggingface_id,
            revision=rev,
            architecture=entry.architecture,
            tokenizer_name=entry.tokenizer,
            recommended_precision=entry.recommended_precision,
            target_modules=entry.lora_defaults.get("target_modules", []),
            context_length=entry.context_length,
            files_to_acquire=files,
            metadata={
                "parameters": entry.parameters,
                "parameters_display": entry.parameters_display,
                "license": entry.license,
                "tags": entry.tags,
                "training_notes": entry.training_notes,
            },
        )

        logger.info("model_resolved", model_id=model_id, hf_id=entry.huggingface_id,
                     revision=rev, files=len(files))
        return resolved

    def _weight_files_for_family(self, family: str) -> List[str]:
        """Return the expected weight file pattern for a model family."""
        # All supported families use safetensors
        return ["model.safetensors.index.json"]
        # Note: actual shard files (model-00001-of-XXXXX.safetensors)
        # are discovered dynamically during download from the index file.


# Singleton
registry_resolver = RegistryResolver()
