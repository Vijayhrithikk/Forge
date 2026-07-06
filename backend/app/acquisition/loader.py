"""
Model Loader — loads models and tokenizers from verified cache.

Never loads directly from the internet. Internet access belongs
only to the Acquisition domain. Loader receives verified local assets.

Uses Transformers AutoConfig, AutoTokenizer, AutoModelForCausalLM.
Architecture-specific code is avoided — the registry drives selection.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from app.core import get_logger
from app.acquisition.resolver import ResolvedAsset
from app.acquisition.manifest import (
    generate_model_manifest, generate_tokenizer_manifest, save_manifest,
)
from app.acquisition.cache import cache_manager

logger = get_logger("app.acquisition.loader")


class ModelLoadResult:
    """Result of loading a model from cache."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.config = None
        self.generation_config = None
        self.model_manifest: Dict[str, Any] = {}
        self.tokenizer_manifest: Dict[str, Any] = {}
        self.load_duration: float = 0.0
        self.validation_results: Dict[str, Any] = {}


class ModelLoader:
    """Loads transformers models and tokenizers from verified cache."""

    def load(
        self,
        asset: ResolvedAsset,
        device: str = "cpu",
        precision: str = "bf16",
    ) -> ModelLoadResult:
        """Load a model and tokenizer from the verified cache.

        Args:
            asset: The resolved acquisition plan.
            device: Target device ('cpu', 'cuda', 'cuda:0').
            precision: Target precision ('fp32', 'fp16', 'bf16').

        Returns:
            ModelLoadResult with model, tokenizer, config, manifests.
        """
        result = ModelLoadResult()
        t0 = time.time()

        cache_dir = cache_manager.cache_path(asset.model_id, asset.revision)
        weights_dir = cache_dir / "weights"

        # Load config
        try:
            from transformers import AutoConfig
            result.config = AutoConfig.from_pretrained(str(weights_dir))
            logger.info("config_loaded", model_id=asset.model_id)
        except ImportError:
            logger.warning("transformers_not_installed",
                           model_id=asset.model_id,
                           message="Install 'transformers' to load models.")
            result.load_duration = time.time() - t0
            return result

        # Load tokenizer
        try:
            from transformers import AutoTokenizer
            result.tokenizer = AutoTokenizer.from_pretrained(str(weights_dir))
            logger.info("tokenizer_loaded", model_id=asset.model_id)
        except Exception as e:
            logger.error("tokenizer_load_failed", model_id=asset.model_id, error=str(e))

        # Load model
        try:
            import torch
            torch_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}.get(
                precision, torch.float32
            )
            from transformers import AutoModelForCausalLM
            result.model = AutoModelForCausalLM.from_pretrained(
                str(weights_dir),
                config=result.config,
                torch_dtype=torch_dtype,
                device_map=device if device != "cpu" else None,
                low_cpu_mem_usage=True,
            )
            logger.info("model_loaded", model_id=asset.model_id, device=device,
                        precision=precision)
        except Exception as e:
            logger.error("model_load_failed", model_id=asset.model_id, error=str(e))

        # Validate
        result.validation_results = self._validate(result, asset)

        # Generate manifests
        verify_hashes = self._compute_cache_hashes(weights_dir)
        arch_meta = self._extract_architecture_meta(result.config) if result.config else {}
        result.model_manifest = generate_model_manifest(
            asset, cache_dir, verify_hashes, result.load_duration, arch_meta,
        )
        result.tokenizer_manifest = generate_tokenizer_manifest(
            asset, cache_dir,
            vocab_size=result.tokenizer.vocab_size if result.tokenizer else 0,
            fast_tokenizer=getattr(result.tokenizer, "is_fast", False) if result.tokenizer else False,
        )

        # Save manifests
        save_manifest(result.model_manifest, cache_dir / "metadata" / "model_manifest.json")
        save_manifest(result.tokenizer_manifest, cache_dir / "metadata" / "tokenizer_manifest.json")

        result.load_duration = time.time() - t0
        logger.info("model_runtime_ready", model_id=asset.model_id,
                     duration=round(result.load_duration, 2))
        return result

    def _validate(self, result: ModelLoadResult, asset: ResolvedAsset) -> Dict[str, Any]:
        """Validate loaded model against registry metadata."""
        checks = {}

        if result.config:
            checks["config_loaded"] = "PASS"
            checks["architecture"] = (
                "PASS" if result.config.architectures and
                asset.architecture in str(result.config.architectures)
                else "WARNING"
            )
        else:
            checks["config_loaded"] = "WARNING"
            checks["architecture"] = "WARNING"

        if result.tokenizer:
            checks["tokenizer_loaded"] = "PASS"
            if result.config:
                checks["vocab_size_match"] = (
                    "PASS" if result.tokenizer.vocab_size == result.config.vocab_size
                    else "WARNING"
                )
        else:
            checks["tokenizer_loaded"] = "WARNING"

        if result.model:
            checks["model_loaded"] = "PASS"
        else:
            checks["model_loaded"] = "WARNING"

        return checks

    @staticmethod
    def _compute_cache_hashes(weights_dir: Path) -> Dict[str, str]:
        import hashlib
        hashes = {}
        if weights_dir.is_dir():
            for f in sorted(weights_dir.iterdir()):
                if f.is_file() and f.suffix in (".json", ".safetensors", ".bin", ".model"):
                    sha = hashlib.sha256()
                    with open(f, "rb") as fh:
                        while chunk := fh.read(8192):
                            sha.update(chunk)
                    hashes[f.name] = sha.hexdigest()
        return hashes

    @staticmethod
    def _extract_architecture_meta(config) -> Dict[str, Any]:
        meta = {}
        for attr in ["hidden_size", "num_hidden_layers", "num_attention_heads",
                      "intermediate_size", "vocab_size", "max_position_embeddings"]:
            if hasattr(config, attr):
                meta[attr] = getattr(config, attr)
        return meta


# Singleton
model_loader = ModelLoader()
