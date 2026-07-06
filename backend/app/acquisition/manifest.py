"""
Asset Manifest Generator — produces model_manifest.json and tokenizer_manifest.json.

Documented, versioned records of every acquired asset. These manifests
enable reproducibility and recovery.
"""

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from app.core import settings, get_logger
from app.acquisition.resolver import ResolvedAsset

logger = get_logger("app.acquisition.manifest")

MANIFEST_SCHEMA_VERSION = "1.0"


def generate_model_manifest(
    asset: ResolvedAsset,
    cache_dir: Path,
    hashes: Dict[str, str],
    load_duration: float = 0.0,
    architecture_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate a model_manifest.json for acquired model assets.

    Args:
        asset: The resolved acquisition plan.
        cache_dir: The cache directory.
        hashes: SHA256 hashes from verification.
        load_duration: Time to load the model (if loaded).
        architecture_meta: Architecture-specific metadata (hidden size, layers, etc.).

    Returns:
        Manifest dict (to be serialized to JSON).
    """
    total_size = sum(
        (cache_dir / "weights" / f).stat().st_size
        for f in hashes
        if (cache_dir / "weights" / f).exists()
    )

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "forge_version": settings.app_version,
        "registry_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "model_id": asset.model_id,
            "huggingface_id": asset.huggingface_id,
            "revision": asset.revision,
            "architecture": asset.architecture,
            "context_length": asset.context_length,
            "recommended_precision": asset.recommended_precision,
        },
        "files": {
            "hashes": hashes,
            "total_size_bytes": total_size,
        },
        "architecture": architecture_meta or {},
        "verification": {
            "status": "verified",
            "verified_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    return manifest


def generate_tokenizer_manifest(
    asset: ResolvedAsset,
    cache_dir: Path,
    vocab_size: int = 0,
    special_tokens: Optional[Dict[str, str]] = None,
    fast_tokenizer: bool = True,
    chat_template: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate a tokenizer_manifest.json for acquired tokenizer assets.

    Args:
        asset: The resolved acquisition plan.
        cache_dir: The cache directory.
        vocab_size: Vocabulary size.
        special_tokens: Special token mapping (BOS, EOS, PAD, UNK).
        fast_tokenizer: Whether the fast tokenizer is available.
        chat_template: Chat template string (if available).

    Returns:
        Manifest dict.
    """
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "forge_version": settings.app_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tokenizer": {
            "tokenizer_name": asset.tokenizer_name,
            "vocab_size": vocab_size,
            "fast_tokenizer": fast_tokenizer,
            "chat_template": chat_template,
            "special_tokens": special_tokens or {},
            "max_length": asset.context_length,
        },
        "verification": {
            "status": "verified",
            "verified_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    return manifest


def save_manifest(manifest: Dict[str, Any], path: Path) -> Path:
    """Save a manifest to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info("manifest_saved", path=str(path))
    return path
