"""
Cache Manager — filesystem cache for downloaded model assets.

Stores downloaded models in a structured cache with revision isolation.
Never overwrites verified cache. Supports invalidation and cleanup.
"""

import json
import shutil
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Optional

from app.core import get_logger
from app.acquisition.exceptions import CacheError

logger = get_logger("app.acquisition.cache")

CACHE_SCHEMA_VERSION = "1.0"


class CacheManager:
    """Manages the model asset cache.

    Cache layout:
        models/cache/{model_id}/{revision}/
            weights/    — model weight files (.safetensors)
            config/     — model configuration
            tokenizer/  — tokenizer files
            metadata/   — cache metadata and manifests
            downloads/  — temporary download staging

    Downloaded assets are never mixed across revisions.
    """

    def __init__(self, cache_root: Path):
        self._root = cache_root
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    def cache_path(self, model_id: str, revision: str = "main") -> Path:
        """Return the cache directory for a model/revision."""
        return self._root / model_id / revision

    def is_cached(self, model_id: str, revision: str = "main") -> bool:
        """Check if a model is fully cached and verified."""
        path = self.cache_path(model_id, revision)
        manifest = path / "download_manifest.json"
        if not manifest.exists():
            return False
        try:
            with open(manifest) as f:
                data = json.load(f)
            return data.get("status") == "verified"
        except Exception:
            return False

    def write_cache_manifest(self, model_id: str, revision: str,
                             hashes: Dict[str, str], asset_meta: dict) -> Path:
        """Write the cache manifest after successful verification.

        Args:
            model_id: Registry model ID.
            revision: Model revision.
            hashes: SHA256 hashes for all cached files.
            asset_meta: Resolved asset metadata.

        Returns:
            Path to the written manifest.
        """
        cache_dir = self.cache_path(model_id, revision)
        cache_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "model_id": model_id,
            "revision": revision,
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "status": "verified",
            "files": hashes,
            "asset_metadata": asset_meta,
        }
        manifest_path = cache_dir / "cache_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info("cache_manifest_written", model_id=model_id, files=len(hashes))
        return manifest_path

    def invalidate(self, model_id: str, revision: str = "main") -> None:
        """Invalidate a cached model — removes it from cache."""
        cache_dir = self.cache_path(model_id, revision)
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            logger.info("cache_invalidated", model_id=model_id, revision=revision)

    def get_cache_info(self, model_id: str, revision: str = "main") -> Optional[dict]:
        """Get cache information for a model."""
        manifest = self.cache_path(model_id, revision) / "cache_manifest.json"
        if not manifest.exists():
            return None
        with open(manifest) as f:
            return json.load(f)

    def list_cached_models(self) -> list[str]:
        """List all model IDs currently cached."""
        if not self._root.is_dir():
            return []
        models = []
        for d in self._root.iterdir():
            if d.is_dir() and not d.name.startswith("."):
                models.append(d.name)
        return models

    def total_cache_size(self) -> int:
        """Return total cache size in bytes."""
        total = 0
        for f in self._root.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
        return total

    def cleanup(self) -> int:
        """Remove incomplete or unverified cached models. Returns bytes freed."""
        freed = 0
        for model_dir in self._root.iterdir():
            if not model_dir.is_dir():
                continue
            for rev_dir in model_dir.iterdir():
                if not rev_dir.is_dir():
                    continue
                manifest = rev_dir / "cache_manifest.json"
                if not manifest.exists():
                    size = sum(f.stat().st_size for f in rev_dir.rglob("*") if f.is_file())
                    shutil.rmtree(rev_dir)
                    freed += size
        logger.info("cache_cleanup_complete", freed_bytes=freed)
        return freed


# Singleton
_cache_root = Path(__file__).parent.parent.parent / "models" / "cache"
cache_manager = CacheManager(_cache_root)
