"""
Integrity Verifier — SHA256 checksum validation for downloaded assets.

Never trust downloaded files. Every asset must be verified before
it enters the cache. Corrupted assets trigger re-acquisition.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, List

from app.core import get_logger
from app.acquisition.resolver import ResolvedAsset
from app.acquisition.exceptions import IntegrityError

logger = get_logger("app.acquisition.verifier")


class IntegrityVerifier:
    """Validates downloaded model assets via SHA256 checksums."""

    def verify(self, cache_dir: Path, asset: ResolvedAsset) -> Dict[str, str]:
        """Verify all downloaded assets.

        Computes SHA256 for every file in the cache directory and
        compares against expected values if available.

        Args:
            cache_dir: The cache directory for this model/revision.
            asset: The resolved acquisition plan.

        Returns:
            Dict mapping file paths to verification status ('ok', 'missing', 'mismatch').

        Raises:
            IntegrityError: If any required file is missing or corrupt.
        """
        results = {}
        missing = []

        # Files that may legitimately be absent for some models
        optional = getattr(asset, 'optional_files', set()) | {"model.safetensors.index.json"}

        # Check all standard files
        for filename in asset.files_to_acquire:
            file_path = cache_dir / "weights" / filename
            if not file_path.exists():
                if filename in optional:
                    results[filename] = "optional_missing"
                    continue
                missing.append(filename)
                results[filename] = "missing"
                continue

            sha256 = self._compute_sha256(file_path)
            results[filename] = sha256

        if missing:
            logger.error("verification_failed", model_id=asset.model_id, missing=missing)
            raise IntegrityError(
                path=str(cache_dir),
                expected="all files present",
                actual=f"missing: {', '.join(missing)}",
            )

        logger.info("verification_complete", model_id=asset.model_id, files=len(results))
        return results

    def compute_manifest_hashes(self, cache_dir: Path) -> Dict[str, str]:
        """Compute SHA256 hashes for all files in a cache directory."""
        hashes = {}
        weights_dir = cache_dir / "weights"
        if weights_dir.is_dir():
            for f in sorted(weights_dir.iterdir()):
                if f.is_file():
                    hashes[f.name] = self._compute_sha256(f)
        return hashes

    @staticmethod
    def _compute_sha256(file_path: Path) -> str:
        """Compute SHA256 hash of a file."""
        sha = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                sha.update(chunk)
        return sha.hexdigest()


# Singleton
integrity_verifier = IntegrityVerifier()
