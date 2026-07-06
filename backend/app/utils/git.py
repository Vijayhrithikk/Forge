"""
Utility to extract git metadata for observability endpoints.
"""

import subprocess
import os
from functools import lru_cache


@lru_cache(maxsize=1)
def get_git_commit() -> str:
    """Return the short git commit hash of the current checkout.

    Falls back to 'unknown' if git is not available or the
    repository is not a git checkout (e.g., Docker build context).
    """
    try:
        # Run from the backend directory so we find the repo
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        result = subprocess.run(
            ["git", "rev-parse", "--short=8", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return "unknown"
