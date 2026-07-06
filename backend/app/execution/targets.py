"""Execution Target Detection — auto-detect local, Kaggle, RunPod, Vast.ai, container."""
import os, platform
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ExecutionTarget:
    name: str; env_type: str; workspace_prefix: str
    cache_prefix: str; output_prefix: str; is_cloud: bool = False

TARGETS = {
    "local": ExecutionTarget("local", "local", ".", "models/cache", "output"),
    "kaggle": ExecutionTarget("kaggle", "notebook", "/kaggle/working", "/kaggle/working/cache", "/kaggle/working/output", True),
    "runpod": ExecutionTarget("runpod", "cloud", "/workspace", "/workspace/cache", "/workspace/output", True),
    "vastai": ExecutionTarget("vastai", "cloud", "/workspace", "/workspace/cache", "/workspace/output", True),
    "github_codespaces": ExecutionTarget("github_codespaces", "container", "/workspaces", "/workspaces/cache", "/workspaces/output"),
}

def detect_target() -> ExecutionTarget:
    env = os.environ
    if env.get("KAGGLE_KERNEL_RUN_TYPE"): return TARGETS["kaggle"]
    if env.get("RUNPOD_POD_ID"): return TARGETS["runpod"]
    if env.get("VAST_CONTAINER_ID"): return TARGETS["vastai"]
    if env.get("CODESPACES"): return TARGETS["github_codespaces"]
    if os.path.exists("/.dockerenv"): return ExecutionTarget("docker", "container", "/app", "/app/models/cache", "/app/output")
    return TARGETS["local"]

def target_info() -> Dict[str, Any]:
    t = detect_target()
    return {"name": t.name, "type": t.env_type, "workspace": t.workspace_prefix,
            "cache": t.cache_prefix, "output": t.output_prefix, "is_cloud": t.is_cloud,
            "os": platform.system(), "hostname": platform.node()}
