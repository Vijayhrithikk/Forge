"""Remote Workspace Manager — filesystem preparation on provider."""
from typing import Dict, Any

class RemoteWorkspaceManager:
    SUBDIRS = ["workspace", "dataset", "output", "checkpoints", "logs", "artifacts", "temp"]
    def prepare(self, provider_name: str, project_id: str, base: str = "/workspace") -> Dict[str, Any]:
        paths = {}
        for d in self.SUBDIRS:
            paths[d] = f"{base}/{project_id}/{d}"
        return {"provider": provider_name, "project_id": project_id, "root": f"{base}/{project_id}", "paths": paths, "ready": True}
