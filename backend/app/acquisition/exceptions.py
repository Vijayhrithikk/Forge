"""Acquisition domain exceptions."""

class AcquisitionError(Exception):
    """Base exception for acquisition failures."""
    def __init__(self, message: str, recoverable: bool = False, error_code: str = "ACQ_ERROR"):
        self.message = message
        self.recoverable = recoverable
        self.error_code = error_code
        super().__init__(message)

    def to_dict(self):
        return {"error_code": self.error_code, "message": self.message, "recoverable": self.recoverable}


class RegistryResolutionError(AcquisitionError):
    def __init__(self, model_id: str, reason: str):
        super().__init__(f"Cannot resolve model '{model_id}': {reason}", False, "ACQ_REGISTRY_RESOLUTION")


class DownloadError(AcquisitionError):
    def __init__(self, resource: str, reason: str = ""):
        super().__init__(f"Download failed for '{resource}': {reason}", True, "ACQ_DOWNLOAD")


class IntegrityError(AcquisitionError):
    def __init__(self, path: str, expected: str, actual: str):
        super().__init__(f"Integrity check failed for '{path}': expected {expected}, got {actual}", False, "ACQ_INTEGRITY")


class CacheError(AcquisitionError):
    def __init__(self, reason: str):
        super().__init__(f"Cache error: {reason}", False, "ACQ_CACHE")
