class RecoveryError(Exception):
    def __init__(self, msg, recoverable=True, error_code="RECOV_ERROR"):
        self.message = msg; self.recoverable = recoverable; self.error_code = error_code; super().__init__(msg)
    def to_dict(self): return {"error_code": self.error_code, "message": self.message, "recoverable": self.recoverable}

class OOMError(RecoveryError):
    def __init__(self): super().__init__("CUDA out of memory", True, "RECOV_OOM")
class CheckpointCorruptionError(RecoveryError):
    def __init__(self, path=""): super().__init__(f"Checkpoint corrupt: {path}", False, "RECOV_CKPT_CORRUPT")
class UnrecoverableError(RecoveryError):
    def __init__(self, reason=""): super().__init__(f"Unrecoverable: {reason}", False, "RECOV_UNRECOVERABLE")
