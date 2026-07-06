class ExecutionError(Exception):
    def __init__(self, msg, recoverable=False): self.message = msg; self.recoverable = recoverable; super().__init__(msg)
class TargetDetectionError(ExecutionError):
    def __init__(self, msg): super().__init__(msg, False)
