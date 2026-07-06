class PerformanceError(Exception):
    def __init__(self, msg, recoverable=True): self.message = msg; self.recoverable = recoverable; super().__init__(msg)
