class CheckpointError(Exception):
    def __init__(self, msg, recoverable=False): self.message = msg; self.recoverable = recoverable; super().__init__(msg)
