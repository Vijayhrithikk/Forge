"""Provider exceptions — never leak provider internals into Forge Runtime."""
class ProviderError(Exception):
    def __init__(self, msg, recoverable=True, code="PROVIDER_ERROR"):
        self.message = msg; self.recoverable = recoverable; self.error_code = code; super().__init__(msg)
    def to_dict(self): return {"error_code": self.error_code, "message": self.message, "recoverable": self.recoverable}
class ProviderAuthError(ProviderError):
    def __init__(self, provider): super().__init__(f"Auth failed for {provider}", True, "PROVIDER_AUTH")
class ProviderUnavailableError(ProviderError):
    def __init__(self, provider, reason=""): super().__init__(f"{provider} unavailable: {reason}", True, "PROVIDER_UNAVAILABLE")
class ProviderGPUError(ProviderError):
    def __init__(self, gpu_type): super().__init__(f"GPU {gpu_type} not available", False, "PROVIDER_GPU")
