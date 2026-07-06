"""Preparation Runtime exceptions."""

class PreparationError(Exception):
    def __init__(self, message: str, recoverable: bool = False, error_code: str = "PREP_ERROR"):
        self.message = message
        self.recoverable = recoverable
        self.error_code = error_code
        super().__init__(message)

    def to_dict(self):
        return {"error_code": self.error_code, "message": self.message, "recoverable": self.recoverable}


class UnsupportedPrecision(PreparationError):
    def __init__(self, precision: str, supported: list):
        super().__init__(f"Precision '{precision}' not supported. Supported: {supported}", False, "PREP_PRECISION")

class UnsupportedQuantization(PreparationError):
    def __init__(self, quant: str):
        super().__init__(f"Quantization '{quant}' not supported by registry.", False, "PREP_QUANT")

class DeviceNotAvailable(PreparationError):
    def __init__(self, device: str, reason: str = ""):
        super().__init__(f"Device '{device}' not available: {reason}", False, "PREP_DEVICE")

class MemoryInsufficient(PreparationError):
    def __init__(self, required_gb: float, available_gb: float):
        super().__init__(f"Insufficient memory: need {required_gb:.1f}GB, have {available_gb:.1f}GB", True, "PREP_MEMORY")
