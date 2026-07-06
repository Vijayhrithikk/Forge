"""Training Runtime exceptions."""

class TrainingError(Exception):
    def __init__(self, message: str, recoverable: bool = False, error_code: str = "TRAIN_ERROR"):
        self.message = message
        self.recoverable = recoverable
        self.error_code = error_code
        super().__init__(message)
    def to_dict(self):
        return {"error_code": self.error_code, "message": self.message, "recoverable": self.recoverable}

class DatasetLoadError(TrainingError):
    def __init__(self, reason: str): super().__init__(f"Dataset load failed: {reason}", False, "TRAIN_DATASET_LOAD")
class TokenizationError(TrainingError):
    def __init__(self, reason: str): super().__init__(f"Tokenization failed: {reason}", False, "TRAIN_TOKENIZATION")
class PEFTInjectionError(TrainingError):
    def __init__(self, reason: str): super().__init__(f"PEFT injection failed: {reason}", False, "TRAIN_PEFT_INJECTION")
class TrainerBuildError(TrainingError):
    def __init__(self, reason: str): super().__init__(f"Trainer build failed: {reason}", False, "TRAIN_TRAINER_BUILD")
class TrainingExecutionError(TrainingError):
    def __init__(self, reason: str): super().__init__(f"Training execution failed: {reason}", True, "TRAIN_EXECUTION")
