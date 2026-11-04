"""Custom exceptions for the models module."""

from typing import Optional


class DetectionError(Exception):
    """Raised when ensemble detection fails and cannot produce a valid result."""

    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.original_error = original_error


class ModelLoadError(Exception):
    """Raised when a model fails to load."""

    def __init__(self, model_name: str, message: str, original_error: Optional[Exception] = None):
        super().__init__(f"Failed to load {model_name}: {message}")
        self.model_name = model_name
        self.original_error = original_error


class ModelInferenceError(Exception):
    """Raised when model inference fails."""

    def __init__(self, model_name: str, message: str, original_error: Optional[Exception] = None):
        super().__init__(f"Inference failed for {model_name}: {message}")
        self.model_name = model_name
        self.original_error = original_error