"""Detection-related Pydantic schemas."""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Literal
from datetime import datetime, timezone
from uuid import UUID, uuid4


class DetectionRequest(BaseModel):
    """Request schema for single image detection (multipart/form-data handled separately)."""
    model_config = ConfigDict(protected_namespaces=())
    pass  # File upload handled by FastAPI File/UploadFile


class Probabilities(BaseModel):
    """Class probabilities."""
    model_config = ConfigDict(protected_namespaces=())
    real: float = Field(ge=0.0, le=1.0, description="Probability of real image")
    fake: float = Field(ge=0.0, le=1.0, description="Probability of fake image")


class DetectionResult(BaseModel):
    """Single detection result."""
    model_config = ConfigDict(protected_namespaces=())
    prediction: Literal["real", "fake", "unknown"] = Field(description="Final prediction")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in prediction")
    probabilities: Probabilities = Field(description="Per-class probabilities")
    model_used: str = Field(description="Model or ensemble used")
    ensemble_active: bool = Field(description="Whether multiple distinct models were used in ensemble")
    models_used: list[str] = Field(description="List of model names that contributed to prediction")
    processing_time_ms: float = Field(description="Inference time in milliseconds")
    face_detected: bool = Field(default=False, description="Whether a face was detected")
    heatmap_url: Optional[str] = Field(default=None, description="URL to Grad-CAM heatmap")


class DetectionResponse(BaseModel):
    """Response schema for detection endpoint."""
    model_config = ConfigDict(protected_namespaces=())
    request_id: UUID = Field(default_factory=uuid4, description="Unique request identifier")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Request timestamp")
    result: DetectionResult = Field(description="Detection result")
    filename: str = Field(description="Original filename")


class BatchDetectionRequest(BaseModel):
    """Request schema for batch detection (not implemented in MVP)."""
    model_config = ConfigDict(protected_namespaces=())
    pass  # Multiple files handled by FastAPI


class BatchDetectionResponse(BaseModel):
    """Response schema for batch detection."""
    model_config = ConfigDict(protected_namespaces=())
    request_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    results: list[DetectionResult] = Field(description="List of detection results")
    total_processing_time_ms: float = Field(description="Total processing time")


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_config = ConfigDict(protected_namespaces=())
    model_name: str = Field(description="Name of the model")
    version: str = Field(description="Model version")
    architecture: str = Field(description="Model architecture")
    input_size: tuple[int, int] = Field(description="Expected input size (H, W)")
    num_classes: int = Field(description="Number of output classes")
    classes: list[str] = Field(description="Class names")
    accuracy: Optional[float] = Field(default=None, description="Reported accuracy on benchmark")
    description: str = Field(description="Model description")


class HealthResponse(BaseModel):
    """Health check response."""
    model_config = ConfigDict(protected_namespaces=())
    status: Literal["healthy", "degraded", "unhealthy"] = Field(description="Service status")
    version: str = Field(description="API version")
    uptime_seconds: float = Field(description="Service uptime in seconds")
    models_loaded: dict[str, bool] = Field(description="Model loading status")
    device: str = Field(description="Inference device")


class ErrorResponse(BaseModel):
    """Error response schema."""
    model_config = ConfigDict(protected_namespaces=())
    error: str = Field(description="Error type")
    message: str = Field(description="Human-readable error message")
    details: Optional[dict] = Field(default=None, description="Additional error details")
    request_id: Optional[UUID] = Field(default=None, description="Request ID for debugging")


# Example responses for OpenAPI documentation
DETECTION_RESPONSE_EXAMPLE = {
    "request_id": "550e8400-e29b-41d4-a716-446655440000",
    "timestamp": "2024-01-15T10:30:00Z",
    "result": {
        "prediction": "fake",
        "confidence": 0.92,
        "probabilities": {"real": 0.08, "fake": 0.92},
        "model_used": "ensemble",
        "processing_time_ms": 145.2,
        "face_detected": True,
        "heatmap_url": "/api/v1/heatmap/550e8400-e29b-41d4-a716-446655440000"
    },
    "filename": "test_image.jpg"
}

HEALTH_RESPONSE_EXAMPLE = {
    "status": "healthy",
    "version": "1.0.0",
    "uptime_seconds": 3600.5,
    "models_loaded": {
        "efficientnet": True,
        "mesonet": True
    },
    "device": "cpu"
}