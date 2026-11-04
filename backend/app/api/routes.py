"""API routes for deepfake detection."""

import asyncio
import logging
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from threading import Lock
from typing import Optional, Dict, Any, Set

from fastapi import (
    APIRouter,
    File,
    UploadFile,
    HTTPException,
    Depends,
    Request,
    BackgroundTasks,
    Header
)
from fastapi.responses import JSONResponse, StreamingResponse, Response
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import settings
from app.schemas.detection import (
    DetectionResponse,
    HealthResponse,
    ModelInfoResponse,
    ErrorResponse,
    DetectionResult,
    Probabilities,
    BatchDetectionResponse,
)
from app.models.detector import EnsembleDetector
from app.models.exceptions import DetectionError
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix=settings.API_PREFIX, tags=["detection"])

# Rate limiter for detection endpoints - configurable via settings
limiter = Limiter(key_func=get_remote_address)

_start_time = time.time()


# API Key authentication
# Valid API keys are loaded dynamically from settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: Optional[str] = Header(None, alias="X-API-Key")) -> Optional[str]:
    """
    Verify API key if authentication is enabled.

    If no API keys are configured (API_KEYS empty), authentication is disabled.
    If keys are configured, the X-API-Key header must match one of them.

    Returns:
        The API key if valid, None if auth disabled

    Raises:
        HTTPException: 401 if auth enabled but key missing/invalid
    """
    # Load keys dynamically each call (allows runtime config changes)
    api_keys_str = getattr(settings, 'API_KEYS', '')
    api_keys = {key.strip() for key in api_keys_str.split(",") if key.strip()}

    # If no keys configured, auth is disabled
    if not api_keys:
        return None

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required. Provide X-API-Key header."
        )

    if api_key not in api_keys:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )

    return api_key


@dataclass
class _HeatmapCacheEntry:
    """Cache entry with timestamp for TTL support."""
    data: bytes
    timestamp: float = field(default_factory=time.time)


class HeatmapCache:
    """Thread-safe LRU cache with TTL expiry for heatmap images."""

    def __init__(self, max_size: int = 100, ttl_seconds: int = 600):
        self._cache: OrderedDict[str, _HeatmapCacheEntry] = OrderedDict()
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._lock = Lock()

    def get(self, key: str) -> Optional[bytes]:
        """Get value if exists and not expired."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None

            # Check TTL
            if time.time() - entry.timestamp > self._ttl_seconds:
                del self._cache[key]
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return entry.data

    def set(self, key: str, value: bytes) -> None:
        """Set value with current timestamp."""
        with self._lock:
            # Remove if exists to update position
            if key in self._cache:
                del self._cache[key]

            # Evict oldest if at capacity
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)  # FIFO = LRU for OrderedDict

            self._cache[key] = _HeatmapCacheEntry(data=value)

    def cleanup_expired(self) -> int:
        """Remove all expired entries. Returns count of removed entries."""
        with self._lock:
            current_time = time.time()
            expired_keys = [
                key for key, entry in self._cache.items()
                if current_time - entry.timestamp > self._ttl_seconds
            ]
            for key in expired_keys:
                del self._cache[key]
            return len(expired_keys)

    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._cache.clear()


# Initialize cache with settings
_heatmap_cache = HeatmapCache(
    max_size=getattr(settings, 'HEATMAP_CACHE_MAX_SIZE', 100),
    ttl_seconds=getattr(settings, 'HEATMAP_CACHE_TTL_SECONDS', 600)
)


def get_detector(request: Request) -> EnsembleDetector:
    """Dependency to get detector instance from app state."""
    return request.app.state.detector


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check service health and model loading status"
)
@limiter.limit(settings.RATE_LIMIT_DEFAULT)
async def health_check(request: Request, detector: EnsembleDetector = Depends(get_detector)) -> HealthResponse:
    """Health check endpoint."""
    model_info = detector.get_model_info()

    return HealthResponse(
        status="healthy" if model_info["efficientnet"]["loaded"] and model_info["mesonet"]["loaded"] else "degraded",
        version=settings.APP_VERSION,
        uptime_seconds=time.time() - _start_time,
        models_loaded={
            "efficientnet": model_info["efficientnet"]["loaded"],
            "mesonet": model_info["mesonet"]["loaded"],
        },
        device=settings.INFERENCE_DEVICE
    )


@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Model information",
    description="Get information about the loaded models"
)
@limiter.limit(settings.RATE_LIMIT_DEFAULT)
async def model_info(request: Request, detector: EnsembleDetector = Depends(get_detector)) -> ModelInfoResponse:
    """Get model information."""
    model_info = detector.get_model_info()
    eff_loaded = model_info["efficientnet"]["loaded"]
    mes_loaded = model_info["mesonet"]["loaded"]

    # Load accuracy from metadata file if available
    metadata = settings.get_model_metadata()
    accuracy = metadata.get("accuracy") if metadata else None

    return ModelInfoResponse(
        model_name="Deepfake Ensemble Detector",
        version="1.0.0",
        architecture="EfficientNet-B0 + MesoNet Ensemble" if eff_loaded and mes_loaded else "EfficientNet-B0" if eff_loaded else "MesoNet",
        input_size=(224, 224),
        num_classes=2,
        classes=["real", "fake"],
        accuracy=accuracy,
        description=(
            "Ensemble of EfficientNet-B0 (fine-tuned on FaceForensics++) and MesoNet "
            "for robust deepfake detection. Weighted combination: 70% EfficientNet, 30% MesoNet."
            if eff_loaded and mes_loaded else
            "EfficientNet-B0 fine-tuned on FaceForensics++ for deepfake detection."
            if eff_loaded else
            "MesoNet for deepfake detection."
        )
    )


@router.post(
    "/detect",
    response_model=DetectionResponse,
    summary="Detect deepfake",
    description="Upload an image to detect if it's a deepfake/AI-generated",
    responses={
        200: {"description": "Successful detection"},
        400: {"model": ErrorResponse, "description": "Invalid file"},
        401: {"model": ErrorResponse, "description": "Invalid or missing API key"},
        413: {"model": ErrorResponse, "description": "File too large"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    }
)
@limiter.limit(settings.RATE_LIMIT_DETECT)
async def detect_deepfake(
    request: Request,
    file: UploadFile = File(..., description="Image file (JPG, PNG, WebP)"),
    detector: EnsembleDetector = Depends(get_detector),
    api_key: Optional[str] = Depends(verify_api_key)
) -> DetectionResponse:
    """
    Detect if an uploaded image is a deepfake.

    - **file**: Image file (max 10MB, JPG/PNG/WebP)
    - **X-API-Key**: Optional API key header (required if API_KEYS configured)
    - Returns prediction with confidence and probabilities
    """
    request_id = str(uuid.uuid4())

    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Sanitize filename
    from app.utils.preprocessing import sanitize_filename
    safe_filename = sanitize_filename(file.filename)

    # Read file
    try:
        file_bytes = await file.read()
    except Exception as e:
        logger.exception(f"Failed to read file: {e}")
        raise HTTPException(status_code=400, detail="Failed to read file")

    # Validate file size
    if len(file_bytes) > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE / 1024 / 1024:.0f}MB"
        )

    # Validate file type
    from app.utils.preprocessing import validate_image_file
    is_valid, error_msg = validate_image_file(file_bytes, safe_filename)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)

    # Run detection
    try:
        result = detector.predict(file_bytes, request_id=request_id)

        # Store heatmap in cache if available
        if result.heatmap_bytes:
            _heatmap_cache.set(result.request_id, result.heatmap_bytes)

        # Build response
        detection_result = DetectionResult(
            prediction=result.prediction,
            confidence=result.confidence,
            probabilities=Probabilities(**result.probabilities),
            model_used=result.model_used,
            ensemble_active=result.ensemble_active,
            models_used=result.models_used,
            processing_time_ms=result.processing_time_ms,
            face_detected=result.face_detected,
            heatmap_url=f"/api/v1/heatmap/{result.request_id}" if result.heatmap_bytes else None
        )

        return DetectionResponse(
            request_id=uuid.UUID(result.request_id) if result.request_id else uuid.uuid4(),
            result=detection_result,
            filename=safe_filename
        )

    except DetectionError as e:
        logger.exception("Detection failed: {e}", e=e)
        raise HTTPException(status_code=500, detail="Detection failed: unable to process image")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Detection failed: {e}", e=e)
        raise HTTPException(status_code=500, detail="Detection failed: internal server error")


# Batch detection endpoint
MAX_BATCH_SIZE = 10

@router.post(
    "/detect/batch",
    response_model=BatchDetectionResponse,
    summary="Batch detect deepfakes",
    description="Upload multiple images (max 10) to detect deepfakes in batch",
    responses={
        200: {"description": "Successful batch detection"},
        400: {"model": ErrorResponse, "description": "Invalid files or too many files"},
        401: {"model": ErrorResponse, "description": "Invalid or missing API key"},
        413: {"model": ErrorResponse, "description": "File too large"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    }
)
@limiter.limit(settings.RATE_LIMIT_DETECT)
async def detect_batch(
    request: Request,
    files: list[UploadFile] = File(..., description="Image files (JPG, PNG, WebP), max 10"),
    detector: EnsembleDetector = Depends(get_detector),
    api_key: Optional[str] = Depends(verify_api_key)
) -> BatchDetectionResponse:
    """
    Detect deepfakes in multiple images (batch processing).

    - **files**: List of image files (max 10, each max 10MB, JPG/PNG/WebP)
    - **X-API-Key**: Optional API key header (required if API_KEYS configured)
    - Returns list of predictions with confidence and probabilities
    - Images are processed sequentially
    """
    import time
    batch_start_time = time.perf_counter()
    batch_request_id = str(uuid.uuid4())

    # Validate batch size
    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Maximum batch size: {MAX_BATCH_SIZE}"
        )

    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No files provided")

    results = []

    # Process each file sequentially
    for idx, file in enumerate(files):
        file_request_id = f"{batch_request_id}-{idx}"

        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail=f"File {idx}: No filename provided")

        # Sanitize filename
        from app.utils.preprocessing import sanitize_filename
        safe_filename = sanitize_filename(file.filename)

        # Read file
        try:
            file_bytes = await file.read()
        except Exception as e:
            logger.exception("Failed to read file {idx}: {e}", idx=idx, e=e)
            raise HTTPException(status_code=400, detail=f"File {idx}: Failed to read file")

        # Validate file size
        if len(file_bytes) > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File {idx} ({safe_filename}): File too large. Maximum size: {settings.MAX_FILE_SIZE / 1024 / 1024:.0f}MB"
            )

        # Validate file type
        from app.utils.preprocessing import validate_image_file
        is_valid, error_msg = validate_image_file(file_bytes, safe_filename)
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"File {idx} ({safe_filename}): {error_msg}")

        # Run detection
        try:
            result = detector.predict(file_bytes, request_id=file_request_id)

            # Store heatmap in cache if available
            if result.heatmap_bytes:
                _heatmap_cache.set(result.request_id, result.heatmap_bytes)

            # Build response
            detection_result = DetectionResult(
                prediction=result.prediction,
                confidence=result.confidence,
                probabilities=Probabilities(**result.probabilities),
                model_used=result.model_used,
                ensemble_active=result.ensemble_active,
                models_used=result.models_used,
                processing_time_ms=result.processing_time_ms,
                face_detected=result.face_detected,
                heatmap_url=f"/api/v1/heatmap/{result.request_id}" if result.heatmap_bytes else None
            )
            results.append(detection_result)

        except DetectionError as e:
            logger.exception("Detection failed for file {idx}: {e}", idx=idx, e=e)
            # Return error result for this file
            results.append(DetectionResult(
                prediction="unknown",
                confidence=0.0,
                probabilities=Probabilities(real=0.5, fake=0.5),
                model_used="ensemble",
                ensemble_active=False,
                models_used=[],
                processing_time_ms=0.0,
                face_detected=False,
                heatmap_url=None
            ))
        except Exception as e:
            logger.exception("Detection failed for file {idx}: {e}", idx=idx, e=e)
            results.append(DetectionResult(
                prediction="unknown",
                confidence=0.0,
                probabilities=Probabilities(real=0.5, fake=0.5),
                model_used="ensemble",
                ensemble_active=False,
                models_used=[],
                processing_time_ms=0.0,
                face_detected=False,
                heatmap_url=None
            ))

    total_time = (time.perf_counter() - batch_start_time) * 1000

    logger.info(
        "Batch detection completed",
        batch_request_id=batch_request_id,
        file_count=len(files),
        total_processing_time_ms=round(total_time, 2)
    )

    return BatchDetectionResponse(
        request_id=uuid.UUID(batch_request_id),
        results=results,
        total_processing_time_ms=total_time
    )


@router.get(
    "/heatmap/{request_id}",
    summary="Get Grad-CAM heatmap",
    description="Retrieve the Grad-CAM heatmap for a previous detection request",
    responses={
        200: {"content": {"image/png": {}}, "description": "Heatmap image"},
        404: {"model": ErrorResponse, "description": "Heatmap not found"},
    }
)
@limiter.limit(settings.RATE_LIMIT_DEFAULT)
async def get_heatmap(request: Request, request_id: str):
    """
    Get heatmap for a previous request.

    Returns the cached Grad-CAM heatmap as a PNG image.
    Heatmaps are cached in memory for recent requests.
    """
    heatmap_bytes = _heatmap_cache.get(request_id)

    if heatmap_bytes is None:
        raise HTTPException(
            status_code=404,
            detail="Heatmap not found or expired. Heatmaps are cached for recent requests only."
        )

    return Response(content=heatmap_bytes, media_type="image/png")


async def _cleanup_temp_files_task(run_once: bool = False) -> None:
    """Clean up old temp files once or periodically in the background."""
    while True:
        try:
            if not run_once:
                await asyncio.sleep(1800)  # 30 minutes

            # Cleanup temp directory
            if settings.TEMP_DIR.exists():
                current_time = time.time()
                for file_path in settings.TEMP_DIR.iterdir():
                    try:
                        if file_path.is_file():
                            mtime = file_path.stat().st_mtime
                            if current_time - mtime > 3600:  # 1 hour
                                file_path.unlink()
                                logger.debug(f"Cleaned up old temp file: {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to clean up temp file {file_path}: {e}")

            # Also cleanup expired cache entries
            removed = _heatmap_cache.cleanup_expired()
            if removed:
                logger.debug(f"Cleaned up {removed} expired heatmap cache entries")

            if run_once:
                return

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in temp file cleanup task: {e}")


# Lifespan snippet for main.py - add this to your lifespan function:
"""
# In main.py lifespan function, add:

# Startup: run initial cleanup
await _cleanup_temp_files_task()  # Run once at startup

# Create background cleanup task
cleanup_task = asyncio.create_task(_cleanup_temp_files_task())

yield

# Shutdown: cancel cleanup task
cleanup_task.cancel()
try:
    await cleanup_task
except asyncio.CancelledError:
    pass
"""