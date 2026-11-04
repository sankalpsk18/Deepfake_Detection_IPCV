"""Main FastAPI application entry point."""

import asyncio
import logging
import time
import uuid
import shutil
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.config import settings
from app.api.routes import router as api_router, _cleanup_temp_files_task
from app.models.detector import create_ensemble_detector
from app.schemas.detection import ErrorResponse
from app.utils.logging_config import setup_logging, get_logger, get_correlation_id, set_correlation_id
from app.utils.metrics import MetricsMiddleware, metrics_endpoint, set_model_loaded

# Configure structured logging
setup_logging()
logger = get_logger(__name__)

# Rate limiter - use configurable limits
limiter = Limiter(key_func=get_remote_address, default_limits=[settings.RATE_LIMIT_DEFAULT])


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Deepfake Detector API...")
    start_time = time.time()

    # Initialize detector
    detector = create_ensemble_detector()
    detector.warmup()
    app.state.detector = detector

    # Record model loaded status
    model_info = detector.get_model_info()
    set_model_loaded("efficientnet", model_info["efficientnet"]["loaded"])
    set_model_loaded("mesonet", model_info["mesonet"]["loaded"])

    # Run initial temp file cleanup
    try:
        await _cleanup_temp_files_task(run_once=True)
    except Exception as e:
        logger.warning("Initial temp file cleanup failed", error=str(e))

    # Create background cleanup task
    cleanup_task = asyncio.create_task(_cleanup_temp_files_task())

    logger.info("API started", duration_seconds=round(time.time() - start_time, 2))

    yield

    # Shutdown
    logger.info("Shutting down Deepfake Detector API...")
    # Cancel background cleanup task
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

    # Clean up temp directory
    try:
        if settings.TEMP_DIR.exists():
            shutil.rmtree(settings.TEMP_DIR)
            logger.info("Cleaned up temp directory")
    except Exception as e:
        logger.warning("Failed to clean up temp directory", error=str(e))


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="AI-Generated / Deepfake Image Detection API",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )

    # Rate limiter
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-API-Key"],
    )

    # Metrics middleware (must be added before other middleware to capture all requests)
    app.add_middleware(MetricsMiddleware)

    # Request timing middleware
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response

    # Correlation ID middleware
    @app.middleware("http")
    async def add_correlation_id(request: Request, call_next):
        # Get or generate correlation ID
        correlation_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())[:8]
        set_correlation_id(correlation_id)

        # Add to response headers
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = correlation_id
        return response

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Custom HTTP exception handler."""
        correlation_id = get_correlation_id()
        logger.warning(
            "HTTP exception",
            status_code=exc.status_code,
            detail=exc.detail,
            path=request.url.path,
            method=request.method
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error="HTTPException",
                message=exc.detail,
                request_id=uuid.uuid4()
            ).model_dump(mode='json')
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """General exception handler."""
        correlation_id = get_correlation_id()
        logger.exception(
            "Unhandled exception",
            error_type=type(exc).__name__,
            error_message=str(exc),
            path=request.url.path,
            method=request.method
        )
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="InternalServerError",
                message="An unexpected error occurred",
                request_id=uuid.uuid4()
            ).model_dump(mode='json')
        )

    # Prometheus metrics endpoint
    app.add_route("/metrics", metrics_endpoint, methods=["GET"])

    # Include API routes
    app.include_router(api_router)

    # Root endpoint
    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "status": "running",
            "docs": "/docs",
            "api": settings.API_PREFIX
        }

    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )