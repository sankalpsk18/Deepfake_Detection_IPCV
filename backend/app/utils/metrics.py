"""Prometheus metrics for monitoring."""

import time
from functools import wraps
from typing import Callable

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

# Request metrics
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"]
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
)

# Detection metrics
DETECTION_COUNT = Counter(
    "detections_total",
    "Total detections performed",
    ["prediction", "model_used", "ensemble_active"]
)

DETECTION_CONFIDENCE = Histogram(
    "detection_confidence",
    "Detection confidence scores",
    ["prediction"],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0)
)

DETECTION_LATENCY = Histogram(
    "detection_duration_seconds",
    "Detection inference latency in seconds",
    ["model_used"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
)

# Model metrics
MODEL_LOADED = Gauge(
    "model_loaded",
    "Whether a model is loaded (1) or not (0)",
    ["model_name"]
)

# Error metrics
ERROR_COUNT = Counter(
    "errors_total",
    "Total errors by type",
    ["error_type", "endpoint"]
)

# Active requests gauge
ACTIVE_REQUESTS = Gauge(
    "http_requests_active",
    "Currently active HTTP requests",
    ["method", "endpoint"]
)

# Rate limit metrics
RATE_LIMIT_EXCEEDED = Counter(
    "rate_limit_exceeded_total",
    "Total rate limit exceeded events",
    ["endpoint"]
)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to collect HTTP request metrics."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        method = request.method
        # Normalize path for metrics (remove IDs)
        path = request.url.path
        endpoint = self._normalize_path(path)

        # Increment active requests
        ACTIVE_REQUESTS.labels(method=method, endpoint=endpoint).inc()

        start_time = time.time()
        status_code = 500  # Default to error

        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception as e:
            # Count exceptions
            ERROR_COUNT.labels(error_type=type(e).__name__, endpoint=endpoint).inc()
            raise
        finally:
            # Record metrics
            duration = time.time() - start_time
            REQUEST_COUNT.labels(
                method=method,
                endpoint=endpoint,
                status_code=status_code
            ).inc()
            REQUEST_LATENCY.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
            ACTIVE_REQUESTS.labels(method=method, endpoint=endpoint).dec()

    def _normalize_path(self, path: str) -> str:
        """Normalize path for metrics (replace IDs with placeholders)."""
        # Replace UUIDs
        import re
        path = re.sub(
            r'/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
            '/:uuid',
            path
        )
        # Replace numeric IDs
        path = re.sub(r'/\d+', '/:id', path)
        # Replace heatmap IDs
        path = re.sub(r'/heatmap/[^/]+', '/heatmap/:id', path)
        return path


def record_detection(
    prediction: str,
    confidence: float,
    model_used: str,
    ensemble_active: bool,
    latency_seconds: float
) -> None:
    """Record detection metrics."""
    DETECTION_COUNT.labels(
        prediction=prediction,
        model_used=model_used,
        ensemble_active=str(ensemble_active).lower()
    ).inc()
    DETECTION_CONFIDENCE.labels(prediction=prediction).observe(confidence)
    DETECTION_LATENCY.labels(model_used=model_used).observe(latency_seconds)


def record_error(error_type: str, endpoint: str) -> None:
    """Record error metrics."""
    ERROR_COUNT.labels(error_type=error_type, endpoint=endpoint).inc()


def record_rate_limit_exceeded(endpoint: str) -> None:
    """Record rate limit exceeded."""
    RATE_LIMIT_EXCEEDED.labels(endpoint=endpoint).inc()


def set_model_loaded(model_name: str, loaded: bool) -> None:
    """Set model loaded status."""
    MODEL_LOADED.labels(model_name=model_name).set(1 if loaded else 0)


async def metrics_endpoint(request: Request) -> Response:
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )