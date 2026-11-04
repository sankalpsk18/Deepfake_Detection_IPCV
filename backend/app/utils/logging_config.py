"""Structured JSON logging configuration with correlation ID support."""

import json
import logging
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from app.config import settings

# Context variable to store correlation ID for the current request
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")


def get_correlation_id() -> str:
    """Get the current correlation ID, generating one if not set."""
    cid = correlation_id_var.get()
    if not cid:
        cid = str(uuid.uuid4())[:8]
        correlation_id_var.set(cid)
    return cid


def set_correlation_id(cid: str) -> None:
    """Set the correlation ID for the current context."""
    correlation_id_var.set(cid)


class JSONFormatter(logging.Formatter):
    """JSON log formatter with correlation ID and structured fields."""

    def format(self, record: logging.LogRecord) -> str:
        # Get correlation ID from context
        correlation_id = get_correlation_id()

        # Base log structure
        log_data: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": correlation_id,
        }

        # Add extra fields from record
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add source location
        log_data["source"] = {
            "file": record.pathname,
            "line": record.lineno,
            "function": record.funcName,
        }

        return json.dumps(log_data, ensure_ascii=False)


class StructuredLogger:
    """Wrapper around standard logger with structured logging helpers."""

    def __init__(self, name: str):
        self._logger = logging.getLogger(name)

    def _log(self, level: int, message: str, **kwargs) -> None:
        """Log with extra structured fields."""
        extra = {"extra_fields": kwargs} if kwargs else {}
        self._logger.log(level, message, extra=extra)

    def debug(self, message: str, **kwargs) -> None:
        self._log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        self._log(logging.ERROR, message, **kwargs)

    def exception(self, message: str, **kwargs) -> None:
        self._log(logging.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs) -> None:
        self._log(logging.CRITICAL, message, **kwargs)


def setup_logging() -> None:
    """Configure application logging with JSON formatter."""
    # Determine log level
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    # Create JSON formatter
    json_formatter = JSONFormatter()

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler with JSON formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(json_formatter)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)

    # Reduce noise from third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(log_level)
    logging.getLogger("slowapi").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Log startup message using standard logger (no extra fields)
    std_logger = logging.getLogger(__name__)
    std_logger.info("Logging configured - format: json, level: %s", settings.LOG_LEVEL)


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger instance."""
    return StructuredLogger(name)