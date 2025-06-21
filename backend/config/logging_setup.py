"""Logging setup utility."""

import logging
import sys
from pathlib import Path
from typing import Optional
from logging.handlers import RotatingFileHandler
import json

from .logging_config import LoggingSettings

def setup_logging(log_file: Optional[str] = None) -> None:
    """Configure logging for the application."""
    settings = LoggingSettings()
    
    # Get log level
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    
    # Create formatter
    if settings.ENABLE_STRUCTURED_LOGGING:
        formatter = logging.Formatter('%(message)s')
    else:
        formatter = logging.Formatter(settings.LOG_FORMAT)
    
    # Create handlers
    handlers = []
    
    # Console handler
    if settings.ENABLE_CONSOLE_LOGGING:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, settings.CONSOLE_LOG_LEVEL.upper(), logging.INFO))
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)
    
    # File handler
    if settings.ENABLE_FILE_LOGGING:
        log_file_path = log_file or settings.LOG_FILE
        log_path = Path(log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file_path,
            maxBytes=settings.LOG_FILE_MAX_SIZE,
            backupCount=settings.LOG_FILE_BACKUP_COUNT
        )
        file_handler.setLevel(getattr(logging, settings.FILE_LOG_LEVEL.upper(), logging.DEBUG))
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        force=True
    )
    
    # Set log levels for specific loggers
    logging.getLogger("uvicorn").setLevel(log_level)
    logging.getLogger("uvicorn.access").handlers = logging.getLogger("uvicorn").handlers
    
    # Configure third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging."""
    
    def format(self, record):
        """Format log record as JSON."""
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ["name", "msg", "args", "levelname", "levelno", "pathname", 
                          "filename", "module", "lineno", "funcName", "created", 
                          "msecs", "relativeCreated", "thread", "threadName", 
                          "processName", "process", "getMessage", "exc_info", 
                          "exc_text", "stack_info"]:
                log_entry[key] = value
        
        return json.dumps(log_entry)

def mask_sensitive_data(data: dict, sensitive_fields: list) -> dict:
    """Mask sensitive data in log entries."""
    if not isinstance(data, dict):
        return data
    
    masked_data = data.copy()
    for key, value in masked_data.items():
        if isinstance(value, dict):
            masked_data[key] = mask_sensitive_data(value, sensitive_fields)
        elif isinstance(value, str) and any(field.lower() in key.lower() for field in sensitive_fields):
            masked_data[key] = "***MASKED***"
    
    return masked_data 
 