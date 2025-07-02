"""Logging configuration settings."""

from enum import Enum
from pydantic import Field
from pydantic_settings import BaseSettings
import logging
import logging.handlers
import os
from typing import Optional
from pydantic import ConfigDict

try:
    import json_log_formatter
    HAS_JSON_LOG = True
except ImportError:
    HAS_JSON_LOG = False

class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class LoggingSettings(BaseSettings):
    """Logging-specific settings."""
    
    # Log level
    LOG_LEVEL: LogLevel = Field(
        default=LogLevel.INFO,
        description="Logging level for the application"
    )
    
    # Log format
    LOG_FORMAT: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Format string for log messages"
    )
    
    # Log file settings
    LOG_FILE: str = Field(
        default="logs/naarad.log",
        description="Path to the main log file"
    )
    LOG_FILE_MAX_SIZE: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        ge=1024 * 1024,  # 1MB
        le=100 * 1024 * 1024,  # 100MB
        description="Maximum size of log file in bytes"
    )
    LOG_FILE_BACKUP_COUNT: int = Field(
        default=5,
        ge=0,
        le=20,
        description="Number of backup log files to keep"
    )
    
    # Console logging
    ENABLE_CONSOLE_LOGGING: bool = Field(
        default=True,
        description="Enable logging to console"
    )
    CONSOLE_LOG_LEVEL: LogLevel = Field(
        default=LogLevel.INFO,
        description="Logging level for console output"
    )
    
    # File logging
    ENABLE_FILE_LOGGING: bool = Field(
        default=True,
        description="Enable logging to file"
    )
    FILE_LOG_LEVEL: LogLevel = Field(
        default=LogLevel.DEBUG,
        description="Logging level for file output"
    )
    
    # Structured logging
    ENABLE_STRUCTURED_LOGGING: bool = Field(
        default=False,
        description="Enable structured logging (JSON format)"
    )
    
    # Performance logging
    ENABLE_PERFORMANCE_LOGGING: bool = Field(
        default=True,
        description="Enable performance logging"
    )
    PERFORMANCE_LOG_THRESHOLD: float = Field(
        default=1.0,
        ge=0.1,
        le=60.0,
        description="Threshold in seconds for performance logging"
    )
    
    # Error tracking
    ENABLE_ERROR_TRACKING: bool = Field(
        default=True,
        description="Enable detailed error tracking"
    )
    ERROR_LOG_STACK_TRACE: bool = Field(
        default=True,
        description="Include stack traces in error logs"
    )
    
    # Request logging
    ENABLE_REQUEST_LOGGING: bool = Field(
        default=True,
        description="Enable HTTP request logging"
    )
    LOG_REQUEST_BODY: bool = Field(
        default=False,
        description="Log request body in HTTP logs"
    )
    LOG_RESPONSE_BODY: bool = Field(
        default=False,
        description="Log response body in HTTP logs"
    )
    
    # Sensitive data masking
    MASK_SENSITIVE_DATA: bool = Field(
        default=True,
        description="Mask sensitive data in logs"
    )
    SENSITIVE_FIELDS: list = Field(
        default=["password", "token", "api_key", "secret"],
        description="List of sensitive field names to mask"
    )
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding='utf-8',
        case_sensitive=True,
        extra='ignore',
    )

def setup_logging(settings: Optional[LoggingSettings] = None):
    """
    Set up logging configuration for the application.
    Args:
        settings (LoggingSettings, optional): Logging settings. If None, defaults are used.
    """
    if settings is None:
        settings = LoggingSettings()

    # Set up root logger to DEBUG by default
    log_level = os.getenv("LOG_LEVEL", "DEBUG").upper()
    logging.basicConfig(level=log_level)
    logging.getLogger().setLevel(log_level)

    log_format = settings.LOG_FORMAT

    # Ensure log directory exists
    log_dir = os.path.dirname(settings.LOG_FILE)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    # Remove all handlers first (for reloads)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console Handler
    if settings.ENABLE_CONSOLE_LOGGING:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, str(settings.CONSOLE_LOG_LEVEL), logging.INFO))
        if settings.ENABLE_STRUCTURED_LOGGING and HAS_JSON_LOG:
            formatter = json_log_formatter.JSONFormatter()
        else:
            formatter = logging.Formatter(log_format)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File Handler
    if settings.ENABLE_FILE_LOGGING:
        file_handler = logging.handlers.RotatingFileHandler(
            settings.LOG_FILE,
            maxBytes=settings.LOG_FILE_MAX_SIZE,
            backupCount=settings.LOG_FILE_BACKUP_COUNT,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, str(settings.FILE_LOG_LEVEL), logging.DEBUG))
        if settings.ENABLE_STRUCTURED_LOGGING and HAS_JSON_LOG:
            formatter = json_log_formatter.JSONFormatter()
        else:
            formatter = logging.Formatter(log_format)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Optionally, add more handlers (e.g., error tracking, performance logs) here

    # Suppress overly verbose loggers if needed
    logging.getLogger('uvicorn.access').setLevel(logging.WARNING)
    logging.getLogger('uvicorn.error').setLevel(logging.INFO)
    logging.getLogger('asyncio').setLevel(logging.WARNING)

    root_logger.info("Logging is configured.")
