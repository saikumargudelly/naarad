"""Logging configuration settings."""

from enum import Enum
from pydantic import Field
from pydantic_settings import BaseSettings

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
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = True
        extra = 'ignore'
