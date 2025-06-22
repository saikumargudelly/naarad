"""Main application configuration.

This module combines all domain-specific configurations into a single settings object.
"""

from enum import Enum
from typing import List, Optional, Dict, Any, Union, Literal
from pathlib import Path
from pydantic import Field, field_validator, model_validator, ConfigDict
from pydantic_settings import BaseSettings
from pydantic.networks import AnyHttpUrl, HttpUrl

# Import domain-specific configurations
from .database_config import DatabaseSettings
from .llm_config import LLMSettings
from .security_config import SecuritySettings
from .logging_config import LoggingSettings, LogLevel

class Environment(str, Enum):
    """Application environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

class Settings(BaseSettings):
    """Main application settings combining all domain configurations."""
    
    # Model configuration
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding='utf-8',
        case_sensitive=True,
        extra='ignore',
        env_nested_delimiter='__',
        validate_default=True,
    )
    
    # --- Application Settings ---
    APP_NAME: str = "Naarad AI Assistant"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: Environment = Environment.DEVELOPMENT
    DEBUG: bool = True
    
    # --- Server Settings ---
    HOST: str = "0.0.0.0"
    PORT: int = Field(default=8000, ge=1024, le=65535)
    RELOAD: bool = True
    WORKERS: int = 1
    
    # API Settings
    API_PREFIX: str = "/api"
    API_V1_STR: str = "/api/v1"
    
    # --- File Storage ---
    UPLOAD_FOLDER: Path = Path("uploads")
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: List[str] = ["jpg", "jpeg", "png", "gif", "pdf"]
    
    # --- Caching ---
    ENABLE_CACHE: bool = True
    CACHE_TTL: int = 300  # 5 minutes
    
    # --- Monitoring ---
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9100
    
    # --- External API Keys ---
    BRAVE_API_KEY: Optional[str] = Field(
        default=None,
        description="API key for Brave Search"
    )
    
    # --- Domain-specific settings ---
    database: DatabaseSettings = DatabaseSettings()
    llm: LLMSettings = LLMSettings()
    security: SecuritySettings = SecuritySettings()
    logging: LoggingSettings = LoggingSettings()
    
    # --- Validation ---
    @field_validator('ENVIRONMENT')
    def validate_environment(cls, v: str) -> str:
        """Validate environment value."""
        if v not in [e.value for e in Environment]:
            raise ValueError(f"Invalid environment: {v}")
        return v
        
    @model_validator(mode='after')
    def validate_settings(self) -> 'Settings':
        """Validate settings after model initialization."""
        if self.ENVIRONMENT == Environment.PRODUCTION and self.DEBUG:
            self.DEBUG = False
            self.logging.LOG_LEVEL = LogLevel.INFO
        return self
    
    # --- Convenience properties ---
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.ENVIRONMENT == Environment.DEVELOPMENT
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.ENVIRONMENT == Environment.PRODUCTION
    
    @property
    def is_testing(self) -> bool:
        """Check if running in testing mode."""
        return self.ENVIRONMENT == Environment.TESTING
    
    @property
    def cors_origins(self) -> List[str]:
        """Get CORS origins from security settings."""
        return self.security.BACKEND_CORS_ORIGINS
    
    @property
    def rate_limit(self) -> str:
        """Get rate limit from security settings."""
        return self.security.RATE_LIMIT
    
    @property
    def log_level(self) -> str:
        """Get log level from logging settings."""
        return self.logging.LOG_LEVEL
    
    @property
    def log_format(self) -> str:
        """Get log format from logging settings."""
        return self.logging.LOG_FORMAT

# Create global settings instance
settings = Settings()
