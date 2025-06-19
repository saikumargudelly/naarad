from enum import Enum
from typing import List, Optional, Dict, Any, Union, Literal
from pathlib import Path
from pydantic import Field, field_validator, model_validator, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic.networks import AnyHttpUrl, HttpUrl

class Environment(str, Enum):
    """Application environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class Settings(BaseSettings):
    """Application settings."""
    
    # Model configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding='utf-8',
        case_sensitive=True,
        extra='ignore',
        env_nested_delimiter='__',
        validate_default=True,
        # Removed NAARAD_ prefix for simpler environment variable names
    )
    
    # --- Application Settings ---
    APP_NAME: str = "Naarad AI Assistant"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: Environment = Environment.DEVELOPMENT
    DEBUG: bool = True
    SECRET_KEY: str = Field(
        default="your-secret-key-here",
        min_length=32,
        description="Secret key for cryptographic operations"
    )
    
    # --- Server Settings ---
    HOST: str = "0.0.0.0"
    PORT: int = Field(default=8000, ge=1024, le=65535)
    RELOAD: bool = True
    WORKERS: int = 1
    
    # CORS Settings
    BACKEND_CORS_ORIGINS: List[Union[HttpUrl, str]] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
    ]
    
    @field_validator("BACKEND_CORS_ORIGINS")
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        """Parse CORS origins from string or list."""
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    # API Settings
    API_PREFIX: str = "/api"
    API_V1_STR: str = "/api/v1"
    
    # --- Logging ---
    LOG_LEVEL: LogLevel = LogLevel.INFO
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # --- Rate Limiting ---
    RATE_LIMIT: str = "100/minute"
    RATE_LIMIT_WINDOW: int = 60  # seconds
    
    # --- Security ---
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    ALGORITHM: str = "HS256"
    
    # --- API Keys ---
    OPENROUTER_API_KEY: Optional[str] = Field(
        default=None,
        description="API key for OpenRouter service"
    )
    TOGETHER_API_KEY: Optional[str] = Field(
        default=None,
        description="API key for Together AI service"
    )
    BRAVE_API_KEY: Optional[str] = Field(
        default=None,
        description="API key for Brave Search"
    )
    GROQ_API_KEY: Optional[str] = Field(
        default=None,
        description="API key for Groq service"
    )
    
    # --- Database ---
    DATABASE_URL: Optional[str] = Field(
        default=None,
        description="Database connection string"
    )
    
    # --- Supabase Configuration ---
    SUPABASE_URL: Optional[HttpUrl] = None
    SUPABASE_KEY: Optional[str] = None
    
    # --- Model Configuration ---
    DEFAULT_MODEL: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    CHAT_MODEL: str = "nousresearch/nous-hermes-2-mixtral-8x7b-dpo"
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    
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
            self.LOG_LEVEL = LogLevel.INFO
        return self

settings = Settings()
