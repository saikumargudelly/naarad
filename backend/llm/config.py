from enum import Enum
from typing import Optional, List, ClassVar, Union, Dict, Any
from pathlib import Path
from pydantic import Field, field_validator, model_validator, HttpUrl, AnyHttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict

class Environment(str, Enum):
    """Application environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

class Settings(BaseSettings):
    """LLM and API configuration settings."""
    
    # --- Model Configuration ---
    # Default provider
    LLM_PROVIDER: str = Field(
        default="groq",
        description="Default LLM provider (groq, openai, etc.)"
    )
    
    # Model names and configurations - Updated to use currently supported models
    REASONING_MODEL: str = Field(
        default="llama3-8b-8192",
        description="Default model for reasoning tasks"
    )
    CHAT_MODEL: str = Field(
        default="llama3-8b-8192",
        description="Model optimized for chat interactions"
    )
    VISION_MODEL: str = Field(
        default="llava-1.5-7b",
        description="Model for vision-language tasks"
    )
    EMBEDDING_MODEL: str = Field(
        default="text-embedding-3-small",
        description="Model for text embeddings"
    )
    
    # --- API Configuration ---
    # Groq Configuration
    GROQ_API_KEY: Optional[str] = Field(
        default=None,
        description="API key for Groq service"
    )
    GROQ_BASE_URL: str = Field(
        default="https://api.groq.com/openai/v1",
        description="Base URL for Groq API"
    )
    
    # Brave Search
    BRAVE_API_KEY: Optional[str] = Field(
        default=None,
        description="API key for Brave Search"
    )
    
    # --- Server Configuration ---
    ENVIRONMENT: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Application environment"
    )
    HOST: str = Field(
        default="0.0.0.0",
        description="Host to bind the server to"
    )
    PORT: int = Field(
        default=8000,
        ge=1024,
        le=65535,
        description="Port to run the server on"
    )
    ALLOWED_ORIGINS: List[Union[HttpUrl, str]] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"],
        description="List of allowed CORS origins"
    )
    
    # --- Supabase Configuration ---
    SUPABASE_URL: Optional[HttpUrl] = Field(
        default=None,
        description="Supabase project URL"
    )
    SUPABASE_KEY: Optional[str] = Field(
        default=None,
        description="Supabase anon/public key"
    )
    
    # --- Generation Parameters ---
    MAX_TOKENS: int = Field(
        default=8192,  # Increased for Groq's larger context windows
        ge=1,
        le=32768,
        description="Maximum number of tokens to generate"
    )
    TEMPERATURE: float = Field(
        default=0.2,  # Lower temperature for more focused responses
        ge=0.0,
        le=1.0,
        description="Sampling temperature for generation"
    )
    TOP_P: float = Field(
        default=0.95,  # Slightly higher top_p for better diversity
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter"
    )
    
    # --- Caching ---
    ENABLE_CACHE: bool = Field(
        default=True,
        description="Enable response caching"
    )
    CACHE_TTL: int = Field(
        default=300,
        ge=0,
        description="Cache time-to-live in seconds"
    )
    
    # --- Validation ---
    @field_validator('ALLOWED_ORIGINS', mode='before')
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        """Parse CORS origins from string or list."""
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    @property
    def allowed_origins_list(self) -> List[str]:
        """Get allowed origins as a list of strings."""
        return [str(origin) for origin in self.ALLOWED_ORIGINS]
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding='utf-8',
        case_sensitive=True,
        extra='ignore',
        env_nested_delimiter='__',
        validate_default=True
    )

settings = Settings()