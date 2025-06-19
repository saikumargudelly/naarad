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
    # Model names and configurations for OpenRouter
    REASONING_MODEL: str = Field(
        default="openai/gpt-4",
        description="Default model for reasoning tasks"
    )
    CHAT_MODEL: str = Field(
        default="openai/gpt-3.5-turbo-0125",
        description="Model optimized for chat interactions"
    )
    VISION_MODEL: str = Field(
        default="openai/gpt-4-vision-preview",
        description="Model for vision-language tasks"
    )
    EMBEDDING_MODEL: str = Field(
        default="text-embedding-ada-002",
        description="Model for text embeddings"
    )
    
    # --- API Configuration ---
    # OpenRouter
    OPENROUTER_API_KEY: Optional[str] = Field(
        default=None,
        description="API key for OpenRouter service"
    )
    OPENROUTER_BASE_URL: str = Field(
        default="https://openrouter.ai/api/v1",
        description="Base URL for OpenRouter API"
    )
    
    # Together.ai
    TOGETHER_API_KEY: Optional[str] = Field(
        default=None,
        description="API key for Together AI service"
    )
    TOGETHER_BASE_URL: str = Field(
        default="https://api.together.xyz/v1",
        description="Base URL for Together AI API"
    )
    
    # Groq
    GROQ_API_KEY: Optional[str] = Field(
        default=None,
        description="API key for Groq service"
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
    
    # --- Model Configuration ---
    MAX_TOKENS: int = Field(
        default=4096,
        ge=1,
        le=32768,
        description="Maximum number of tokens to generate"
    )
    TEMPERATURE: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for generation"
    )
    TOP_P: float = Field(
        default=0.9,
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
        validate_default=True,
        env_prefix='NAARAD_LLM_',
    )

settings = Settings()