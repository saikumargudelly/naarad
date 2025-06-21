"""LLM configuration settings."""

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings

class LLMSettings(BaseSettings):
    """LLM-specific settings."""
    
    # API Keys
    OPENROUTER_API_KEY: Optional[str] = Field(
        default=None,
        description="API key for OpenRouter service"
    )
    TOGETHER_API_KEY: Optional[str] = Field(
        default=None,
        description="API key for Together AI service"
    )
    GROQ_API_KEY: Optional[str] = Field(
        default=None,
        description="API key for Groq service"
    )
    
    # Model Configuration
    DEFAULT_MODEL: str = Field(
        default="mistralai/Mixtral-8x7B-Instruct-v0.1",
        description="Default model to use for general tasks"
    )
    CHAT_MODEL: str = Field(
        default="nousresearch/nous-hermes-2-mixtral-8x7b-dpo",
        description="Model to use for chat interactions"
    )
    EMBEDDING_MODEL: str = Field(
        default="sentence-transformers/all-mpnet-base-v2",
        description="Model to use for embeddings"
    )
    
    # Model parameters
    MODEL_TEMPERATURE: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for model generation"
    )
    MODEL_MAX_TOKENS: int = Field(
        default=2048,
        ge=1,
        le=8192,
        description="Maximum number of tokens to generate"
    )
    MODEL_TOP_P: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Top-p sampling parameter"
    )
    MODEL_FREQUENCY_PENALTY: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Frequency penalty for token repetition"
    )
    MODEL_PRESENCE_PENALTY: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Presence penalty for token repetition"
    )
    
    # Request settings
    REQUEST_TIMEOUT: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Request timeout in seconds"
    )
    MAX_RETRIES: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum number of retries for failed requests"
    )
    RETRY_DELAY: float = Field(
        default=1.0,
        ge=0.1,
        le=60.0,
        description="Delay between retries in seconds"
    )
    
    # Rate limiting
    RATE_LIMIT_REQUESTS: int = Field(
        default=100,
        ge=1,
        description="Number of requests allowed per time window"
    )
    RATE_LIMIT_WINDOW: int = Field(
        default=60,
        ge=1,
        description="Time window for rate limiting in seconds"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = True
        extra = 'ignore' 