"""Database configuration settings."""

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from pydantic import ConfigDict

class DatabaseSettings(BaseSettings):
    """Database-specific settings."""
    
    # Database URL
    DATABASE_URL: Optional[str] = Field(
        default=None,
        description="Database connection string"
    )
    
    # Supabase Configuration
    SUPABASE_URL: Optional[str] = Field(
        default=None,
        description="Supabase project URL"
    )
    SUPABASE_KEY: Optional[str] = Field(
        default=None,
        description="Supabase API key"
    )
    
    # Database connection settings
    DB_POOL_SIZE: int = Field(
        default=10,
        description="Database connection pool size"
    )
    DB_MAX_OVERFLOW: int = Field(
        default=20,
        description="Maximum database connection overflow"
    )
    DB_POOL_TIMEOUT: int = Field(
        default=30,
        description="Database connection pool timeout in seconds"
    )
    DB_POOL_RECYCLE: int = Field(
        default=3600,
        description="Database connection pool recycle time in seconds"
    )
    
    # Migration settings
    DB_AUTO_MIGRATE: bool = Field(
        default=False,
        description="Automatically run database migrations on startup"
    )
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding='utf-8',
        case_sensitive=True,
        extra='ignore',
    ) 