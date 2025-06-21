"""Security configuration settings."""

from typing import List, Union
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from pydantic.networks import HttpUrl

class SecuritySettings(BaseSettings):
    """Security-specific settings."""
    
    # Secret key for cryptographic operations
    SECRET_KEY: str = Field(
        default="your-secret-key-here",
        min_length=32,
        description="Secret key for cryptographic operations"
    )
    
    # CORS Settings
    BACKEND_CORS_ORIGINS: List[Union[HttpUrl, str]] = Field(
        default=[
            "http://localhost:3000",
            "http://localhost:8000",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8000",
        ],
        description="Allowed CORS origins"
    )
    
    @field_validator("BACKEND_CORS_ORIGINS")
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        """Parse CORS origins from string or list."""
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    # Authentication settings
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=60 * 24 * 7,  # 7 days
        ge=1,
        le=60 * 24 * 365,  # 1 year
        description="Access token expiration time in minutes"
    )
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Refresh token expiration time in days"
    )
    ALGORITHM: str = Field(
        default="HS256",
        description="JWT signing algorithm"
    )
    
    # Password settings
    PASSWORD_MIN_LENGTH: int = Field(
        default=8,
        ge=6,
        le=128,
        description="Minimum password length"
    )
    PASSWORD_REQUIRE_UPPERCASE: bool = Field(
        default=True,
        description="Require uppercase letters in passwords"
    )
    PASSWORD_REQUIRE_LOWERCASE: bool = Field(
        default=True,
        description="Require lowercase letters in passwords"
    )
    PASSWORD_REQUIRE_DIGITS: bool = Field(
        default=True,
        description="Require digits in passwords"
    )
    PASSWORD_REQUIRE_SPECIAL_CHARS: bool = Field(
        default=True,
        description="Require special characters in passwords"
    )
    
    # Rate limiting
    RATE_LIMIT: str = Field(
        default="100/minute",
        description="Rate limit string (e.g., '100/minute')"
    )
    RATE_LIMIT_WINDOW: int = Field(
        default=60,
        ge=1,
        description="Rate limiting window in seconds"
    )
    
    # Session settings
    SESSION_COOKIE_SECURE: bool = Field(
        default=False,
        description="Use secure cookies for sessions"
    )
    SESSION_COOKIE_HTTPONLY: bool = Field(
        default=True,
        description="Use HTTP-only cookies for sessions"
    )
    SESSION_COOKIE_SAMESITE: str = Field(
        default="lax",
        description="SameSite attribute for session cookies"
    )
    
    # API security
    API_KEY_HEADER: str = Field(
        default="X-API-Key",
        description="Header name for API key authentication"
    )
    ENABLE_API_KEY_AUTH: bool = Field(
        default=False,
        description="Enable API key authentication"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = True
        extra = 'ignore' 