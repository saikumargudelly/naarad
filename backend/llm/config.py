from pydantic_settings import BaseSettings
from typing import Optional, List

class Settings(BaseSettings):
    # OpenRouter Configuration
    OPENROUTER_API_KEY: str
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    
    # Together.ai Configuration
    TOGETHER_API_KEY: str
    TOGETHER_BASE_URL: str = "https://api.together.xyz/v1"
    
    # Model configurations
    REASONING_MODEL: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    CHAT_MODEL: str = "nousresearch/nous-hermes-2-mixtral-8x7b-dpo"
    VISION_MODEL: str = "llava-hf/llava-1.6-vicuna-7b-hf"
    
    # Brave Search API
    BRAVE_API_KEY: str
    
    # Supabase Configuration (Optional)
    SUPABASE_URL: Optional[str] = None
    SUPABASE_KEY: Optional[str] = None
    
    # Server Configuration
    APP_ENV: str = "development"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    ALLOWED_ORIGINS: str = "http://localhost:3000"
    
    # Groq API Configuration
    GROQ_API_KEY: str
    
    @property
    def allowed_origins_list(self) -> List[str]:
        """Convert comma-separated ALLOWED_ORIGINS string to list"""
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",")]
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()