from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # OpenRouter Configuration
    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    
    # Together.ai Configuration
    together_api_key: str
    together_base_url: str = "https://api.together.xyz/v1"
    
    # Model configurations
    reasoning_model: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    chat_model: str = "nousresearch/nous-hermes-2-mixtral-8x7b-dpo"
    vision_model: str = "llava-hf/llava-1.6-vicuna-7b-hf"
    
    # Brave Search API
    brave_api_key: str
    
    # Supabase (optional)
    supabase_url: Optional[str] = None
    supabase_key: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
