"""Mock settings for testing."""

class Settings:
    # Agent settings
    AGENT_TEMPERATURE = 0.7
    AGENT_MODEL_NAME = "gpt-3.5-turbo"
    AGENT_MAX_TOKENS = 1000
    AGENT_REQUEST_TIMEOUT = 30
    
    # LLM settings
    LLM_API_KEY = "test_api_key"
    LLM_API_BASE = "https://api.openai.com/v1"
    
    # App settings
    DEBUG = True
    TESTING = True
    ENVIRONMENT = "test"
    
    # Feature flags
    ENABLE_FEATURE_X = True
    ENABLE_FEATURE_Y = False
    
    # Strict mode for testing
    strict_mode = False

# Create a singleton instance
settings = Settings()
