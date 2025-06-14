"""Mock implementation of LLM config for testing."""

class Settings:
    """Mock settings class for LLM configuration."""
    
    def __init__(self):
        # LLM Settings
        self.LLM_API_KEY = "test_api_key"
        self.LLM_MODEL_NAME = "gpt-3.5-turbo"
        self.LLM_TEMPERATURE = 0.7
        self.LLM_MAX_TOKENS = 1000
        self.LLM_REQUEST_TIMEOUT = 30
        
        # OpenAI Settings
        self.OPENAI_API_KEY = "test_openai_key"
        self.OPENAI_ORGANIZATION = "test_org"
        
        # Other settings
        self.DEBUG = True
        self.TESTING = True
        self.ENVIRONMENT = "test"

# Create a singleton instance
settings = Settings()

# Mock the module
import sys
sys.modules['agent.llm.config'] = sys.modules[__name__]
