"""Test utilities for the agent system."""

class MockSettings:
    """Mock settings for testing."""
    DEBUG = True
    TESTING = True
    ENVIRONMENT = "test"
    
    # LLM settings
    LLM_API_KEY = "test_api_key"
    LLM_MODEL_NAME = "gpt-3.5-turbo"
    LLM_TEMPERATURE = 0.7
    LLM_MAX_TOKENS = 1000
    LLM_REQUEST_TIMEOUT = 30
    
    # OpenAI settings
    OPENAI_API_KEY = "test_openai_key"
    OPENAI_ORGANIZATION = "test_org"
    
    # Other settings
    STRICT_MODE = False

class MockMemoryManager:
    """Mock memory manager for testing."""
    def __init__(self):
        self.memories = {}
    
    def get_memory(self, key):
        return self.memories.get(key)
    
    def set_memory(self, key, value):
        self.memories[key] = value

class MockAgentMonitor:
    """Mock agent monitor for testing."""
    def __init__(self):
        self.metrics = {}
    
    def track_metric(self, name, value, tags=None):
        self.metrics[name] = (value, tags or {})
    
    def increment_counter(self, name, value=1, tags=None):
        current = self.metrics.get(name, (0, tags or {}))[0]
        self.metrics[name] = (current + value, tags or {})
    
    def record_gauge(self, name, value, tags=None):
        self.metrics[name] = (value, tags or {})
    
    def record_histogram(self, name, value, tags=None):
        if name not in self.metrics:
            self.metrics[name] = (0, tags or {}, 0, 0, 0, 0, 0)  # count, min, max, sum, avg
        count, min_val, max_val, total, avg = self.metrics[name][1:]
        count += 1
        min_val = min(min_val, value) if count > 1 else value
        max_val = max(max_val, value) if count > 1 else value
        total += value
        avg = total / count
        self.metrics[name] = (value, tags or {}, count, min_val, max_val, total, avg)

# Create singleton instances
settings = MockSettings()
memory_manager = MockMemoryManager()
agent_monitor = MockAgentMonitor()
