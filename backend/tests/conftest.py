"""Pytest configuration and fixtures for agent tests."""
import sys
import pytest
from unittest.mock import MagicMock, Mock, patch
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Mock settings before any imports
class MockSettings:
    DEBUG = True
    TESTING = True
    ENVIRONMENT = "test"
    LLM_API_KEY = "test_key"
    LLM_MODEL_NAME = "test_model"
    LLM_TEMPERATURE = 0.7
    OPENAI_API_KEY = "test_openai_key"
    STRICT_MODE = False

# Mock the settings module
sys.modules['agent.settings'] = Mock()
sys.modules['agent.settings'].settings = MockSettings()

# Mock LLM classes
class MockLLM:
    def __init__(self, *args, **kwargs):
        pass

class MockChatOpenAI:
    def __init__(self, *args, **kwargs):
        pass

    async def ainvoke(self, *args, **kwargs):
        return "Mocked response"

class MockAgentExecutor:
    def __init__(self, *args, **kwargs):
        pass

    async def ainvoke(self, *args, **kwargs):
        return {"output": "Mocked agent response"}

class MockPromptTemplate:
    @classmethod
    def from_template(cls, *args, **kwargs):
        return cls()

# Mock the modules
sys.modules['langchain'] = MagicMock()
sys.modules['langchain.llms'] = MagicMock()
sys.modules['langchain.llms'].OpenAI = MockLLM
sys.modules['langchain.chat_models'] = MagicMock()
sys.modules['langchain.chat_models'].ChatOpenAI = MockChatOpenAI
sys.modules['langchain.agents'] = MagicMock()
sys.modules['langchain.agents'].AgentExecutor = MockAgentExecutor
sys.modules['langchain.prompts'] = MagicMock()
sys.modules['langchain.prompts'].PromptTemplate = MockPromptTemplate

# Mock langchain_openai
sys.modules['langchain_openai'] = MagicMock()
sys.modules['langchain_openai'].ChatOpenAI = MockChatOpenAI

# Mock LLM config and domain_agents
class LLMConfig:
    settings = MockSettings()

# Mock the modules
sys.modules['agent.llm'] = MagicMock()
sys.modules['agent.llm.config'] = LLMConfig()
sys.modules['agent.domain_agents'] = MagicMock()

# Create a mock for the settings import
sys.modules['agent.llm.config'].settings = MockSettings()

# Mock the DomainAgent base class
class MockDomainAgent:
    agent_name = "base_agent"
    description = "Base agent class"
    
    async def process(self, *args, **kwargs):
        return {"result": "Base agent response"}

sys.modules['agent.domain_agents'].DomainAgent = MockDomainAgent

# Mock other required modules
class MockMemoryManager:
    def __init__(self):
        self.memories = {}
    
    def get_memory(self, key):
        return self.memories.get(key)
    
    def set_memory(self, key, value):
        self.memories[key] = value

class MockAgentMonitor:
    def __init__(self):
        self.metrics = {}
    
    def track_metric(self, name, value, tags=None):
        self.metrics[name] = (value, tags or {})
    
    def increment_counter(self, name, value=1, tags=None):
        current = self.metrics.get(name, (0, tags or {}))[0]
        self.metrics[name] = (current + value, tags or {})
    
    def record_gauge(self, name, value, tags=None):
        self.metrics[name] = (value, tags or {})

# Set up the mock modules
sys.modules['agent.memory.memory_manager'] = MagicMock()
sys.modules['agent.memory.memory_manager'].memory_manager = MockMemoryManager()
sys.modules['agent.monitoring.agent_monitor'] = MagicMock()
sys.modules['agent.monitoring.agent_monitor'].agent_monitor = MockAgentMonitor()

# Now import the modules under test
with patch.dict('sys.modules', sys.modules):
    from agent.registry import AgentRegistry, AgentInitializationError
    from agent.domain_agents import (
        DomainAgent,
        TaskManagementAgent,
        CreativeWritingAgent,
        AnalysisAgent,
        ContextAwareChatAgent,
        create_domain_agents
    )

@pytest.fixture
def mock_settings():
    """Return a mock settings object."""
    return settings_mock.settings

@pytest.fixture
def mock_llm():
    """Return a mock LLM."""
    return langchain_mock.MockLLM()

@pytest.fixture
def mock_chat_openai():
    """Return a mock ChatOpenAI instance."""
    return langchain_mock.MockChatOpenAI()

@pytest.fixture
def mock_agent_executor():
    """Return a mock AgentExecutor."""
    return langchain_mock.MockAgentExecutor()

@pytest.fixture
def agent_registry():
    """Return a fresh AgentRegistry instance for testing."""
    return AgentRegistry()

@pytest.fixture
def task_management_agent():
    """Return a TaskManagementAgent instance."""
    return TaskManagementAgent()

@pytest.fixture
def creative_writing_agent():
    """Return a CreativeWritingAgent instance."""
    return CreativeWritingAgent()

@pytest.fixture
def analysis_agent():
    """Return an AnalysisAgent instance."""
    return AnalysisAgent()

@pytest.fixture
def context_aware_chat_agent():
    """Return a ContextAwareChatAgent instance."""
    return ContextAwareChatAgent()

@pytest.fixture(autouse=True)
def reset_agent_registry():
    """Reset the AgentRegistry before each test."""
    registry = AgentRegistry()
    registry._agents = {}
    registry._initialized_agents = {}
    return registry
