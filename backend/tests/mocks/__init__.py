"""Mock modules for testing the agent system."""

# This file makes the mocks directory a Python package, allowing for clean imports in test files.
# It also serves as a central place to expose mock objects for easier access in tests.

from .langchain_mock import MockLLM, MockChatOpenAI, MockAgentExecutor, MockPromptTemplate, MockModules
from .settings_mock import settings, Settings
from .langchain_openai_mock import MockLangchainOpenAIModule
from .agent_monitor_mock import agent_monitor, MockAgentMonitor
from .llm_config_mock import settings as llm_settings, Settings as LLMSettings

# Initialize the mock modules
import sys
sys.modules['langchain_openai'] = MockLangchainOpenAIModule()

# Mock the LLM config module
sys.modules['agent.llm.config'] = sys.modules[__name__]
sys.modules['agent.llm.config'].settings = llm_settings

# Mock the agent monitoring module
sys.modules['agent.monitoring.agent_monitor'] = sys.modules[__name__]
sys.modules['agent.monitoring.agent_monitor'].agent_monitor = agent_monitor

__all__ = [
    'MockLLM',
    'MockChatOpenAI',
    'MockAgentExecutor',
    'MockPromptTemplate',
    'MockModules',
    'MockLangchainOpenAIModule',
    'MockAgentMonitor',
    'agent_monitor',
    'settings',
    'Settings',
    'llm_settings',
    'LLMSettings',
]
