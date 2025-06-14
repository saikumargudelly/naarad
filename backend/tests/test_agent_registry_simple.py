"""Simplified tests for the agent registry."""
import pytest
from unittest.mock import MagicMock, patch

# Import test utilities
from tests.test_utils import settings, memory_manager, agent_monitor

# Mock the necessary modules before importing the code under test
import sys

class MockDomainAgent:
    """Mock DomainAgent class for testing."""
    def __init__(self, agent_name, description, temperature=0.7):
        self.agent_name = agent_name
        self.description = description
        self.temperature = temperature
    
    async def process(self, *args, **kwargs):
        return {"response": f"Processed by {self.agent_name}"}

# Create mock agent classes
class MockTaskManagementAgent(MockDomainAgent):
    agent_name = "task_manager"
    description = "Manages tasks and to-dos"
    default_temperature = 0.3

class MockCreativeWritingAgent(MockDomainAgent):
    agent_name = "creative_writer"
    description = "Generates creative content"
    default_temperature = 0.8

class MockAnalysisAgent(MockDomainAgent):
    agent_name = "analyst"
    description = "Performs data analysis"
    default_temperature = 0.3

class MockContextAwareChatAgent(MockDomainAgent):
    agent_name = "context_manager"
    description = "Manages conversation context"
    default_temperature = 0.5

# Now import the actual registry code
from agent.registry import AgentRegistry, AgentInitializationError

# Test the agent registry
def test_agent_registration():
    """Test that agents can be registered and retrieved."""
    registry = AgentRegistry()
    
    # Test registering a new agent
    registry.register("test_agent", MockDomainAgent)
    assert registry.is_registered("test_agent")
    
    # Test getting an agent
    agent = registry.get_agent("test_agent")
    assert isinstance(agent, MockDomainAgent)
    assert agent.agent_name == "test_agent"
    
    # Test getting all agent names
    agent_names = registry.get_all_agent_names()
    assert "test_agent" in agent_names
    
    # Test duplicate registration
    with pytest.raises(ValueError):
        registry.register("test_agent", MockDomainAgent)
    
    # Test getting non-existent agent
    with pytest.raises(KeyError):
        registry.get_agent("non_existent_agent")
    
    # Test clearing the cache
    registry.clear_cache()
    assert len(registry._AgentRegistry__initialized_agents) == 0

def test_create_domain_agents():
    """Test creating domain agents."""
    # Mock the registry to return our mock agents
    with patch('agent.domain_agents.agent_registry') as mock_registry:
        # Set up the mock registry
        mock_registry.get_all_agent_names.return_value = [
            "task_manager",
            "creative_writer",
            "analyst",
            "context_manager"
        ]
        
        # Configure get_agent to return the appropriate mock agent class
        def get_agent_side_effect(agent_name):
            if agent_name == "task_manager":
                return MockTaskManagementAgent()
            elif agent_name == "creative_writer":
                return MockCreativeWritingAgent()
            elif agent_name == "analyst":
                return MockAnalysisAgent()
            elif agent_name == "context_manager":
                return MockContextAwareChatAgent()
            raise KeyError(f"Unknown agent: {agent_name}")
        
        mock_registry.get_agent.side_effect = get_agent_side_effect
        
        # Import the function we want to test
        from agent.domain_agents import create_domain_agents
        
        # Test with strict mode on
        with patch('agent.domain_agents.settings') as mock_settings:
            mock_settings.strict_mode = True
            agents = create_domain_agents()
            
            # Verify we got all the agents
            assert len(agents) == 4
            assert "task_manager" in agents
            assert "creative_writer" in agents
            assert "analyst" in agents
            assert "context_manager" in agents
            
            # Verify the agents have the correct types
            assert isinstance(agents["task_manager"], MockTaskManagementAgent)
            assert isinstance(agents["creative_writer"], MockCreativeWritingAgent)
            assert isinstance(agents["analyst"], MockAnalysisAgent)
            assert isinstance(agents["context_manager"], MockContextAwareChatAgent)

if __name__ == "__main__":
    pytest.main(["-v", __file__])
