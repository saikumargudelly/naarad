"""Tests for the agent registry and domain agents."""
import pytest
import sys
from unittest.mock import patch, MagicMock, Mock

# Import mock modules first to avoid importing real modules
from tests.mocks import langchain_mock, settings_mock

# Patch the imports before importing the modules under test
sys.modules['langchain'] = langchain_mock.MockModules()
sys.modules['langchain.agents'] = langchain_mock.MockModules.agents
sys.modules['langchain.llms'] = langchain_mock.MockModules.llms
sys.modules['langchain.chat_models'] = langchain_mock.MockModules.chat_models
sys.modules['langchain.prompts'] = langchain_mock.MockModules.prompts

# Patch settings
sys.modules['agent.settings'] = settings_mock

# Now import the modules under test
from agent.registry import AgentRegistry, AgentInitializationError
from agent.domain_agents import (
    DomainAgent,
    TaskManagementAgent,
    CreativeWritingAgent,
    AnalysisAgent,
    ContextAwareChatAgent,
    create_domain_agents
)

# Test agent for registration
def test_agent_registration():
    """Test that agent classes can be registered and retrieved."""
    # Create a test agent class
    class TestAgent(DomainAgent):
        agent_name = "test_agent"
        agent_description = "A test agent"
        default_temperature = 0.7
        
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
    
    # Create a new registry instance
    registry = AgentRegistry()
    
    # Test registering the agent
    registry.register(TestAgent.agent_name, TestAgent)
    
    # Verify registration
    assert registry.is_registered("test_agent")
    assert "test_agent" in registry.get_all_agent_names()
    
    # Test getting the agent instance
    agent = registry.get_agent("test_agent")
    assert isinstance(agent, TestAgent)
    assert agent.agent_name == "test_agent"
    assert agent.agent_description == "A test agent"
    
    # Test getting all agent names
    agent_names = registry.get_all_agent_names()
    assert isinstance(agent_names, list)
    assert "test_agent" in agent_names
    
    # Test duplicate registration
    with pytest.raises(ValueError):
        registry.register("test_agent", TestAgent)
    
    # Test getting non-existent agent
    with pytest.raises(KeyError):
        registry.get_agent("non_existent_agent")
    
    # Test clearing the cache
    registry.clear_cache()
    assert len(registry._AgentRegistry__initialized_agents) == 0
    
    # Test registering with invalid agent class
    with pytest.raises(ValueError):
        registry.register("invalid_agent", object)

def test_domain_agent_decorator():
    """Test that the @DomainAgent.register decorator works correctly."""
    # Create a new registry to avoid test pollution
    registry = AgentRegistry()
    
    # Test that the decorator registers the agent class
    @DomainAgent.register
    class DecoratedTestAgent(DomainAgent):
        agent_name = "decorated_test_agent"
        agent_description = "A test agent registered via decorator"
        default_temperature = 0.5
        
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
    
    # Verify registration
    assert registry.is_registered("decorated_test_agent")
    
    # Test getting the agent instance
    agent = registry.get_agent("decorated_test_agent")
    assert isinstance(agent, DecoratedTestAgent)
    assert agent.agent_name == "decorated_test_agent"
    assert agent.agent_description == "A test agent registered via decorator"
    
    # Test that the decorator enforces required class variables
    with pytest.raises(ValueError):
        @DomainAgent.register
        class InvalidAgent(DomainAgent):
            # Missing required agent_name
            agent_description = "This should fail"
            
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
    
    # Test that the decorator can be used without parentheses
    @DomainAgent.register
    class AnotherTestAgent(DomainAgent):
        agent_name = "another_test_agent"
        agent_description = "Another test agent"
        
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
    
    assert registry.is_registered("another_test_agent")

def test_create_domain_agents():
    """Test that create_domain_agents returns all registered agents."""
    # Create a test agent class
    class TestAgent1(DomainAgent):
        agent_name = "test_agent_1"
        agent_description = "Test Agent 1"
        
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
    
    class TestAgent2(DomainAgent):
        agent_name = "test_agent_2"
        agent_description = "Test Agent 2"
        
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
    
    # Create a test registry
    registry = AgentRegistry()
    registry.register("test_agent_1", TestAgent1)
    registry.register("test_agent_2", TestAgent2)
    
    # Patch the agent_registry in the domain_agents module
    with patch('agent.domain_agents.agent_registry', registry):
        # Call the function
        agents = create_domain_agents()
        
        # Verify results
        assert isinstance(agents, dict)
        assert len(agents) == 2
        assert "test_agent_1" in agents
        assert "test_agent_2" in agents
        assert isinstance(agents["test_agent_1"], TestAgent1)
        assert isinstance(agents["test_agent_2"], TestAgent2)
        
        # Test that agents are cached
        assert agents["test_agent_1"] is agents["test_agent_1"]
        
        # Test with agent initialization failure
        registry.get_agent = MagicMock(side_effect=Exception("Test error"))
        
        # Should raise in strict mode
        with patch('agent.domain_agents.settings') as mock_settings:
            mock_settings.strict_mode = True
            with pytest.raises(AgentInitializationError):
                create_domain_agents()
        
        # Should not raise in non-strict mode
        with patch('agent.domain_agents.settings') as mock_settings:
            mock_settings.strict_mode = False
            agents = create_domain_agents()
            assert agents == {}

def test_agent_initialization_error():
    """Test error handling during agent initialization."""
    # Create a test registry
    registry = AgentRegistry()
    
    # Create a test agent that will raise an exception during initialization
    class FailingAgent(DomainAgent):
        agent_name = "failing_agent"
        agent_description = "An agent that fails during initialization"
        
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            raise Exception("Test initialization error")
    
    # Register the failing agent
    registry.register("failing_agent", FailingAgent)
    
    # Patch the agent_registry in the domain_agents module
    with patch('agent.domain_agents.agent_registry', registry):
        # Test with strict mode on - should raise
        with patch('agent.domain_agents.settings') as mock_settings:
            mock_settings.strict_mode = True
            with pytest.raises(AgentInitializationError) as exc_info:
                create_domain_agents()
            assert "Failed to initialize agent failing_agent" in str(exc_info.value)
        
        # Test with strict mode off - should return empty dict
        with patch('agent.domain_agents.settings') as mock_settings:
            mock_settings.strict_mode = False
            agents = create_domain_agents()
            assert agents == {}
    
    # Test with a non-existent agent class
    registry = AgentRegistry()
    registry._agents["nonexistent_agent"] = "not_a_class"
    
    with patch('agent.domain_agents.agent_registry', registry):
        with pytest.raises(AgentInitializationError) as exc_info:
            create_domain_agents()
        assert "Failed to initialize agent nonexistent_agent" in str(exc_info.value)

def test_agent_metadata():
    """Test that agent metadata is correctly set."""
    # Test TaskManagementAgent
    task_agent = TaskManagementAgent()
    assert task_agent.agent_name == "task_manager"
    assert isinstance(task_agent.agent_name, str)
    assert "task" in task_agent.agent_description.lower()
    assert isinstance(task_agent.agent_description, str)
    assert task_agent.default_temperature == 0.3
    assert 0 <= task_agent.default_temperature <= 1.0
    
    # Test CreativeWritingAgent
    creative_agent = CreativeWritingAgent()
    assert creative_agent.agent_name == "creative_writer"
    assert isinstance(creative_agent.agent_name, str)
    assert "creative" in creative_agent.agent_description.lower()
    assert isinstance(creative_agent.agent_description, str)
    assert creative_agent.default_temperature == 0.8
    assert 0 <= creative_agent.default_temperature <= 1.0
    assert hasattr(creative_agent, 'writing_styles')
    assert isinstance(creative_agent.writing_styles, list)
    
    # Test AnalysisAgent
    analysis_agent = AnalysisAgent()
    assert analysis_agent.agent_name == "analyst"
    assert isinstance(analysis_agent.agent_name, str)
    assert "analysis" in analysis_agent.agent_description.lower()
    assert isinstance(analysis_agent.agent_description, str)
    assert analysis_agent.default_temperature == 0.3
    assert 0 <= analysis_agent.default_temperature <= 1.0
    assert hasattr(analysis_agent, 'analysis_methods')
    assert isinstance(analysis_agent.analysis_methods, list)
    
    # Test ContextAwareChatAgent
    context_agent = ContextAwareChatAgent()
    assert context_agent.agent_name == "context_manager"
    assert isinstance(context_agent.agent_name, str)
    assert "context" in context_agent.agent_description.lower()
    assert isinstance(context_agent.agent_description, str)
    assert context_agent.default_temperature == 0.5
    assert 0 <= context_agent.default_temperature <= 1.0
    assert hasattr(context_agent, 'conversation_history')
    assert isinstance(context_agent.conversation_history, dict)
    
    # Test that all agents have the required base methods
    for agent in [task_agent, creative_agent, analysis_agent, context_agent]:
        assert hasattr(agent, 'process')
        assert callable(agent.process)
        assert hasattr(agent, '_create_llm')
        assert callable(agent._create_llm)
        assert hasattr(agent, '_create_agent')
        assert callable(agent._create_agent)
        assert hasattr(agent, '_update_agent_state')
        assert callable(agent._update_agent_state)

if __name__ == "__main__":
    pytest.main(["-v", __file__])
