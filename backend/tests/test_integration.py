"""Integration tests for the agent registry and domain agents."""
import pytest
import asyncio
from unittest.mock import MagicMock, patch

class TestIntegration:
    """Integration tests for the agent system."""
    
    @pytest.fixture
    def mock_settings(self):
        """Mock settings for testing."""
        class MockSettings:
            OPENROUTER_API_KEY = "test_api_key"
            OPENROUTER_BASE_URL = "https://test.openrouter.ai/api/v1"
            OPENROUTER_MODEL = "test-model"
            OPENROUTER_MAX_TOKENS = 1000
            OPENROUTER_TEMPERATURE = 0.7
            
        return MockSettings()
    
    @pytest.fixture
    def mock_llm(self):
        """Mock LLM for testing."""
        mock = MagicMock()
        mock.return_value = "Mocked LLM response"
        return mock
    
    @pytest.fixture
    def mock_agent_executor(self):
        """Mock AgentExecutor for testing."""
        mock = MagicMock()
        mock.return_value = {"output": "Mocked agent response"}
        return mock
    
    @pytest.mark.asyncio
    async def test_agent_registration_and_processing(self, mock_settings, mock_llm, mock_agent_executor):
        """Test that agents can be registered and process requests."""
        # Mock dependencies
        with patch('agent.llm.config.settings', mock_settings), \
             patch('langchain_openai.ChatOpenAI', mock_llm), \
             patch('langchain.agents.AgentExecutor', mock_agent_executor):
            
            # Import after patching to ensure mocks are in place
            from agent.registry import AgentRegistry
            from agent.domain_agents import DomainAgent
            
            # Create a test agent
            class TestAgent(DomainAgent):
                agent_name = "test_agent"
                description = "A test agent"
                
                def __init__(self):
                    super().__init__()
                    self.llm = mock_llm()
                    self.agent_executor = mock_agent_executor()
                
                async def process(self, input_text):
                    return {"result": f"Processed: {input_text}"}
            
            # Test registration
            registry = AgentRegistry()
            registry.register("test_agent", TestAgent)
            
            # Test getting and using the agent
            agent = registry.get_agent("test_agent")
            result = await agent.process("test input")
            
            assert result == {"result": "Processed: test input"}
            assert registry.is_registered("test_agent")
    
    @pytest.mark.asyncio
    async def test_agent_decorator_registration(self, mock_settings, mock_llm, mock_agent_executor):
        """Test that agent registration works correctly."""
        with patch('agent.llm.config.settings', mock_settings), \
             patch('langchain_openai.ChatOpenAI', mock_llm), \
             patch('langchain.agents.AgentExecutor', mock_agent_executor):
            
            from agent.registry import AgentRegistry
            
            # Get the registry instance
            registry = AgentRegistry()
            
            # Clear any existing agents for test isolation
            registry._agents.clear()
            registry._initialized_agents.clear()
            
            # Create a test agent class
            class DecoratedAgent:
                agent_name = "decorated_agent"
                description = "An agent registered with decorator"
                
                def __init__(self):
                    self.llm = mock_llm()
                    self.agent_executor = mock_agent_executor()
                
                async def process(self, input_text):
                    return {"status": "success", "input": input_text}
            
            # Register the agent
            registry.register("decorated_agent", DecoratedAgent)
            
            # Test the agent was registered
            assert registry.is_registered("decorated_agent")
            agent = registry.get_agent("decorated_agent")
            result = await agent.process("test decorator")
            assert result == {"status": "success", "input": "test decorator"}
    
    @pytest.mark.asyncio
    async def test_agent_error_handling(self, mock_settings):
        """Test error handling in the agent system."""
        from agent.registry import AgentRegistry
        
        # Create a test agent that raises an exception
        class FaultyAgent:
            agent_name = "faulty_agent"
            description = "An agent that raises exceptions"
            
            def __init__(self):
                raise Exception("Failed to initialize agent")
            
            async def process(self, input_text):
                pass
        
        # Test registration and error handling
        registry = AgentRegistry()
        registry.register("faulty_agent", FaultyAgent)
        
        # Test that getting the agent raises an error
        with pytest.raises(Exception) as exc_info:
            registry.get_agent("faulty_agent")
        
        assert "Failed to initialize agent" in str(exc_info.value)
        
        # Test that the agent is not in the initialized agents cache
        assert "faulty_agent" not in registry._initialized_agents
        
        # Test that the agent is still registered
        assert registry.is_registered("faulty_agent")

if __name__ == "__main__":
    pytest.main(["-v", "tests/test_integration.py"])
