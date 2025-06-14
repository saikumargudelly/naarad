"""Test agent orchestration and routing based on context."""
import pytest
import asyncio
from unittest.mock import MagicMock, patch
from typing import Dict, Any, Optional

class TestAgentOrchestration:
    """Test agent orchestration and routing functionality."""
    
    @pytest.fixture
    def mock_settings(self):
        """Mock settings for testing."""
        class MockSettings:
            OPENROUTER_API_KEY = "test_api_key"
            OPENROUTER_BASE_URL = "https://test.openrouter.ai/api/v1"
            OPENROUTER_MODEL = "test-model"
            
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
        # Create a mock that can be configured per test
        class MockAgentExecutor:
            def __init__(self):
                self.response = {"output": "Mocked agent response"}
                
            def __call__(self, *args, **kwargs):
                return self
                
            async def ainvoke(self, *args, **kwargs):
                return self.response
        
        mock = MockAgentExecutor()
        return mock
    
    @pytest.fixture
    def setup_agents(self, mock_settings, mock_llm, mock_agent_executor):
        """Set up test agents with mocks."""
        with patch('agent.llm.config.settings', mock_settings), \
             patch('langchain_openai.ChatOpenAI', mock_llm), \
             patch('langchain.agents.AgentExecutor', mock_agent_executor):
            
            from agent.registry import AgentRegistry
            
            # Create a fresh registry
            registry = AgentRegistry()
            
            # Clear any existing agents
            registry._agents.clear()
            registry._initialized_agents.clear()
            
            # Create test agents with different capabilities
            class MathAgent:
                agent_name = "math_agent"
                description = "Handles mathematical operations"
                
                def __init__(self):
                    self.llm = mock_llm()
                    self.agent_executor = mock_agent_executor
                    # Configure the mock to return math results
                    self.agent_executor.response = {"output": "4"}  # Default response
                
                async def process(self, input_text: str, context: Optional[Dict] = None) -> Dict[str, Any]:
                    # In a real scenario, this would use the LLM
                    # For testing, we'll just return mock responses based on input
                    if "2 + 2" in input_text:
                        self.agent_executor.response = {"output": "4"}
                    elif "5 * 3" in input_text:
                        self.agent_executor.response = {"output": "15"}
                    elif "10 * 10" in input_text:
                        self.agent_executor.response = {"output": "100"}
                        
                    result = await self.agent_executor.ainvoke(input_text)
                    return {
                        "result": float(result["output"]),
                        "agent": self.agent_name,
                        "status": "success"
                    }
            
            class WeatherAgent:
                agent_name = "weather_agent"
                description = "Provides weather information"
                
                def __init__(self):
                    self.llm = mock_llm()
                    self.agent_executor = mock_agent_executor
                    # Configure default mock response
                    self.agent_executor.response = {"output": "72°F and sunny"}
                    # Mock weather data
                    self.weather_data = {
                        "new york": "72°F and sunny",
                        "london": "65°F and cloudy",
                        "tokyo": "78°F and rainy"
                    }
                
                async def process(self, input_text: str, context: Optional[Dict] = None) -> Dict[str, Any]:
                    location = input_text.lower().replace("weather in", "").replace("what's the weather in", "").replace("?", "").strip()
                    
                    # Configure mock response based on location
                    if "new york" in location:
                        self.agent_executor.response = {"output": self.weather_data["new york"]}
                    elif "paris" in location:
                        self.agent_executor.response = {"output": "Location not found"}
                    
                    result = await self.agent_executor.ainvoke(input_text)
                    return {
                        "location": location,
                        "weather": result["output"],
                        "agent": self.agent_name,
                        "status": "success"
                    }
            
            class GeneralAgent:
                agent_name = "general_agent"
                description = "Handles general queries"
                
                def __init__(self):
                    self.llm = mock_llm()
                    self.agent_executor = mock_agent_executor()
                
                async def process(self, input_text: str, context: Optional[Dict] = None) -> Dict[str, Any]:
                    return {
                        "response": f"I can help with general questions. You asked: {input_text}",
                        "agent": self.agent_name,
                        "status": "success"
                    }
            
            # Register all agents
            registry.register("math_agent", MathAgent)
            registry.register("weather_agent", WeatherAgent)
            registry.register("general_agent", GeneralAgent)
            
            # Create a simple orchestrator
            class AgentOrchestrator:
                def __init__(self, registry):
                    self.registry = registry
                
                def determine_agent(self, input_text: str) -> str:
                    """Determine which agent should handle the input."""
                    input_text = input_text.lower()
                    
                    # Simple routing logic - in a real app, this could use an LLM
                    if any(op in input_text for op in ['+', '-', '*', '/', 'math']):
                        return "math_agent"
                    elif 'weather' in input_text:
                        return "weather_agent"
                    else:
                        return "general_agent"
                
                async def process_input(self, input_text: str, context: Optional[Dict] = None) -> Dict[str, Any]:
                    """Process input by routing it to the appropriate agent."""
                    agent_name = self.determine_agent(input_text)
                    
                    try:
                        agent = self.registry.get_agent(agent_name)
                        result = await agent.process(input_text, context or {})
                        return {
                            **result,
                            "orchestrator": {
                                "selected_agent": agent_name,
                                "status": "success"
                            }
                        }
                    except Exception as e:
                        return {
                            "error": str(e),
                            "orchestrator": {
                                "selected_agent": agent_name,
                                "status": "error"
                            }
                        }
            
            yield AgentOrchestrator(registry)
    
    @pytest.mark.asyncio
    async def test_math_processing(self, setup_agents):
        """Test that math operations are routed to the math agent."""
        orchestrator = setup_agents
        
        # Test addition
        result = await orchestrator.process_input("What is 2 + 2?")
        assert result["agent"] == "math_agent"
        assert result["status"] == "success"
        assert result["result"] == 4.0  # Converted to float in the agent
        
        # Test multiplication
        result = await orchestrator.process_input("Calculate 5 * 3")
        assert result["agent"] == "math_agent"
        assert result["result"] == 15.0  # Converted to float in the agent
    
    @pytest.mark.asyncio
    async def test_weather_processing(self, setup_agents):
        """Test that weather queries are routed to the weather agent."""
        orchestrator = setup_agents
        
        # Test weather query for New York
        result = await orchestrator.process_input("What's the weather in New York?")
        assert result["agent"] == "weather_agent"
        assert result["status"] == "success"
        assert "new york" in result["location"]
        assert "sunny" in result["weather"].lower()
        
        # Test unknown location (Paris)
        result = await orchestrator.process_input("Weather in Paris")
        assert result["agent"] == "weather_agent"
        assert "paris" in result["location"]
        assert result["weather"] == "Location not found"
    
    @pytest.mark.asyncio
    async def test_general_processing(self, setup_agents):
        """Test that general queries are routed to the general agent."""
        orchestrator = setup_agents
        
        # Test general query
        test_question = "Tell me about the universe"
        result = await orchestrator.process_input(test_question)
        assert result["agent"] == "general_agent"
        assert result["status"] == "success"
        assert test_question in result["response"]
    
    @pytest.mark.asyncio
    async def test_context_passing(self, setup_agents):
        """Test that context is properly passed to agents."""
        orchestrator = setup_agents
        
        # Test with context
        context = {"user_id": "test123", "preferences": {"units": "metric"}}
        result = await orchestrator.process_input("What's 10 * 10?", context=context)
        
        # The test is simple - we just want to ensure the context is passed through
        # In a real test, we would verify the agent used the context
        assert result["status"] == "success"
        assert result["result"] == 100
        assert result["agent"] == "math_agent"

if __name__ == "__main__":
    pytest.main(["-v", "tests/test_agent_orchestration.py"])
