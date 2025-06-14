"""Simple tests for the agent registry without complex imports."""
import pytest

class TestSimpleRegistry:
    """Simple test cases for the agent registry pattern."""
    
    def test_basic_registry(self):
        """Test basic registry functionality."""
        # Create a simple registry
        class AgentRegistry:
            def __init__(self):
                self._agents = {}
                self._initialized_agents = {}
            
            def register(self, name, agent_cls):
                if name in self._agents:
                    raise ValueError(f"Agent {name} already registered")
                self._agents[name] = agent_cls
                return agent_cls
            
            def register_agent(self, agent_cls):
                name = getattr(agent_cls, 'agent_name', None)
                if not name:
                    raise ValueError("Agent class must have 'agent_name' attribute")
                return self.register(name, agent_cls)
            
            def get_agent(self, name):
                if name not in self._agents:
                    raise KeyError(f"No agent registered: {name}")
                if name not in self._initialized_agents:
                    self._initialized_agents[name] = self._agents[name]()
                return self._initialized_agents[name]
            
            def is_registered(self, name):
                return name in self._agents
            
            def get_all_agent_names(self):
                return list(self._agents.keys())
            
            def clear_cache(self):
                self._initialized_agents.clear()
        
        # Create a simple agent class
        class TestAgent:
            agent_name = "test_agent"
            description = "A test agent"
            
            def process(self, input_text):
                return f"Processed: {input_text}"
        
        # Initialize the registry
        registry = AgentRegistry()
        
        # Test registration
        registry.register("test_agent", TestAgent)
        assert registry.is_registered("test_agent")
        assert "test_agent" in registry.get_all_agent_names()
        
        # Test getting an agent
        agent = registry.get_agent("test_agent")
        assert agent.process("hello") == "Processed: hello"
        
        # Test decorator registration
        @registry.register_agent
        class AnotherAgent:
            agent_name = "another_agent"
            description = "Another test agent"
            
            def process(self, input_text):
                return f"Another: {input_text}"
        
        assert registry.is_registered("another_agent")
        agent = registry.get_agent("another_agent")
        assert agent.process("test") == "Another: test"
        
        # Test duplicate registration
        with pytest.raises(ValueError):
            registry.register("test_agent", TestAgent)
        
        # Test getting non-existent agent
        with pytest.raises(KeyError):
            registry.get_agent("non_existent")
        
        # Test clear cache
        registry.clear_cache()
        assert len(registry._initialized_agents) == 0
        
        # Test agent creation after clear
        agent = registry.get_agent("test_agent")
        assert agent is not None
        
        print("All basic registry tests passed!")

if __name__ == "__main__":
    test = TestSimpleRegistry()
    test.test_basic_registry()
    print("All tests completed successfully!")
