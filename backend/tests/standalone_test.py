"""Standalone tests that don't depend on the actual codebase."""
import pytest

class TestStandaloneAgentRegistry:
    """Test the agent registry functionality in a standalone manner."""
    
    def test_agent_registry_basic(self):
        """Test basic agent registry operations."""
        # Create a simple agent registry
        class AgentRegistry:
            def __init__(self):
                self._agents = {}
                self._initialized_agents = {}
            
            def register(self, name, agent_cls):
                if name in self._agents:
                    raise ValueError(f"Agent {name} already registered")
                self._agents[name] = agent_cls
            
            def get_agent(self, name):
                if name not in self._agents:
                    raise KeyError(f"No agent registered with name: {name}")
                
                # Lazy initialization
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
            def __init__(self, name="test_agent"):
                self.name = name
            
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
        assert isinstance(agent, TestAgent)
        assert agent.process("hello") == "Processed: hello"
        
        # Test lazy initialization
        registry.clear_cache()
        assert len(registry._initialized_agents) == 0
        agent = registry.get_agent("test_agent")
        assert len(registry._initialized_agents) == 1
        
        # Test duplicate registration
        with pytest.raises(ValueError):
            registry.register("test_agent", TestAgent)
        
        # Test getting non-existent agent
        with pytest.raises(KeyError):
            registry.get_agent("non_existent_agent")
        
        # Test clear cache
        registry.clear_cache()
        assert len(registry._initialized_agents) == 0

class TestDomainAgentPattern:
    """Test the domain agent pattern."""
    
    def test_domain_agent_registration(self):
        """Test domain agent registration pattern."""
        # Create a simple registry
        class AgentRegistry:
            def __init__(self):
                self._agents = {}
            
            def register(self, agent_cls):
                agent_name = getattr(agent_cls, 'agent_name', None)
                if not agent_name:
                    raise ValueError("Agent class must have an 'agent_name' attribute")
                if agent_name in self._agents:
                    raise ValueError(f"Agent {agent_name} already registered")
                self._agents[agent_name] = agent_cls
                return agent_cls
            
            def get_agent(self, name):
                if name not in self._agents:
                    raise KeyError(f"No agent registered with name: {name}")
                return self._agents[name]()
            
            def get_all_agent_names(self):
                return list(self._agents.keys())
        
        # Create a registry instance
        registry = AgentRegistry()
        
        # Create a decorator
        def domain_agent(cls):
            return registry.register(cls)
        
        # Define some domain agents
        @domain_agent
        class TaskManager:
            agent_name = "task_manager"
            description = "Manages tasks and to-dos"
            
            def process(self, input_text):
                return f"Task manager processing: {input_text}"
        
        @domain_agent
        class CreativeWriter:
            agent_name = "creative_writer"
            description = "Generates creative content"
            
            def process(self, input_text):
                return f"Creative writer processing: {input_text}"
        
        # Test the registration
        assert "task_manager" in registry.get_all_agent_names()
        assert "creative_writer" in registry.get_all_agent_names()
        
        # Test agent instantiation and processing
        task_agent = registry.get_agent("task_manager")
        assert task_agent.process("test") == "Task manager processing: test"
        
        writer_agent = registry.get_agent("creative_writer")
        assert writer_agent.process("story") == "Creative writer processing: story"
        
        # Test duplicate registration
        with pytest.raises(ValueError):
            @domain_agent
            class DuplicateTaskManager:
                agent_name = "task_manager"  # Duplicate name
                description = "Duplicate task manager"

if __name__ == "__main__":
    pytest.main(["-v", __file__])
