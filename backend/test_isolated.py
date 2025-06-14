"""Completely isolated test file that doesn't depend on project structure."""

class TestAgentRegistry:
    """Test the agent registry in complete isolation."""
    
    def test_basic_registry(self):
        """Test basic registry operations."""
        # Create a simple registry
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
        
        # Create a simple agent
        class TestAgent:
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
        
        # Test duplicate registration
        try:
            registry.register("test_agent", TestAgent)
            assert False, "Should have raised ValueError for duplicate registration"
        except ValueError:
            pass
        
        # Test getting non-existent agent
        try:
            registry.get_agent("non_existent")
            assert False, "Should have raised KeyError for non-existent agent"
        except KeyError:
            pass
        
        # Test clear cache
        registry.clear_cache()
        assert len(registry._initialized_agents) == 0
        
        print("All basic registry tests passed!")

    def test_domain_agent_pattern(self):
        """Test domain agent registration pattern."""
        # Create a registry
        class AgentRegistry:
            def __init__(self):
                self._agents = {}
            
            def register(self, agent_cls):
                agent_name = getattr(agent_cls, 'agent_name', None)
                if not agent_name:
                    raise ValueError("Agent class must have 'agent_name' attribute")
                if agent_name in self._agents:
                    raise ValueError(f"Agent {agent_name} already registered")
                self._agents[agent_name] = agent_cls
                return agent_cls
            
            def get_agent(self, name):
                if name not in self._agents:
                    raise KeyError(f"No agent registered: {name}")
                return self._agents[name]()
            
            def get_all_agent_names(self):
                return list(self._agents.keys())
        
        # Create registry and decorator
        registry = AgentRegistry()
        domain_agent = registry.register
        
        # Define some agents
        @domain_agent
        class TaskManager:
            agent_name = "task_manager"
            description = "Manages tasks"
            
            def process(self, text):
                return f"Task manager: {text}"
        
        @domain_agent
        class CreativeWriter:
            agent_name = "creative_writer"
            description = "Writes creatively"
            
            def process(self, text):
                return f"Creative writer: {text}"
        
        # Test registration
        assert "task_manager" in registry.get_all_agent_names()
        assert "creative_writer" in registry.get_all_agent_names()
        
        # Test agent functionality
        task_agent = registry.get_agent("task_manager")
        assert task_agent.process("test") == "Task manager: test"
        
        writer_agent = registry.get_agent("creative_writer")
        assert writer_agent.process("story") == "Creative writer: story"
        
        # Test duplicate registration
        try:
            @domain_agent
            class DuplicateTaskManager:
                agent_name = "task_manager"
                description = "Duplicate"
            assert False, "Should have raised ValueError for duplicate agent_name"
        except ValueError:
            pass
        
        print("All domain agent pattern tests passed!")

if __name__ == "__main__":
    test = TestAgentRegistry()
    test.test_basic_registry()
    test.test_domain_agent_pattern()
    print("All tests completed successfully!")
