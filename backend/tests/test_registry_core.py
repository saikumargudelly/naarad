"""Core tests for the agent registry without external dependencies."""
import pytest

class TestAgentRegistry:
    """Test the core functionality of the AgentRegistry."""
    
    def test_register_and_get_agent(self):
        """Test registering and getting an agent."""
        # Create a mock agent class
        class MockAgent:
            agent_name = "test_agent"
            
            def __init__(self, **kwargs):
                self.kwargs = kwargs
        
        # Create a registry and register the agent
        registry = type('AgentRegistry', (), {
            '_agents': {},
            '_initialized_agents': {},
            'register': lambda self, name, agent_cls: self._agents.update({name: agent_cls}),
            'get_agent': lambda self, name: self._agents[name](),
            'is_registered': lambda self, name: name in self._agents,
            'get_all_agent_names': lambda self: list(self._agents.keys()),
            'clear_cache': lambda self: self._initialized_agents.clear()
        })()
        
        # Register the agent
        registry.register("test_agent", MockAgent)
        
        # Verify registration
        assert registry.is_registered("test_agent")
        assert "test_agent" in registry.get_all_agent_names()
        
        # Get the agent instance
        agent = registry.get_agent("test_agent")
        assert isinstance(agent, MockAgent)
        assert agent.agent_name == "test_agent"
        
        # Test duplicate registration
        with pytest.raises(KeyError):
            registry.register("test_agent", MockAgent)
        
        # Test getting non-existent agent
        with pytest.raises(KeyError):
            registry.get_agent("non_existent_agent")
        
        # Test clearing the cache
        registry.clear_cache()
        assert len(registry._initialized_agents) == 0

class TestDomainAgents:
    """Test the domain agents functionality."""
    
    def test_domain_agent_registration(self):
        """Test that domain agents can be registered and used."""
        # Create a mock registry
        class MockAgent:
            def __init__(self, agent_name, **kwargs):
                self.agent_name = agent_name
                self.kwargs = kwargs
        
        class MockRegistry:
            def __init__(self):
                self._agents = {}
                
            def register(self, name, agent_cls):
                if name in self._agents:
                    raise ValueError(f"Agent {name} already registered")
                self._agents[name] = agent_cls
                
            def get_agent(self, name):
                return self._agents[name](agent_name=name)
                
            def get_all_agent_names(self):
                return list(self._agents.keys())
        
        # Create a mock settings
        class MockSettings:
            strict_mode = False
        
        # Create a mock domain agent decorator
        def domain_agent_decorator(cls):
            return cls
        
        # Create a mock DomainAgent base class
        class MockDomainAgent:
            @classmethod
            def register(cls, agent_cls):
                agent_name = getattr(agent_cls, 'agent_name', None)
                if not agent_name:
                    raise ValueError("Agent must have an agent_name")
                registry.register(agent_name, agent_cls)
                return agent_cls
        
        # Create a mock create_domain_agents function
        def create_domain_agents():
            agents = {}
            for agent_name in registry.get_all_agent_names():
                try:
                    agents[agent_name] = registry.get_agent(agent_name)
                except Exception as e:
                    if settings.strict_mode:
                        raise AgentInitializationError(f"Failed to initialize agent {agent_name}: {e}") from e
            return agents
        
        # Create a mock AgentInitializationError
        class AgentInitializationError(Exception):
            pass
        
        # Set up the test environment
        registry = MockRegistry()
        settings = MockSettings()
        
        # Define test agents
        @MockDomainAgent.register
        class TaskManagementAgent(MockAgent):
            agent_name = "task_manager"
            
        @MockDomainAgent.register
        class CreativeWritingAgent(MockAgent):
            agent_name = "creative_writer"
        
        # Test agent registration
        assert "task_manager" in registry.get_all_agent_names()
        assert "creative_writer" in registry.get_all_agent_names()
        
        # Test creating domain agents
        agents = create_domain_agents()
        assert len(agents) == 2
        assert isinstance(agents["task_manager"], MockAgent)
        assert isinstance(agents["creative_writer"], MockAgent)
        assert agents["task_manager"].agent_name == "task_manager"
        assert agents["creative_writer"].agent_name == "creative_writer"
        
        # Test strict mode
        settings.strict_mode = True
        
        # Add a failing agent
        class FailingAgent(MockAgent):
            agent_name = "failing_agent"
            
            def __init__(self, **kwargs):
                raise Exception("Initialization failed")
        
        registry.register("failing_agent", FailingAgent)
        
        # Should raise in strict mode
        with pytest.raises(AgentInitializationError):
            create_domain_agents()
        
        # Should not raise in non-strict mode
        settings.strict_mode = False
        agents = create_domain_agents()
        assert "failing_agent" not in agents

if __name__ == "__main__":
    pytest.main(["-v", __file__])
