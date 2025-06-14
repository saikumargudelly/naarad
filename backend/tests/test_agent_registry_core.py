"""Core tests for the AgentRegistry and domain agents."""
import pytest
from agent.registry import AgentRegistry, AgentInitializationError
from agent.domain_agents import DomainAgent, create_domain_agents

class TestAgentRegistry:
    """Test cases for AgentRegistry class."""
    
    @pytest.fixture(autouse=True)
    def setup(self, agent_registry):
        """Setup test environment."""
        self.registry = agent_registry
        self.registry._agents = {}  # Clear any existing registrations
        self.registry._initialized_agents = {}
    
    def test_register_and_get_agent(self):
        """Test registering and retrieving an agent."""
        # Define a test agent class
        class TestAgent(DomainAgent):
            agent_name = "test_agent"
            description = "A test agent"
            
            def __init__(self):
                super().__init__()
                
            async def process(self, *args, **kwargs):
                return {"result": "Test response"}
        
        # Register the agent
        self.registry.register("test_agent", TestAgent)
        
        # Test retrieval
        agent = self.registry.get_agent("test_agent")
        assert agent is not None
        assert isinstance(agent, TestAgent)
        
        # Test agent methods
        import asyncio
        response = asyncio.run(agent.process())
        assert response == {"result": "Test response"}
        
        # Test listing agents
        assert "test_agent" in self.registry.get_all_agent_names()
        assert self.registry.is_registered("test_agent")
    
    def test_duplicate_registration(self):
        """Test that duplicate agent registration raises an error."""
        class TestAgent:
            agent_name = "test_agent"
            
        self.registry.register("test_agent", TestAgent)
        
        with pytest.raises(ValueError):
            self.registry.register("test_agent", TestAgent)
    
    def test_nonexistent_agent(self):
        """Test that getting a non-existent agent raises an error."""
        with pytest.raises(KeyError):
            self.registry.get_agent("nonexistent_agent")

class TestDomainAgents:
    """Test cases for domain agents."""
    
    @pytest.fixture(autouse=True)
    def setup(self, agent_registry):
        """Setup test environment."""
        self.registry = agent_registry
        self.registry._agents = {}  # Clear any existing registrations
        self.registry._initialized_agents = {}
    
    def test_domain_agent_registration(self):
        """Test domain agent registration and usage."""
        # Define a domain agent
        @self.registry.register_agent
        class TestDomainAgent(DomainAgent):
            agent_name = "test_domain_agent"
            description = "A test domain agent"
            
            async def process(self, *args, **kwargs):
                return {"result": f"Processed by {self.agent_name}"}
        
        # Test retrieval
        agent = self.registry.get_agent("test_domain_agent")
        assert agent is not None
        assert isinstance(agent, TestDomainAgent)
        
        # Test agent functionality
        import asyncio
        response = asyncio.run(agent.process())
        assert response == {"result": "Processed by test_domain_agent"}
    
    def test_create_domain_agents(self):
        """Test creating all domain agents."""
        # Register test agents
        @self.registry.register_agent
        class TaskAgent(DomainAgent):
            agent_name = "task_manager"
            description = "Manages tasks"
            
            async def process(self, *args, **kwargs):
                return {"result": "Task processed"}
        
        @self.registry.register_agent
        class WriterAgent(DomainAgent):
            agent_name = "creative_writer"
            description = "Generates creative content"
            
            async def process(self, *args, **kwargs):
                return {"result": "Content generated"}
        
        # Create all domain agents
        agents = create_domain_agents()
        
        # Verify all agents were created
        assert len(agents) >= 2
        assert "task_manager" in agents
        assert "creative_writer" in agents
        
        # Test agent functionality
        import asyncio
        task_result = asyncio.run(agents["task_manager"].process())
        assert task_result == {"result": "Task processed"}
        
        writer_result = asyncio.run(agents["creative_writer"].process())
        assert writer_result == {"result": "Content generated"}
