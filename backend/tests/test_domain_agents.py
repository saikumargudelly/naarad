"""Tests for domain agents and registry integration."""
import pytest
import asyncio

class TestDomainAgents:
    """Test cases for domain agents and registry integration."""
    
    @pytest.fixture
    def registry(self):
        """Create a clean registry instance for testing."""
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
        
        return AgentRegistry()
    
    @pytest.fixture
    def domain_agents(self, registry):
        """Register and return test domain agents."""
        # Task Management Agent
        @registry.register_agent
        class TaskManagementAgent:
            agent_name = "task_manager"
            description = "Manages tasks and to-do lists"
            
            async def process(self, task):
                return {"status": "completed", "task": task, "agent": self.agent_name}
        
        # Creative Writing Agent
        @registry.register_agent
        class CreativeWritingAgent:
            agent_name = "creative_writer"
            description = "Generates creative content"
            
            async def process(self, prompt):
                return {"content": f"Creative response to: {prompt}", "agent": self.agent_name}
        
        # Analysis Agent
        @registry.register_agent
        class AnalysisAgent:
            agent_name = "analyst"
            description = "Performs data analysis"
            
            async def process(self, data):
                return {
                    "analysis": f"Analysis of {len(data)} items",
                    "summary": "Sample analysis summary",
                    "agent": self.agent_name
                }
        
        # Context-Aware Chat Agent
        @registry.register_agent
        class ContextAwareChatAgent:
            agent_name = "context_aware_chat"
            description = "Maintains conversation context"
            
            def __init__(self):
                self.context = {}
            
            async def process(self, message, context=None):
                if context:
                    self.context.update(context)
                return {
                    "response": f"Response to: {message}",
                    "context": self.context,
                    "agent": self.agent_name
                }
        
        return {
            "task_manager": TaskManagementAgent,
            "creative_writer": CreativeWritingAgent,
            "analyst": AnalysisAgent,
            "context_aware_chat": ContextAwareChatAgent
        }
    
    @pytest.mark.asyncio
    async def test_agent_processing(self, registry, domain_agents):
        """Test that agents can be retrieved and process requests."""
        # Test task manager
        task_agent = registry.get_agent("task_manager")
        task_result = await task_agent.process("Complete project")
        assert task_result["status"] == "completed"
        assert task_result["agent"] == "task_manager"
        
        # Test creative writer
        writer_agent = registry.get_agent("creative_writer")
        content_result = await writer_agent.process("Write a story")
        assert "Creative response to:" in content_result["content"]
        assert content_result["agent"] == "creative_writer"
    
    @pytest.mark.asyncio
    async def test_agent_context(self, registry):
        """Test context maintenance in context-aware agents."""
        # Define and register the context-aware chat agent
        @registry.register_agent
        class ContextAwareChatAgent:
            agent_name = "context_aware_chat"
            description = "Maintains conversation context"
            
            def __init__(self):
                self.context = {}
            
            async def process(self, message, context=None):
                if context:
                    self.context.update(context)
                return {
                    "response": f"Response to: {message}",
                    "context": self.context,
                    "agent": self.agent_name
                }
        
        # Get the agent and test context handling
        chat_agent = registry.get_agent("context_aware_chat")
        
        # First message with context
        response1 = await chat_agent.process("Hello", {"user_id": 123, "name": "Test User"})
        assert response1["context"]["user_id"] == 123
        assert response1["context"]["name"] == "Test User"
        
        # Second message should maintain context
        response2 = await chat_agent.process("How are you?")
        assert response2["context"]["user_id"] == 123
        assert response2["context"]["name"] == "Test User"
    
    def test_agent_registration(self, registry, domain_agents):
        """Test that all agents are properly registered."""
        registered_agents = registry.get_all_agent_names()
        
        # Check all expected agents are registered
        for agent_name in domain_agents.keys():
            assert agent_name in registered_agents
            assert registry.is_registered(agent_name)
    
    def test_duplicate_registration(self, registry):
        """Test that duplicate agent registration raises an error."""
        @registry.register_agent
        class TestAgent:
            agent_name = "test_agent"
            description = "Test agent"
        
        # Try to register again
        with pytest.raises(ValueError):
            @registry.register_agent
            class DuplicateTestAgent:
                agent_name = "test_agent"  # Same name
                description = "Duplicate test agent"
    
    def test_missing_agent_name(self, registry):
        """Test that agents without agent_name raise an error."""
        with pytest.raises(ValueError):
            @registry.register_agent
            class InvalidAgent:
                # Missing agent_name
                description = "Invalid agent"

# This allows running the tests with: python -m pytest tests/test_domain_agents.py -v
