"""Agent registry implementation for managing and initializing agents."""
from typing import Dict, Type, Any, Optional, TypeVar, Generic
from functools import lru_cache
import logging
import os

logger = logging.getLogger(__name__)

T = TypeVar('T')

class AgentInitializationError(Exception):
    """Exception raised when an agent fails to initialize."""
    pass

class AgentRegistry(Generic[T]):
    """Generic registry for managing agent instances with lazy initialization."""
    
    _instance = None
    _agents: Dict[str, Type[T]] = {}
    _initialized_agents: Dict[str, T] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AgentRegistry, cls).__new__(cls)
        return cls._instance
    
    def register(self, name: str, agent_class: Type[T]) -> None:
        """Register an agent class with the given name.
        
        Args:
            name: Unique identifier for the agent
            agent_class: The agent class to register
            
        Raises:
            ValueError: If the agent class is invalid or name is already registered
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Agent name must be a non-empty string")
            
        if not (isinstance(agent_class, type) and hasattr(agent_class, '__call__')):
            raise ValueError(f"Invalid agent class: {agent_class}")
            
        if name in self._agents:
            logger.warning(f"Agent with name '{name}' is already registered. Overwriting.")
            
        self._agents[name] = agent_class
        logger.debug(f"Registered agent: {name} -> {agent_class.__name__}")
    
    def get_agent(self, name: str, **kwargs) -> T:
        """Get an instance of the named agent with lazy initialization.
        
        Args:
            name: Name of the agent to retrieve
            **kwargs: Arguments to pass to the agent's __init__ method
            
        Returns:
            An instance of the requested agent
            
        Raises:
            KeyError: If no agent is registered with the given name
            AgentInitializationError: If agent initialization fails
        """
        if name not in self._agents:
            raise KeyError(f"No agent registered with name: {name}")
            
        # Return existing instance if available
        if name in self._initialized_agents:
            return self._initialized_agents[name]
            
        # Create and cache a new instance
        try:
            agent_class = self._agents[name]
            logger.debug(f"Initializing agent: {name} ({agent_class.__name__})")
            agent = agent_class(**kwargs)
            self._initialized_agents[name] = agent
            return agent
        except Exception as e:
            error_msg = f"Failed to initialize agent '{name}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise AgentInitializationError(error_msg) from e
    
    def is_registered(self, name: str) -> bool:
        """Check if an agent with the given name is registered."""
        return name in self._agents
    
    def list_agents(self) -> Dict[str, Type[T]]:
        """Get a dictionary of all registered agent names and their classes."""
        return self._agents.copy()
    
    def clear(self) -> None:
        """Clear all registered agents and instances."""
        self._agents.clear()
        self._initialized_agents.clear()
    
    def get_all_agent_names(self) -> list[str]:
        """Get a list of all registered agent names.
        
        Returns:
            List of agent names as strings
        """
        return list(self._agents.keys())

# Create a global instance of the registry
agent_registry = AgentRegistry()
