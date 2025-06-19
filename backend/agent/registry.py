"""Agent registry implementation for managing and initializing agents."""
from typing import Dict, Type, Any, Optional, TypeVar, Generic
from functools import lru_cache
import logging
import os

from .types import AgentInitializationError

logger = logging.getLogger(__name__)

T = TypeVar('T')

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
            raise ValueError(f"Agent with name '{name}' is already registered")
            
        self._agents[name] = agent_class
        logger.debug(f"Registered agent: {name} -> {agent_class.__name__}")
    
    def get_agent(self, name: str, **kwargs) -> T:
        """Get an instance of the named agent with lazy initialization.
        
        Args:
            name: Name of the agent to retrieve
            **kwargs: Arguments to pass to the agent constructor
            
        Returns:
            An instance of the requested agent
            
        Raises:
            KeyError: If no agent is registered with the given name
            AgentInitializationError: If agent initialization fails
        """
        if name not in self._agents:
            raise KeyError(f"No agent registered with name: {name}")
        
        # Create a cache key based on agent name and kwargs
        cache_key = self._get_cache_key(name, kwargs)
        
        # Check if we already have an instance
        if cache_key in self._initialized_agents:
            return self._initialized_agents[cache_key]
            
        # Create new instance if not exists
        try:
            agent_class = self._agents[name]
            instance = agent_class(**kwargs)
            self._initialized_agents[cache_key] = instance
            logger.info(f"Initialized agent: {name} (cache_key: {cache_key})")
            return instance
        except Exception as e:
            logger.error(f"Failed to initialize agent {name}: {str(e)}", exc_info=True)
            raise AgentInitializationError(f"Failed to initialize agent {name}: {str(e)}")
    
    def _get_cache_key(self, name: str, kwargs: dict) -> str:
        """Generate a cache key for the agent instance."""
        # Sort kwargs by key for consistent ordering
        sorted_kwargs = tuple(sorted(kwargs.items()))
        return f"{name}:{hash(sorted_kwargs)}"
    
    def is_registered(self, name: str) -> bool:
        """Check if an agent with the given name is registered.
        
        Args:
            name: Name of the agent to check
            
        Returns:
            bool: True if agent is registered, False otherwise
        """
        return name in self._agents
    
    def get_all_agent_names(self) -> list:
        """Get a list of all registered agent names."""
        return list(self._agents.keys())
    
    def clear_cache(self) -> None:
        """Clear all cached agent instances."""
        self._initialized_agents.clear()
        logger.info("Cleared all cached agent instances")

# Create a singleton instance of AgentRegistry
agent_registry = AgentRegistry()
