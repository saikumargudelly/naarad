"""Agent factory module.

This module provides factory functions for creating and managing agents.
"""

import importlib
import logging
from typing import Dict, List, Optional, Type, TypeVar, Any, Type

# Import only BaseAgent to avoid circular imports
from .agents.base import BaseAgent

logger = logging.getLogger(__name__)

# Type variable for agent classes
AgentT = TypeVar('AgentT', bound=BaseAgent)

def create_agent(agent_type: str, **kwargs) -> BaseAgent:
    """Create an agent of the specified type.
    
    Args:
        agent_type: Type of agent to create (researcher, analyst, responder, quality)
        **kwargs: Additional arguments to pass to the agent constructor
        
    Returns:
        An instance of the requested agent
        
    Raises:
        ValueError: If the agent type is not recognized or creation fails
    """
    # Map agent types to their module paths
    agent_map = {
        'researcher': '.agents.researcher.ResearcherAgent',
        'analyst': '.agents.analyst.AnalystAgent',
        'responder': '.agents.responder.ResponderAgent',
        'quality': '.agents.quality.QualityAgent',
    }
    
    agent_path = agent_map.get(agent_type.lower())
    if not agent_path:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    try:
        module_path, class_name = agent_path.rsplit('.', 1)
        module = importlib.import_module(module_path, package='agent')
        agent_class = getattr(module, class_name)
        # Always pass config as dict
        config = kwargs if kwargs else {}
        return agent_class(config=config)
    except Exception as e:
        logger.error(f"Failed to create {agent_type} agent: {str(e)}")
        raise ValueError(f"Failed to create {agent_type} agent: {str(e)}") from e

def create_default_agents() -> Dict[str, BaseAgent]:
    """Create default instances of all available agents.
    
    Returns:
        Dict mapping agent names to agent instances
    """
    agents = {}
    agent_types = ['responder', 'researcher', 'analyst', 'quality']
    
    for agent_type in agent_types:
        try:
            agents[agent_type] = create_agent(agent_type)  # No AgentConfig, just dict
            logger.info(f"Successfully created {agent_type} agent")
        except Exception as e:
            logger.error(f"Failed to create {agent_type} agent: {str(e)}", exc_info=True)
            continue
    
    if not agents:
        logger.warning("No agents were successfully created")
    else:
        logger.info(f"Successfully created {len(agents)} agents")
    
    return agents

def get_agent_class(agent_name: str) -> Optional[Type[BaseAgent]]:
    """Get an agent class by name.
    
    Args:
        agent_name: Name of the agent class to retrieve (e.g., 'researcher', 'analyst')
        
    Returns:
        The agent class if found, None otherwise
    """
    # Map agent names to their module paths
    agent_map = {
        'researcher': '.agents.researcher.ResearcherAgent',
        'analyst': '.agents.analyst.AnalystAgent',
        'responder': '.agents.responder.ResponderAgent',
        'quality': '.agents.quality.QualityAgent',
    }
    
    agent_path = agent_map.get(agent_name.lower())
    if not agent_path:
        logger.warning(f"Unknown agent name: {agent_name}")
        return None
    
    try:
        # Dynamically import the agent class
        module_path, class_name = agent_path.rsplit('.', 1)
        module = importlib.import_module(module_path, package='agent')
        agent_class = getattr(module, class_name)
        
        # Verify it's a subclass of BaseAgent
        if not issubclass(agent_class, BaseAgent):
            logger.error(f"{class_name} is not a subclass of BaseAgent")
            return None
            
        return agent_class
        
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to import agent class {agent_name}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error getting agent class {agent_name}: {str(e)}")
        return None

class AgentManager:
    """Manages a collection of agents and their lifecycle."""
    
    def __init__(self):
        """Initialize the agent manager with an empty agent registry."""
        self._agents: Dict[str, BaseAgent] = {}
    
    def register_agent(self, name: str, agent: BaseAgent) -> None:
        """Register an agent with the manager.
        
        Args:
            name: Unique name for the agent
            agent: Agent instance to register
            
        Raises:
            ValueError: If an agent with the same name is already registered
        """
        if name in self._agents:
            raise ValueError(f"Agent with name '{name}' is already registered")
        self._agents[name] = agent
    
    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get an agent by name.
        
        Args:
            name: Name of the agent to retrieve
            
        Returns:
            The agent instance if found, None otherwise
        """
        return self._agents.get(name)
    
    def create_agent(self, name: str, agent_type: str, **kwargs) -> BaseAgent:
        """Create and register a new agent.
        
        Args:
            name: Unique name for the agent
            agent_type: Type of agent to create (e.g., 'researcher', 'analyst')
            **kwargs: Additional arguments to pass to the agent constructor
            
        Returns:
            The created agent instance
            
        Raises:
            ValueError: If agent creation fails or if the name is already taken
        """
        if name in self._agents:
            raise ValueError(f"Agent with name '{name}' already exists")
        
        try:
            # Prepare configuration for the agent
            config = {}
            
            # Handle tools parameter properly
            if 'tools' in kwargs:
                config['tools'] = kwargs.pop('tools')
            
            # Handle other configuration parameters
            for key, value in kwargs.items():
                if key in ['description', 'model_name', 'temperature', 'system_prompt', 'max_iterations', 'verbose']:
                    config[key] = value
            
            # Set default values if not provided
            config['name'] = name
            if 'description' not in config:
                config['description'] = f'{agent_type.capitalize()} agent for {agent_type} tasks'
            
            # Create agent using the updated create_agent function
            agent = create_agent(agent_type, **config)
            self.register_agent(name, agent)
            logger.info(f"Successfully created and registered {agent_type} agent: {name}")
            return agent
        except Exception as e:
            logger.error(f"Failed to create {agent_type} agent: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to create {agent_type} agent: {str(e)}") from e
    
    def get_agent_class(self, agent_name: str) -> Optional[Type[BaseAgent]]:
        """Get an agent class by name.
        
        Args:
            agent_name: Name of the agent class to retrieve (e.g., 'researcher', 'analyst')
            
        Returns:
            The agent class if found, None otherwise
        """
        # Map agent names to their module paths
        agent_map = {
            'researcher': '.agents.researcher.ResearcherAgent',
            'analyst': '.agents.analyst.AnalystAgent',
            'responder': '.agents.responder.ResponderAgent',
            'quality': '.agents.quality.QualityAgent',
        }
        
        agent_path = agent_map.get(agent_name.lower())
        if not agent_path:
            logger.warning(f"Unknown agent name: {agent_name}")
            return None
        
        try:
            # Dynamically import the agent class
            module_path, class_name = agent_path.rsplit('.', 1)
            module = importlib.import_module(module_path, package='agent')
            agent_class = getattr(module, class_name)
            
            # Verify it's a subclass of BaseAgent
            if not issubclass(agent_class, BaseAgent):
                logger.error(f"{class_name} is not a subclass of BaseAgent")
                return None
                
            return agent_class
            
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to import agent class {agent_name}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting agent class {agent_name}: {str(e)}")
            return None

    def list_agents(self) -> Dict[str, str]:
        """List all registered agents.
        
        Returns:
            Dict mapping agent names to their types
        """
        return {name: type(agent).__name__ for name, agent in self._agents.items()}
        
    def remove_agent(self, name: str) -> bool:
        """Remove an agent from the manager.
        
        Args:
            name: Name of the agent to remove
            
        Returns:
            True if the agent was removed, False if not found
        """
        if name in self._agents:
            del self._agents[name]
            logger.info(f"Removed agent: {name}")
            return True
        return False
