"""Shared types and base classes for agents."""
from typing import Dict, Any, List, Type, TypeVar, Generic, Optional
from dataclasses import dataclass, field
from pydantic import BaseModel

T = TypeVar('T')

class AgentInitializationError(Exception):
    """Exception raised when an agent fails to initialize."""
    pass

@dataclass
class AgentConfig:
    """Configuration for an agent."""
    name: str
    description: str
    model_name: str = "llama3-8b-8192"
    temperature: float = 0.7
    system_prompt: str = ""
    tools: List[Any] = field(default_factory=list)
    max_iterations: int = 10
    verbose: bool = True

class BaseAgent:
    """Base class for all agents."""
    
    def __init__(self, config: AgentConfig):
        """Initialize the agent with configuration."""
        self.config = config
        self.agent = self._create_agent()
    
    def _create_agent(self):
        """Create and return a configured agent."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def process(self, input_text: str, **kwargs):
        """Process input using the agent."""
        raise NotImplementedError("Subclasses must implement this method")
