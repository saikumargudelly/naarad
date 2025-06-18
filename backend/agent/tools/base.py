"""Base classes for agent tools."""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, TypeVar
from pydantic import BaseModel, Field

T = TypeVar('T', bound='BaseTool')

class ToolParameter(BaseModel):
    """Schema for tool parameters."""
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None

class BaseTool(ABC):
    """Base class for all agent tools.
    
    All tools should inherit from this class and implement the required methods.
    """
    
    name: str = "base_tool"
    description: str = "A base tool that does nothing."
    parameters: Dict[str, ToolParameter] = {}
    
    def __init__(self, **kwargs):
        """Initialize the tool with any required dependencies."""
        self._validate_parameters(kwargs)
        
    def _validate_parameters(self, params: Dict[str, Any]) -> None:
        """Validate that all required parameters are provided."""
        for name, param in self.parameters.items():
            if param.required and name not in params:
                if param.default is None:
                    raise ValueError(f"Missing required parameter: {name}")
                params[name] = param.default
    
    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """Execute the tool with the given parameters.
        
        Args:
            **kwargs: Tool-specific parameters
            
        Returns:
            The result of the tool execution
        """
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the tool to a dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                name: {
                    "type": param.type,
                    "description": param.description,
                    "required": param.required,
                    "default": param.default
                }
                for name, param in self.parameters.items()
            }
        }
    
    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Create a tool instance from a dictionary."""
        return cls(**data)
    
    def __call__(self, **kwargs) -> Any:
        """Allow the tool to be called directly."""
        return self.execute(**kwargs)
