"""Base agent implementation and configuration.

This module contains the base agent class and configuration that all other agents inherit from.
"""

from typing import Any, Dict, List, Optional, Type, TypeVar, Union, Literal
import logging
import os
from pathlib import Path
from enum import Enum

# LangChain imports
from langchain.agents import AgentExecutor
from langchain.agents.agent import AgentOutputParser
from langchain.agents.react.output_parser import ReActOutputParser
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseChatModel, BaseLanguageModel
from langchain_core.prompts import BaseChatPromptTemplate
from langchain_core.agents import AgentAction, AgentFinish
from pydantic import BaseModel, Field, ConfigDict

# Local imports
from llm.config import settings

# Configure logging
logger = logging.getLogger(__name__)

# Type variable for agent classes
AgentT = TypeVar('AgentT', bound='BaseAgent')

class AgentConfig(BaseModel):
    """Configuration for an agent.
    
    Models are configured to use Groq with the following mapping:
    - Language Reasoning: mixtral-8x7b-instruct-v0.1
    - Chat: mixtral-8x7b-instruct-v0.1
    - Default: mixtral-8x7b-instruct-v0.1
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra='forbid',
        validate_assignment=True
    )
    
    name: str = Field(..., description="Unique identifier for the agent")
    description: str = Field(..., description="Brief description of the agent's purpose")
    model_name: str = Field("mixtral-8x7b-instruct-v0.1", description="Name of the model to use")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature for model generation")
    system_prompt: str = Field("", description="Initial system prompt for the agent")
    tools: List[BaseTool] = Field(default_factory=list, description="List of tools available to the agent")
    max_iterations: int = Field(10, ge=1, le=100, description="Maximum number of iterations for the agent to run")
    verbose: bool = Field(True, description="Enable verbose logging")
    handle_parsing_errors: bool = Field(True, description="Whether to handle parsing errors gracefully")
    early_stopping_method: str = Field("force", description="Method to use for early stopping")
    
    def model_post_init(self, __context):
        """Post-init processing for Pydantic v2 compatibility."""
        # Convert tools to list if they're not already
        if not isinstance(self.tools, list):
            self.tools = [self.tools] if self.tools else []
            
        # Ensure all tools are instances of BaseTool
        self.tools = [
            tool if isinstance(tool, BaseTool) else BaseTool(**tool) 
            if isinstance(tool, dict) else tool
            for tool in self.tools
        ]

class BaseAgent:
    """Base class for all agents.
    
    This class provides a base implementation for all agents in the system.
    It handles the creation and management of the underlying LangChain agent.
    """
    
    def __init__(self, config: Union[AgentConfig, Dict[str, Any]]):
        """Initialize the agent with configuration.
        
        Args:
            config: The configuration for the agent. Can be a dictionary or AgentConfig instance.
            
        Raises:
            ValueError: If the configuration is invalid
        """
        try:
            if not isinstance(config, AgentConfig):
                config = AgentConfig(**config) if isinstance(config, dict) else AgentConfig.model_validate(config)
                
            self.config = config
            self._agent = None  # Will be initialized on first use
            logger.info(f"Initialized {self.__class__.__name__} with model: {config.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize {self.__class__.__name__}: {str(e)}")
            raise ValueError(f"Invalid agent configuration: {str(e)}") from e
    
    @property
    def agent(self):
        """Lazy-load the agent when needed to avoid pickling issues."""
        if self._agent is None:
            self._agent = self._create_agent()
        return self._agent
    
    @property
    def name(self) -> str:
        """Get the agent's name.
        
        Returns:
            str: The agent's name
        """
        return self.config.name
    
    def _create_agent(self) -> AgentExecutor:
        """Create and configure the underlying LangChain agent.
        
        This method should be implemented by subclasses to create the specific
        type of agent they require.
        
        Returns:
            AgentExecutor: The configured agent executor
            
        Raises:
            NotImplementedError: If the method is not implemented by a subclass
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    async def aprocess(self, input_text: str, **kwargs) -> Dict[str, Any]:
        """Process input asynchronously.
        
        Args:
            input_text: The input text to process
            **kwargs: Additional keyword arguments
            
        Returns:
            Dict containing the response and metadata
        """
        try:
            # Get the agent executor
            agent = self.agent
            
            # If the agent has an ainvoke method, use it
            if hasattr(agent, 'ainvoke') and callable(agent.ainvoke):
                result = await agent.ainvoke({"input": input_text, **kwargs})
            else:
                # Fall back to sync processing in a thread
                import asyncio
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None, agent.invoke, {"input": input_text, **kwargs}
                )
                
            # Standardize the output key. LLMChain uses 'text', AgentExecutor uses 'output'.
            if "output" not in result:
                if "text" in result:
                    result["output"] = result.pop("text")
                elif "response" in result:
                    result["output"] = result.pop("response")
                
            result.setdefault("success", True)
            return result
            
        except Exception as e:
            error_msg = f"Error in async processing: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "output": error_msg,
                "error": str(e),
                "success": False
            }
    
    async def process(self, input_text: str, **kwargs) -> Dict[str, Any]:
        """Process input text with the agent.
        
        Args:
            input_text: The input text to process
            **kwargs: Additional arguments to pass to the agent
            
        Returns:
            Dict containing the agent's response and metadata
            
        Raises:
            RuntimeError: If the agent fails to process the input
        """
        try:
            if not input_text or not isinstance(input_text, str):
                raise ValueError("Input text must be a non-empty string")
                
            # Delegate to the actual agent implementation
            result = await self.aprocess(input_text, **kwargs)
            
            # The result from aprocess is already a dictionary with an 'output' key.
            # We just need to ensure the metadata is included correctly.
            
            metadata = {
                'agent': self.name,
                'model': self.config.model_name,
                'success': result.get('success', True),
                'simple_query': kwargs.get('simple_query', False),
                'tools_used': [] # This can be populated later if tools are used
            }
            
            # Merge with any metadata from the agent run
            if 'metadata' in result and isinstance(result['metadata'], dict):
                metadata.update(result['metadata'])

            return {
                'output': result.get('output', 'No response generated.'),
                'metadata': metadata
            }

        except Exception as e:
            logger.error(f"Error in {self.__class__.__name__}.process: {str(e)}", exc_info=True)
            return {
                'output': f"An error occurred: {str(e)}",
                'metadata': {
                    'agent': self.name,
                    'model': self.config.model_name,
                    'success': False,
                    'error': str(e)
                }
            }
    
    def _process_impl(self, input_text: str, **kwargs) -> Any:
        """Implementation of the process method to be overridden by subclasses.
        
        Args:
            input_text: The input text to process
            **kwargs: Additional arguments
            
        Returns:
            The processed result
            
        Raises:
            Exception: If processing fails
        """
        raise NotImplementedError("Subclasses must implement _process_impl")
