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
    """Configuration for an agent (Pydantic v2 only, no custom post-init)."""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra='forbid',
        validate_assignment=True
    )
    name: str = Field(..., description="Unique identifier for the agent")
    description: str = Field(..., description="Brief description of the agent's purpose")
    model_name: str = Field("llama3-8b-8192", description="Name of the model to use")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature for model generation")
    system_prompt: str = Field("", description="Initial system prompt for the agent")
    tools: List[BaseTool] = Field(default_factory=list, description="List of tools available to the agent")
    max_iterations: int = Field(10, ge=1, le=100, description="Maximum number of iterations for the agent to run")
    verbose: bool = Field(True, description="Enable verbose logging")
    handle_parsing_errors: bool = Field(True, description="Whether to handle parsing errors gracefully")
    early_stopping_method: str = Field("force", description="Method to use for early stopping")

class BaseAgent:
    """Base class for all agents.
    Modular, supports memory_manager injection, and logging for all child agents.
    """
    def __init__(self, config: Dict[str, Any], memory_manager: Any = None):
        """Initialize the agent with configuration and optional memory manager.
        Args:
            config: The configuration for the agent. Must be a dictionary.
            memory_manager: Optional memory manager for context/state sharing.
        Raises:
            ValueError: If the configuration is invalid
        """
        try:
            if not isinstance(config, dict):
                raise ValueError("Agent config must be a dict (Pydantic v2 compatible)")
            self.config = AgentConfig(**config)
            self._agent = None  # Will be initialized on first use
            self.memory_manager = memory_manager
            logger.info(f"Initialized {self.__class__.__name__} with model: {self.config.model_name}, memory_manager: {bool(memory_manager)}")
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
            
            # Ensure intermediate_steps is always a list
            if 'intermediate_steps' not in kwargs or kwargs['intermediate_steps'] is None or isinstance(kwargs['intermediate_steps'], str):
                kwargs['intermediate_steps'] = []
            
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
    
    async def process(self, input_text: str, context: Dict[str, Any] = None, conversation_id: str = None, user_id: str = None, conversation_memory=None, **kwargs) -> Dict[str, Any]:
        """Base process method: agents can use conversation_memory for context-aware responses."""
        # Example: Access previous messages, topics, intents, entities
        if conversation_memory:
            last_user_message = None
            for msg in reversed(conversation_memory.messages):
                if msg['role'] == 'user':
                    last_user_message = msg['content']
                    break
            topics = conversation_memory.topics
            intents = conversation_memory.intents
            entities = conversation_memory.entities
            # Use these for context-aware logic, e.g.:
            # if 'cricket' in topics: ...
        # ... rest of agent logic ...
        return {'output': f"[DEBUG] Last user message: {last_user_message}, Topics: {topics}, Intents: {intents}, Entities: {entities}", 'metadata': {}}
    
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

    def _contextual_followup(self, user_query: str, results: list, domain: str) -> str:
        """Generate a contextual follow-up question based on the query, results, and domain."""
        q = user_query.lower()
        titles = ' '.join([str(r).lower() for r in results[:5]])
        followup = []
        if domain == "sports":
            if any(x in q or x in titles for x in ['women', "women's"]):
                followup.append("Do you want Women's or Men's match?")
            elif 'men' in q or "men's" in titles:
                followup.append("Do you want Men's or Women's match?")
            if 'highlight' in q or 'highlight' in titles:
                followup.append("Are you looking for highlights or live scorecard?")
            if 'score' in q or 'live' in q or 'scorecard' in titles:
                followup.append("Do you want the live score, full scorecard, or match summary?")
            if not followup:
                followup.append("Can you specify if you want news, scores, or something else? Or would you like to see more results?")
        elif domain == "finance":
            if "stock" in q or "market" in q:
                followup.append("Are you interested in a specific stock, index, or market news?")
            if not followup:
                followup.append("Would you like more details, charts, or recent news?")
        elif domain == "news":
            if 'local' in q or 'city' in q or 'state' in q:
                followup.append("Are you looking for local, national, or international news?")
            if 'breaking' in q or 'alert' in q:
                followup.append("Do you want breaking news, top headlines, or a specific topic?")
            if not followup:
                followup.append("Would you like more headlines, in-depth articles, or news on a specific topic?")
        elif domain == "weather":
            if 'today' in q or 'now' in q:
                followup.append("Do you want the current weather, forecast, or severe weather alerts?")
            if not followup:
                followup.append("Would you like the forecast for today, this week, or a specific location?")
        else:
            followup.append("Can you clarify what specific information you want, or would you like to see more results?")
        return '\n'.join(followup)
