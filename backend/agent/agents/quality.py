"""Quality agent implementation.

This module contains the QualityAgent class which is specialized in refining
and improving responses for quality and clarity.
"""

import logging
from typing import Any, Dict, List, Optional, Union

# LangChain imports
from langchain_core.tools import BaseTool

# Local imports
from .base import BaseAgent, AgentConfig
from llm.config import settings

logger = logging.getLogger(__name__)

class QualityAgent(BaseAgent):
    """Agent specialized in refining and improving responses for quality and clarity."""
    
    def __init__(self, config: Union[AgentConfig, Dict[str, Any]]):
        """Initialize the quality agent with configuration.
        
        Args:
            config: The configuration for the agent. Can be a dictionary or AgentConfig instance.
                   If None, default values will be used.
        """
        if not isinstance(config, (AgentConfig, dict)):
            raise ValueError("config must be an AgentConfig instance or a dictionary")
            
        if isinstance(config, dict):
            # Set default values if not provided
            default_config = {
                'name': 'quality',
                'description': 'Specialized in refining and improving responses for quality and clarity.',
                'model_name': settings.REASONING_MODEL,
                'temperature': 0.3,  # Lower temperature for more consistent, focused outputs
                'system_prompt': """You are a quality assurance specialist. Your job is to review and improve responses 
                for clarity, conciseness, accuracy, and tone. Ensure the response is well-structured, free of errors, 
                and effectively addresses the user's query. Pay special attention to:
                
                1. Grammar and spelling
                2. Logical flow and coherence
                3. Accuracy of information
                4. Tone and professionalism
                5. Completeness of the response
                
                If the response needs significant improvement, rewrite it completely while preserving the core meaning.
                If the response is already high quality, make only minor adjustments or return it as is.""",
                'max_iterations': 2  # Fewer iterations for quality checking
            }
            # Update with any provided config values
            default_config.update(config)
            config = default_config
            
        super().__init__(config)
    
    def _create_agent(self):
        """Create and configure the LangChain agent for quality checking.
        
        Returns:
            Configured agent executor
            
        Raises:
            RuntimeError: If agent creation fails
        """
        try:
            # Import here to avoid circular imports
            from langchain_groq import ChatGroq
            from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
            
            # Store the prompt template as a string instead of creating it here
            self._prompt_template = """You are a quality assurance specialist. Your job is to review and improve responses 
            for clarity, conciseness, accuracy, and tone. Ensure the response is well-structured, free of errors, 
            and effectively addresses the user's query.
            
            Please review and improve the following response. Focus on clarity, accuracy, and professionalism.
            
            Original Query: {query}
            
            Current Response:
            {response}
            
            Improved Response (or type 'NO_CHANGE' if no changes are needed):"""
            
            # Return a simple dict that can be pickled
            return {
                'model_name': self.config.model_name,
                'temperature': self.config.temperature,
                'api_key': os.getenv('GROQ_API_KEY')
            }
            
        except Exception as e:
            logger.error(f"Failed to create quality agent: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to initialize quality agent: {str(e)}") from e
    
    def _process_impl(self, input_text: str, **kwargs) -> Dict[str, Any]:
        """Process and improve a response using the quality agent.
        
        Args:
            input_text: The response text to be checked/improved
            **kwargs: Additional arguments including:
                - query: The original user query that prompted the response
                - strict: If True, enforce stricter quality checks
                
        Returns:
            Dict containing the improved response and metadata
        """
        try:
            # Get the original query and response
            original_query = kwargs.pop('query', 'Unknown query')
            current_response = input_text.strip()
            
            if not current_response:
                return {
                    'output': '',
                    'metadata': {
                        'improved': False,
                        'error': 'Empty response provided for quality check',
                        'success': False
                    }
                }
            
            # Get the model configuration
            model_config = self.agent
            
            # Initialize the model here to avoid pickling issues
            from langchain_groq import ChatGroq
            from langchain.schema import HumanMessage, SystemMessage
            
            llm = ChatGroq(
                temperature=model_config['temperature'],
                model_name=model_config['model_name'],
                groq_api_key=model_config['api_key']
            )
            
            # Format the prompt
            prompt = self._prompt_template.format(
                query=original_query,
                response=current_response
            )
            
            # Get the improved response
            messages = [
                SystemMessage(content="You are a helpful assistant that improves response quality."),
                HumanMessage(content=prompt)
            ]
            result = llm.invoke(messages)
            
            # Process the result
            improved_response = result.content.strip()
            
            # Check if no changes were needed
            if improved_response.upper() == 'NO_CHANGE':
                return {
                    'output': current_response,
                    'metadata': {
                        'improved': False,
                        'success': True
                    }
                }
            
            # Check if the response was actually improved
            is_improved = improved_response != current_response
            
            return {
                'output': improved_response,
                'metadata': {
                    'improved': is_improved,
                    'success': True
                }
            }
            
        except Exception as e:
            logger.error(f"Error in quality check: {str(e)}", exc_info=True)
            return {
                'output': current_response,
                'metadata': {
                    'improved': False,
                    'error': str(e),
                    'success': False
                }
            }
    
    def check_quality(self, query: str, response: str, **kwargs) -> Dict[str, Any]:
        """Check and improve the quality of a response.
        
        This is a convenience method that wraps _process_impl with named parameters.
        
        Args:
            query: The original user query
            response: The response to check/improve
            **kwargs: Additional arguments for process_impl
            
        Returns:
            Dict containing the improved response and metadata
        """
        return self._process_impl(response, query=query, **kwargs)
