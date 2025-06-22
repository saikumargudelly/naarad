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
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the quality agent with configuration.
        
        Args:
            config: The configuration for the agent. Must be a dictionary.
        """
        # Always define system_prompt first
        routing_info = config.get('routing_info', {}) if isinstance(config, dict) else {}
        intent = routing_info.get('intent', 'unknown')
        confidence = routing_info.get('confidence', None)
        entities = routing_info.get('entities', {})
        routing_str = f"\n[Routing Info]\nIntent: {intent} (confidence: {confidence})\nEntities: {entities}\n" if intent != 'unknown' else ''
        system_prompt = f"""
{routing_str}
You are a quality assurance specialist. Your responsibilities are:

1. Review and improve responses for clarity, conciseness, accuracy, tone, and professionalism.
2. Ensure the improved response fully addresses the user's original query and preserves all relevant context from the conversation.
3. Highlight and explain any major changes made to the original response, especially if you rewrite or significantly alter the content.
4. If the response is already high quality, make only minor adjustments or return it as is.
5. Never introduce factual errors or remove important context.
6. If information is missing, ambiguous, or unclear, suggest clarifications or improvements.
7. Always be transparent about any limitations or uncertainties in the improved response.

Be objective, constructive, and ensure every response is actionable and easy to understand.

If the query is outside your quality domain or the intent confidence is low, escalate to the appropriate agent or ask the user for clarification before proceeding. If you cannot help, say so and suggest the correct agent or next step.
"""
        # Set default values if not provided
        default_config = {
            'name': 'quality',
            'description': 'Specialized in refining and improving responses for quality and clarity.',
            'model_name': settings.REASONING_MODEL,
            'temperature': 0.3,  # Lower temperature for more consistent, focused outputs
            'system_prompt': system_prompt,
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
            model_config = self.agent
            from langchain_groq import ChatGroq
            from langchain.schema import HumanMessage, SystemMessage
            llm = ChatGroq(
                temperature=model_config['temperature'],
                model_name=model_config['model_name'],
                groq_api_key=model_config['api_key']
            )
            prompt = self._prompt_template.format(
                query=original_query,
                response=current_response
            )
            messages = [
                SystemMessage(content="You are a helpful assistant that improves response quality."),
                HumanMessage(content=prompt)
            ]
            result = llm.invoke(messages)
            improved_response = result.content.strip()
            if improved_response.upper() == 'NO_CHANGE':
                return {
                    'output': current_response,
                    'metadata': {
                        'improved': False,
                        'success': True
                    }
                }
            is_improved = improved_response != current_response
            # Contextual follow-up if multiple suggestions/improvements
            suggestions = improved_response.split('\n') if isinstance(improved_response, str) else []
            followup = ''
            if len(suggestions) > 4:
                followup = self._contextual_followup(original_query, suggestions, domain='quality')
                improved_response += f"\n\n{followup}"
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
