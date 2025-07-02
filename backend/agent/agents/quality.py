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
from agent.memory.memory_manager import MemoryManager

logger = logging.getLogger(__name__)

class QualityAgent(BaseAgent):
    """Agent specialized in refining and improving responses for quality and clarity.
    Modular, stateless, and uses injected memory manager for context/state.
    """
    
    def __init__(self, config: Dict[str, Any], memory_manager: MemoryManager = None):
        """Initialize the quality agent with configuration.
        
        Args:
            config: The configuration for the agent. Must be a dictionary.
            memory_manager: The memory manager for context/state injection
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
        self.memory_manager = memory_manager
        logger.info(f"QualityAgent initialized with memory_manager: {bool(memory_manager)}")
    
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
    
    async def process(self, input_text: str, context: Dict[str, Any] = None, conversation_id: str = None, user_id: str = None, conversation_memory=None, **kwargs) -> Dict[str, Any]:
        logger.info(f"QualityAgent.process called | input_text: {input_text} | conversation_id: {conversation_id} | user_id: {user_id}")
        try:
            chat_history = kwargs.get('chat_history', '')
            topic = None
            intent = None
            last_user_message = None
            if conversation_memory:
                topic = conversation_memory.topics[-1] if conversation_memory.topics else None
                intent = conversation_memory.intents[-1] if conversation_memory.intents else None
                for msg in reversed(conversation_memory.messages):
                    if msg['role'] == 'user':
                        last_user_message = msg['content']
                        break
            # Compose a context-aware prompt
            context_snippets = "\n".join([
                f"{m['role'].capitalize()}: {m['content']}" for m in conversation_memory.messages[-6:]
            ]) if conversation_memory else ""
            system_prompt = (
                "You are a quality assurance assistant. Use the conversation context, topic, and intent to answer the user's question as accurately and helpfully as possible. "
                "If the user is following up, use the previous context to disambiguate."
            )
            from langchain_groq import ChatGroq
            from langchain_core.messages import SystemMessage, HumanMessage
            from llm.config import settings
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Conversation context:\n{context_snippets}\n\nTopic: {topic}\nIntent: {intent}\n\nUser question: {input_text}")
            ]
            llm = ChatGroq(
                temperature=0.2,
                model_name=settings.REASONING_MODEL,
                groq_api_key=os.getenv('GROQ_API_KEY')
            )
            result = await llm.ainvoke(messages)
            return {"output": result.content.strip(), "metadata": {"success": True, "topic": topic, "intent": intent}}
        except Exception as e:
            logger.error(f"Async error in quality process: {str(e)}", exc_info=True)
            return {
                'output': f"I encountered an error while processing your quality request: {str(e)}",
                'metadata': {
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
