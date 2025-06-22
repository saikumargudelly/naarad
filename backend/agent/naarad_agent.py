from typing import Dict, Any, List, Optional, Union, Type, TypeVar, Literal, TYPE_CHECKING
import os
import uuid
import json
import logging
import importlib
from datetime import datetime
from fastapi import HTTPException
from pydantic import BaseModel, Field, ConfigDict, validate_call
from pydantic.v1 import BaseModel as BaseModelV1

# LangChain imports
from langchain.agents import AgentExecutor
from langchain_core.tools import BaseTool as LangChainBaseTool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.language_models import BaseChatModel
from langchain.schema.runnable import RunnableSequence
from langchain.chains import LLMChain
from langchain_core.callbacks.manager import CallbackManagerForChainRun
from langchain.callbacks.manager import AsyncCallbackManager
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Custom imports for agent creation
from langchain.agents import ZeroShotAgent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableSequence

# LangChain compatibility
if TYPE_CHECKING:
    from langchain_core.chains import LLMChain
else:
    LLMChain = Any

# Groq import
from groq import Groq

# Local imports
from dotenv import load_dotenv
from llm.config import settings

# Type hints
AgentExecutor = TypeVar('AgentExecutor', bound='AgentExecutor')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Local imports
from dotenv import load_dotenv
from .tools.vision_tool import LLaVAVisionTool
from .tools.brave_search import BraveSearchTool
from .memory.memory_manager import memory_manager
from .monitoring.agent_monitor import agent_monitor
from .types import BaseAgent, AgentConfig, AgentInitializationError

# Lazy imports to avoid circular dependencies
_orchestrator = None
_base_agents = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_orchestrator(base_agents=None):
    global _orchestrator
    if _orchestrator is None:
        from .orchestrator import AgentOrchestrator
        _orchestrator = AgentOrchestrator(base_agents=base_agents or {})
    return _orchestrator

def get_base_agents():
    """Legacy function removed: Use NaaradAgent class for all agent instantiation."""
    raise NotImplementedError("get_base_agents() is deprecated. Use NaaradAgent class instead.")

class ConversationContext(BaseModel):
    """Represents the context for a conversation."""
    
    model_config = ConfigDict(
        extra='forbid',
        validate_assignment=True,
        json_encoders={
            datetime: lambda v: v.isoformat()
        },
        arbitrary_types_allowed=True,  # Allow arbitrary types for LangChain compatibility
        from_attributes=True  # Enable model validation from attributes
    )
    
    user_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the user"
    )
    conversation_id: str = Field(
        default_factory=lambda: f"conv_{uuid.uuid4().hex[:8]}",
        description="Unique identifier for the conversation"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the conversation"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the conversation was created"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the conversation was last updated"
    )
    
    def update_timestamp(self) -> 'ConversationContext':
        """Update the last updated timestamp.
        
        Returns:
            ConversationContext: The updated conversation context
        """
        self.updated_at = datetime.utcnow()
        return self
    
    def model_post_init(self, __context):
        """Post-init processing for Pydantic v2 compatibility."""
        pass

# Import from the new modular agent structure
from agent.agents import AgentConfig
from agent.factory import AgentManager

class NaaradAgent:
    def __init__(self, enable_monitoring: bool = True):
        """Initialize the Naarad AI agent with its tools, agents, and monitoring."""
        # Load environment variables from .env file in the backend directory
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env_path = os.path.join(backend_dir, '.env')
        load_dotenv(env_path)
        
        # Initialize core components
        self.vision_tool = LLaVAVisionTool()
        self.brave_search = BraveSearchTool()
        self.memory_manager = memory_manager
        self.monitor = agent_monitor if enable_monitoring else None
        self._initialized = False
        
        # Import DummyTool
        from .tools import DummyTool
        
        # Create tools for different agent types
        research_tools = [self.brave_search]
        analysis_tools = [DummyTool()]  # Add DummyTool for analyst
        responder_tools = [self.vision_tool]
        quality_tools = [DummyTool()]  # Add DummyTool for quality agent
        
        # Initialize agent manager
        self.agent_manager = AgentManager()
        
        # Create and register agents with their respective tools
        self.base_agents = {}
        
        # Agent configurations with their tools
        agent_configs = [
            ('responder', {'tools': responder_tools, 'model_name': settings.CHAT_MODEL}),
            ('researcher', {'tools': research_tools, 'model_name': settings.REASONING_MODEL}),
            ('analyst', {'tools': analysis_tools, 'model_name': settings.REASONING_MODEL}),
            ('quality', {'tools': quality_tools, 'model_name': settings.REASONING_MODEL})
        ]
        for agent_type, config in agent_configs:
            self.base_agents[agent_type] = self.agent_manager.get_agent_class(agent_type)(config)
        
        logger.info(f"Initialized {len(self.base_agents)} base agents: {list(self.base_agents.keys())}")
        
        # Store base_agents for lazy orchestrator initialization
        self._base_agents = self.base_agents
        self._orchestrator = None  # Lazy initialization
        
        # System personality and constraints
        self.personality = {
            "name": "Naarad",
            "version": "2.0.0",
            "traits": [
                "Curious and playful 😺",
                "Helpful and informative",
                "Concise but thorough",
                "Honest about limitations",
                "Respectful and considerate"
            ],
            "capabilities": [
                "Multi-agent collaboration",
                "Context-aware responses",
                "Image understanding",
                "Task management",
                "Creative writing",
                "In-depth analysis"
            ]
        }
    
    @property
    def orchestrator(self):
        """Lazy-load the orchestrator when needed."""
        if self._orchestrator is None:
            from .orchestrator import AgentOrchestrator
            logger.info("Creating AgentOrchestrator singleton instance...")
            self._orchestrator = AgentOrchestrator(base_agents=self._base_agents)
            if hasattr(self._orchestrator, 'agent_manager'):
                self._orchestrator.agent_manager = self.agent_manager
        return self._orchestrator
    
    async def ensure_initialized(self):
        """Ensure the agent is properly initialized."""
        if not self._initialized:
            # Initialize memory manager
            if hasattr(self.memory_manager, 'ensure_initialized'):
                await self.memory_manager.ensure_initialized()
            self._initialized = True

    async def _process_images(self, images: List[str], context: ConversationContext) -> None:
        """Process images and add them to the context.
        
        Args:
            images: List of image URLs or base64 strings
            context: The conversation context to update
        """
        if not images:
            return
            
        try:
            # Add image information to context metadata
            if 'images' not in context.metadata:
                context.metadata['images'] = []
                
            for i, image in enumerate(images):
                image_info = {
                    'index': i,
                    'type': 'url' if image.startswith('http') else 'base64',
                    'processed': False
                }
                
                # If it's a URL, store it directly
                if image.startswith('http'):
                    image_info['url'] = image
                    image_info['processed'] = True
                else:
                    # For base64, we'll need to handle it differently
                    # For now, just store the fact that we have image data
                    image_info['has_data'] = True
                    image_info['data_length'] = len(image)
                
                context.metadata['images'].append(image_info)
                
            # Mark that we have images in this conversation
            context.metadata['has_images'] = True
            context.metadata['image_count'] = len(images)
            
            logger.info(f"Processed {len(images)} images for conversation {context.conversation_id}")
            
        except Exception as e:
            logger.error(f"Error processing images: {str(e)}", exc_info=True)
            # Don't fail the entire request if image processing fails
            pass

    async def process_message(
        self, 
        message: str, 
        conversation_id: str = None, 
        user_id: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a user message and return a structured response.
        
        Args:
            message: The user's message
            conversation_id: Optional conversation ID for tracking context
            user_id: Optional user ID for personalization
            **kwargs: Additional context (e.g., images, metadata)
            
        Returns:
            Dict containing response and metadata
        """
        try:
            # Ensure agent is initialized
            await self.ensure_initialized()
            
            # Get or create conversation context
            context = await self._get_or_create_context(conversation_id, user_id)
            if context is None:
                raise ValueError("Failed to create or retrieve conversation context")
                
            # Update the timestamp
            context = context.update_timestamp()
            
            # Process any images if provided
            if 'images' in kwargs:
                await self._process_images(kwargs['images'], context)
            
            # Process the message with the orchestrator
            response = await self.orchestrator.process_query(
                user_input=message,
                context={
                    'conversation_id': context.conversation_id,
                    'user_id': context.user_id,
                    'metadata': context.metadata,
                    **kwargs
                }
            )
            
            # Ensure response is a string
            response_text = response.get('output', '')
            if not isinstance(response_text, str):
                response_text = str(response_text)
            
            # Update conversation in memory
            await self.memory_manager.save_conversation(
                conversation_id=context.conversation_id,
                user_id=context.user_id,
                messages=[{
                    'role': 'user',
                    'content': message,
                    'timestamp': datetime.utcnow().isoformat()
                }, {
                    'role': 'assistant',
                    'content': response_text,
                    'timestamp': datetime.utcnow().isoformat()
                }],
                metadata=context.metadata
            )
            
            return {
                'success': True,
                'response': response,
                'conversation_id': context.conversation_id,
                'metadata': {
                    'timestamp': datetime.utcnow().isoformat(),
                    'agent_used': response.get('agent_used', 'unknown')
                }
            }
            
        except Exception as e:
            error_id = str(uuid.uuid4())
            logger.error(f"Error processing message (ID: {error_id}): {str(e)}", exc_info=True)
            
            if self.monitor:
                self.monitor.record_processing_error(
                    agent_name='naarad_agent',
                    error_type=type(e).__name__,
                    error_message=str(e),
                    conversation_id=context.conversation_id if 'context' in locals() else None
                )
            
            return {
                'success': False,
                'error': {
                    'id': error_id,
                    'type': type(e).__name__,
                    'message': 'An error occurred while processing your request.',
                    'details': str(e) if str(e) else 'No additional details available'
                },
                'conversation_id': context.conversation_id if 'context' in locals() else None,
                'user_id': user_id
            }
    
    async def _get_or_create_context(
        self, 
        conversation_id: Optional[str], 
        user_id: Optional[str]
    ) -> ConversationContext:
        """Retrieve or create a conversation context."""
        if conversation_id:
            # Try to load existing conversation
            try:
                conversation = await self.memory_manager.get_conversation(conversation_id, user_id or "anonymous")
                if conversation:
                    # Ensure we have proper timestamps
                    metadata = conversation.get('metadata', {})
                    if 'created_at' not in metadata:
                        metadata['created_at'] = datetime.utcnow().isoformat()
                    if 'updated_at' not in metadata:
                        metadata['updated_at'] = datetime.utcnow().isoformat()
                        
                    return ConversationContext(
                        user_id=user_id or conversation.get('user_id', str(uuid.uuid4())),
                        conversation_id=conversation_id,
                        metadata=metadata,
                        created_at=datetime.fromisoformat(metadata['created_at']),
                        updated_at=datetime.fromisoformat(metadata['updated_at'])
                    )
            except Exception as e:
                logger.warning(f"Error loading conversation {conversation_id}: {str(e)}")
        
        # Create a new conversation context
        now = datetime.utcnow()
        return ConversationContext(
            user_id=user_id or str(uuid.uuid4()),
            conversation_id=conversation_id or f"conv_{uuid.uuid4().hex[:8]}",
            metadata={
                'created_at': now.isoformat(),
                'updated_at': now.isoformat(),
                'message_count': 0
            },
            created_at=now,
            updated_at=now
        )
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the agent and its capabilities."""
        return {
            **self.personality,
            'agents': [
                {"name": name, "description": agent.description}
                for name, agent in self.base_agents.items()
            ],
            'memory': {
                'enabled': True,
                'type': self.memory_manager.__class__.__name__
            },
            'monitoring': {
                'enabled': self.monitor is not None,
                'metrics_port': getattr(self.monitor, 'metrics_port', None) if self.monitor else None
            }
        }
    
    async def get_conversation_history(
        self, 
        conversation_id: str, 
        user_id: str = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Retrieve conversation history with pagination.
        
        Args:
            conversation_id: The ID of the conversation to retrieve
            user_id: Optional user ID for authorization
            limit: Maximum number of messages to return
            offset: Number of messages to skip
            
        Returns:
            Dict containing conversation history and metadata
        """
        try:
            # Ensure we're initialized
            await self.ensure_initialized()
            
            # Get conversation from memory manager
            conversation = await self.memory_manager.get_conversation(conversation_id, user_id or "anonymous")
            if not conversation:
                raise HTTPException(status_code=404, detail="Conversation not found")
                
            # Get messages and apply pagination
            messages = conversation.get('messages', [])
            total_messages = len(messages)
            
            # Apply pagination
            paginated_messages = messages[offset:offset + limit]
            
            # Ensure timestamps are present
            metadata = conversation.get('metadata', {})
            if 'created_at' not in metadata:
                metadata['created_at'] = datetime.utcnow().isoformat()
            if 'updated_at' not in metadata:
                metadata['updated_at'] = datetime.utcnow().isoformat()
            
            return {
                'success': True,
                'conversation_id': conversation_id,
                'user_id': user_id,
                'total_messages': total_messages,
                'returned_messages': len(paginated_messages),
                'offset': offset,
                'limit': limit,
                'messages': paginated_messages,
                'metadata': {
                    'created_at': metadata['created_at'],
                    'updated_at': metadata['updated_at']
                }
            }
            
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error retrieving conversation: {str(e)}"
            )
    
    async def get_agent_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for all agents.
        
        Returns:
            Dict containing agent metrics and status information
        """
        if not self.monitor:
            return {
                'success': False,
                'error': 'Monitoring is not enabled for this agent',
                'metrics': {},
                'timestamp': datetime.utcnow().isoformat()
            }
            
        try:
            # Ensure we're initialized
            await self.ensure_initialized()
            
            # Get metrics from monitor
            metrics = {}
            if hasattr(self.monitor, 'get_metrics'):
                if hasattr(self.monitor.get_metrics, '__await__'):
                    metrics = await self.monitor.get_metrics()
                else:
                    metrics = self.monitor.get_metrics()
            
            # Add basic agent info if available
            if not metrics.get('agents'):
                metrics['agents'] = {
                    name: {
                        'status': 'active' if agent.is_initialized() else 'inactive',
                        'last_used': getattr(agent, 'last_used', 'never')
                    }
                    for name, agent in self.base_agents.items()
                }
            
            return {
                'success': True,
                'metrics': metrics,
                'timestamp': datetime.utcnow().isoformat(),
                'agent_count': len(self.base_agents)
            }
            
        except Exception as e:
            logger.error(f"Error retrieving agent metrics: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'metrics': {},
                'timestamp': datetime.utcnow().isoformat()
            }

    async def process_with_streaming(
        self,
        message: str,
        conversation_id: str = None,
        user_id: str = None,
        message_type: str = "text",
        **kwargs
    ):
        """Process a message and stream the response back."""
        try:
            context = await self._get_or_create_context(conversation_id, user_id)
            
            # Get existing conversation
            conversation = await self.memory_manager.get_conversation(
                conversation_id=context.conversation_id,
                user_id=context.user_id
            )
            messages = conversation.get('messages', []) if conversation else []

            # Add user message to history
            messages.append({
                "role": "user",
                "content": message,
                "timestamp": datetime.utcnow().isoformat()
            })

            # Save updated conversation
            await self.memory_manager.save_conversation(
                conversation_id=context.conversation_id,
                user_id=context.user_id,
                messages=messages,
                metadata=context.metadata
            )

            full_response = ""
            async for chunk in self.orchestrator.process_message_stream(
                message, 
                context.conversation_id,
                context.user_id,
                message_type,
                **kwargs
            ):
                if isinstance(chunk, dict) and chunk.get("type") == "stream_chunk":
                    full_response += chunk.get("chunk", "")
                yield chunk

            # Add final AI message to memory
            messages.append({
                "role": "assistant",
                "content": full_response,
                "timestamp": datetime.utcnow().isoformat()
            })
            await self.memory_manager.save_conversation(
                conversation_id=context.conversation_id,
                user_id=context.user_id,
                messages=messages,
                metadata=context.metadata
            )

        except Exception as e:
            logger.error(f"Error in process_with_streaming: {e}", exc_info=True)
            yield {
                "type": "error",
                "error": f"An unexpected error occurred: {str(e)}"
            }
