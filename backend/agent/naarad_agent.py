from typing import Dict, Any, List, Optional, Union
import os
import uuid
import json
import logging
from datetime import datetime
from fastapi import HTTPException
from pydantic import BaseModel, Field

from dotenv import load_dotenv
from langchain.agents import AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from .tools.vision_tool import LLaVAVisionTool
from .tools.brave_search import BraveSearchTool
from .orchestrator import AgentOrchestrator
from .agents import (
    create_base_agents,
    ResearcherAgent,
    AnalystAgent,
    ResponderAgent,
    QualityAgent
)
from .memory.memory_manager import memory_manager
from .monitoring.agent_monitor import agent_monitor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConversationContext(BaseModel):
    """Represents the context for a conversation."""
    user_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str = Field(default_factory=lambda: f"conv_{uuid.uuid4().hex[:8]}")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    def update_timestamp(self):
        """Update the last updated timestamp."""
        self.updated_at = datetime.utcnow()
        return self

class NaaradAgent:
    def __init__(self, enable_monitoring: bool = True):
        """Initialize the Naarad AI agent with its tools, agents, and monitoring."""
        load_dotenv()
        
        # Initialize core components
        self.vision_tool = LLaVAVisionTool()
        self.brave_search = BraveSearchTool()
        self.memory_manager = memory_manager
        self.monitor = agent_monitor if enable_monitoring else None
        
        # Create tools for different agent types
        research_tools = [self.brave_search]
        analysis_tools = []
        responder_tools = [self.vision_tool]
        quality_tools = []
        
        # Create base agents with appropriate tools
        self.base_agents = {
            'researcher': ResearcherAgent(tools=research_tools),
            'analyst': AnalystAgent(tools=analysis_tools),
            'responder': ResponderAgent(tools=responder_tools),
            'quality': QualityAgent(tools=quality_tools)
        }
        
        # Initialize orchestrator with the agents
        self.orchestrator = AgentOrchestrator(self.base_agents)
        
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
        start_time = datetime.utcnow()
        
        # Initialize or retrieve conversation context
        context = await self._get_or_create_context(conversation_id, user_id)
        
        try:
            # Track the request
            with agent_monitor.track_request('naarad_agent'):
                # Process any images if provided
                images = kwargs.get('images', [])
                context = await self._process_images(images, context)
                
                # Prepare the context for processing
                processing_context = {
                    **context.metadata,
                    'images_processed': len(images) > 0,
                    'user_metadata': kwargs.get('user_metadata', {})
                }
                
                # Process the message through the orchestrator
                response = await self.orchestrator.process_query(
                    user_input=message,
                    context=processing_context,
                    conversation_id=context.conversation_id,
                    user_id=context.user_id
                )
                
                # Update conversation metadata
                context.metadata.update({
                    'last_interaction': datetime.utcnow().isoformat(),
                    'message_count': context.metadata.get('message_count', 0) + 1
                })
                
                # Save the updated context
                self.memory_manager.save_conversation(
                    conversation_id=context.conversation_id,
                    user_id=context.user_id,
                    metadata=context.metadata
                )
                
                # Calculate processing time
                processing_time = (datetime.utcnow() - start_time).total_seconds()
                
                # Log successful processing
                if self.monitor:
                    self.monitor.record_processing_time(
                        agent_name='naarad_agent',
                        processing_time=processing_time,
                        conversation_id=context.conversation_id
                    )
                
                return {
                    'success': True,
                    'response': response['output'],
                    'conversation_id': context.conversation_id,
                    'user_id': context.user_id,
                    'metadata': {
                        'processing_time_seconds': processing_time,
                        'agent_used': response.get('agent_used', 'unknown'),
                        'supporting_agents': response.get('metadata', {}).get('supporting_agents', [])
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
                    conversation_id=context.conversation_id
                )
            
            return {
                'success': False,
                'error': {
                    'id': error_id,
                    'type': type(e).__name__,
                    'message': 'An error occurred while processing your request.',
                    'details': str(e) if str(e) else 'No additional details available'
                },
                'conversation_id': context.conversation_id,
                'user_id': context.user_id
            }
    
    async def _get_or_create_context(
        self, 
        conversation_id: Optional[str], 
        user_id: Optional[str]
    ) -> ConversationContext:
        """Retrieve or create a conversation context."""
        if conversation_id:
            # Try to load existing conversation
            conversation = self.memory_manager.get_conversation(conversation_id, user_id)
            if conversation:
                return ConversationContext(
                    user_id=user_id or conversation.get('user_id', str(uuid.uuid4())),
                    conversation_id=conversation_id,
                    metadata=conversation.get('metadata', {}),
                    created_at=datetime.fromisoformat(conversation['created_at']),
                    updated_at=datetime.fromisoformat(conversation['updated_at'])
                )
        
        # Create a new conversation context
        return ConversationContext(
            user_id=user_id or str(uuid.uuid4()),
            conversation_id=conversation_id or f"conv_{uuid.uuid4().hex[:8]}",
            metadata={
                'created_at': datetime.utcnow().isoformat(),
                'updated_at': datetime.utcnow().isoformat(),
                'message_count': 0
            }
        )
    
    async def _process_images(
        self, 
        images: List[Any], 
        context: ConversationContext
    ) -> ConversationContext:
        """Process any attached images and update the context."""
        if not images:
            return context
            
        image_descriptions = []
        
        for img_data in images:
            try:
                description = await self.vision_tool.process_image(img_data)
                image_descriptions.append(description)
            except Exception as e:
                logger.error(f"Error processing image: {e}")
                continue
        
        if image_descriptions:
            # Store image descriptions in conversation metadata
            if 'images' not in context.metadata:
                context.metadata['images'] = []
                
            context.metadata['images'].extend([
                {
                    'description': desc,
                    'processed_at': datetime.utcnow().isoformat()
                }
                for desc in image_descriptions
            ])
            
            # Update the context timestamp
            context.update_timestamp()
        
        return context
    
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
    
    def get_conversation_history(
        self, 
        conversation_id: str, 
        user_id: str = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Retrieve conversation history with pagination."""
        try:
            conversation = self.memory_manager.get_conversation(conversation_id, user_id)
            if not conversation:
                raise HTTPException(status_code=404, detail="Conversation not found")
                
            messages = conversation.get('messages', [])
            total_messages = len(messages)
            
            # Apply pagination
            paginated_messages = messages[offset:offset + limit]
            
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
                    'created_at': conversation.get('created_at'),
                    'updated_at': conversation.get('updated_at')
                }
            }
            
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error retrieving conversation: {str(e)}"
            )
    
    def get_agent_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for all agents."""
        if not self.monitor:
            return {
                'success': False,
                'error': 'Monitoring is not enabled'
            }
            
        return {
            'success': True,
            'metrics': {
                'agent_performance': self.monitor.get_agent_performance(),
                'average_processing_time': self.monitor.get_average_processing_time()
            }
        }

# Singleton instance
naarad_agent = NaaradAgent()
