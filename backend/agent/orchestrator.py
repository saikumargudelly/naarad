from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json
import logging
from langchain.agents import AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from .memory.memory_manager import memory_manager
from .monitoring.agent_monitor import agent_monitor
from .domain_agents import create_domain_agents

logger = logging.getLogger(__name__)

class AgentOrchestrator:
    def __init__(self, base_agents: Dict[str, AgentExecutor]):
        """Initialize the orchestrator with base agents and domain agents."""
        self.base_agents = base_agents
        self.domain_agents = create_domain_agents()
        self.conversation_history = []
        self.current_topic = None
        self.last_interaction_time = datetime.utcnow()
    
    async def process_query(
        self, 
        user_input: str, 
        context: Dict[str, Any] = None,
        conversation_id: str = "default",
        user_id: str = "default"
    ) -> Dict[str, Any]:
        """Process a user query through the appropriate agent network.
        
        Args:
            user_input: The user's input text
            context: Additional context including any files or metadata
            conversation_id: ID of the current conversation
            user_id: ID of the current user
            
        Returns:
            Dict containing the response and metadata
        """
        context = context or {}
        
        # Check for image data in context
        if 'image_data' in context or 'image_url' in context:
            context['has_image'] = True
            
            # If the input is just about the image, add a default prompt
            if not user_input.strip() or user_input.strip().lower() in ['what is this', 'describe this image']:
                user_input = "What is in this image?"
        
        # Track processing time
        start_time = datetime.utcnow()
        
        try:
            # Update conversation history (without the potentially large image data)
            history_context = context.copy()
            if 'image_data' in history_context:
                history_context['has_image'] = True
                del history_context['image_data']
                
            self._update_history(
                user_input=user_input,
                conversation_id=conversation_id,
                user_id=user_id
            )
            
            # 1. Analyze context and determine the best agent(s) to handle the query
            agent_decision = await self._select_agent(
                user_input=user_input,
                context=context,
                conversation_id=conversation_id,
                user_id=user_id
            )
            
            # 2. Process with the selected agent(s)
            response = await self._route_to_agent(
                agent_name=agent_decision['primary_agent'],
                user_input=user_input,
                context=context,
                conversation_id=conversation_id,
                user_id=user_id,
                supporting_agents=agent_decision.get('supporting_agents', [])
            )
            
            # 3. Update conversation history with the response
            self._update_history(
                user_input=user_input,
                assistant_response=response['output'],
                conversation_id=conversation_id,
                user_id=user_id
            )
            
            # 4. Update metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            agent_monitor.record_processing_time(
                agent_name=agent_decision['primary_agent'],
                processing_time=processing_time,
                conversation_id=conversation_id
            )
            
            return {
                'success': True,
                'output': response['output'],
                'agent_used': agent_decision['primary_agent'],
                'metadata': {
                    'processing_time_seconds': processing_time,
                    'supporting_agents': agent_decision.get('supporting_agents', [])
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            agent_monitor.record_processing_error(
                agent_name='orchestrator',
                error_type=type(e).__name__,
                error_message=str(e),
                conversation_id=conversation_id
            )
            
            return {
                'success': False,
                'output': 'I encountered an error processing your request. Please try again.',
                'error': str(e)
            }
    
    async def _select_agent(
        self, 
        user_input: str,
        context: Dict[str, Any],
        conversation_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """Determine which agent(s) should handle the query."""
        # Get conversation context
        memory = memory_manager.get_conversation(conversation_id, user_id) or {}
        
        # Check for domain-specific intents
        domain_agent = self._identify_domain(user_input, context, memory)
        
        if domain_agent:
            return {
                'primary_agent': domain_agent,
                'supporting_agents': ['context_manager']
            }
        
        # Default to general response agent
        return {
            'primary_agent': 'responder',
            'supporting_agents': ['context_manager']
        }
    
    def _identify_domain(
        self, 
        user_input: str, 
        context: Dict[str, Any],
        memory: Dict[str, Any]
    ) -> Optional[str]:
        """Identify if the query belongs to a specific domain.
        
        Args:
            user_input: The user's input text
            context: Additional context including any attached files or metadata
            memory: The conversation memory
            
        Returns:
            Name of the domain agent to handle the query, or None for default handling
        """
        input_lower = user_input.lower()
        
        # Check for image in context (handled by responder with vision tool)
        if context.get('has_image', False) or 'image' in input_lower or 'photo' in input_lower:
            return 'responder'  # Will use vision tool
            
        # Web search queries
        search_keywords = ['search', 'find', 'look up', 'latest', 'current', 'news', 'update']
        question_words = ['who', 'what', 'when', 'where', 'why', 'how']
        
        # If it's a question that might need current info
        is_question = any(input_lower.startswith(word) for word in question_words)
        needs_search = any(keyword in input_lower for keyword in search_keywords)
        
        if is_question or needs_search:
            return 'researcher'  # Will use Brave Search tool
        
        # Task management
        task_keywords = ['task', 'todo', 'remind', 'due', 'deadline']
        if any(keyword in input_lower for keyword in task_keywords):
            return 'task_manager'
            
        # Creative writing
        writing_keywords = ['write', 'story', 'poem', 'creative', 'compose']
        if any(keyword in input_lower for keyword in writing_keywords):
            return 'creative_writer'
            
        # Analysis - complex queries that need reasoning
        analysis_keywords = ['analyze', 'research', 'compare', 'explain in detail', 'pros and cons']
        if any(keyword in input_lower for keyword in analysis_keywords):
            return 'analyst'
            
        # Check conversation context from memory
        if 'metadata' in memory and 'active_domain' in memory['metadata']:
            return memory['metadata']['active_domain']
            
        return None
    
    async def _route_to_agent(
        self,
        agent_name: str,
        user_input: str,
        context: Dict[str, Any],
        conversation_id: str,
        user_id: str,
        supporting_agents: List[str] = None
    ) -> Dict[str, Any]:
        """Route the query to the appropriate agent.
        
        Args:
            agent_name: Primary agent to handle the request
            user_input: The user's input text
            context: Additional context including any files or metadata
            conversation_id: ID of the current conversation
            user_id: ID of the current user
            supporting_agents: List of additional agents that can assist
            
        Returns:
            Dict containing the response and metadata
        """
        supporting_agents = supporting_agents or []
        
        # Special handling for researcher to include web search context
        if agent_name == 'researcher':
            context['search_performed'] = True
            
        # First, try to use a domain agent if available
        if agent_name in self.domain_agents:
            response = await self._process_with_domain_agent(
                agent_name=agent_name,
                user_input=user_input,
                context=context,
                conversation_id=conversation_id,
                user_id=user_id
            )
        else:
            # Use a base agent
            response = await self._process_with_base_agent(
                agent_name=agent_name,
                user_input=user_input,
                context=context,
                conversation_id=conversation_id,
                user_id=user_id
            )
        
        # If we have supporting agents and the primary agent needs help
        if supporting_agents and response.get('needs_assistance', False):
            for support_agent in supporting_agents:
                if support_agent != agent_name:  # Don't call the same agent again
                    support_response = await self._route_to_agent(
                        agent_name=support_agent,
                        user_input=user_input,
                        context={
                            **context,
                            'previous_agent': agent_name,
                            'previous_response': response
                        },
                        conversation_id=conversation_id,
                        user_id=user_id
                    )
                    
                    # If the supporting agent was successful, use its response
                    if support_response.get('success', False):
                        return support_response
        
        return response
    
    async def _process_with_base_agent(
        self,
        agent_name: str,
        user_input: str,
        context: Dict[str, Any],
        conversation_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """Process a query with one of the base agents.
        
        Handles special cases like image analysis using the vision tool.
        """
        if agent_name not in self.base_agents:
            raise ValueError(f"Unknown base agent: {agent_name}")
            
        agent = self.base_agents[agent_name]
        
        # Handle image analysis
        if context.get('has_image', False) and agent_name == 'responder':
            return await self._handle_image_analysis(user_input, context, conversation_id, user_id)
        
        # Prepare the input with conversation history
        history = self._get_formatted_history(conversation_id, user_id)
        
        try:
            # For researcher, add web search context if needed
            if agent_name == 'researcher' and not context.get('search_performed', False):
                user_input = f"[Web search required] {user_input}"
            
            response = await agent.process(
                input_text=user_input,
                chat_history=history,
                **context
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in {agent_name} agent: {str(e)}", exc_info=True)
            return {
                'success': False,
                'output': f"I encountered an error while processing your request with {agent_name}.",
                'error': str(e)
            }
            
    async def _handle_image_analysis(
        self,
        user_input: str,
        context: Dict[str, Any],
        conversation_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """Handle image analysis using the vision tool."""
        try:
            # Get the responder agent which has the vision tool
            responder = self.base_agents['responder']
            
            # Prepare the input for the vision tool
            image_url = context.get('image_url')
            if not image_url and 'image_data' in context:
                # If we have raw image data, we'd need to upload it somewhere first
                # For now, we'll just use a placeholder
                return {
                    'success': False,
                    'output': "Image upload from data is not yet supported. Please provide a URL.",
                    'error': 'image_upload_not_supported'
                }
                
            # Create the input for the vision tool
            vision_input = {
                'image_url': image_url,
                'prompt': user_input or "What is in this image?"
            }
            
            # Process with the vision tool
            vision_response = await responder.process(
                input_text=json.dumps(vision_input),
                chat_history=self._get_formatted_history(conversation_id, user_id)
            )
            
            if not vision_response.get('success', False):
                raise Exception(vision_response.get('error', 'Unknown error in vision tool'))
                
            return {
                'success': True,
                'output': vision_response.get('output', 'No description available'),
                'metadata': {
                    'tool_used': 'vision',
                    'image_url': image_url
                }
            }
            
        except Exception as e:
            logger.error(f"Error in image analysis: {str(e)}", exc_info=True)
            return {
                'success': False,
                'output': "I couldn't analyze the image. Please try again with a different image or description.",
                'error': str(e)
            }

    def _update_history(
        self,
        user_input: str,
        assistant_response: str = None,
        conversation_id: str = "default",
        user_id: str = "default"
    ) -> None:
        """Update the conversation history in memory."""
        memory = memory_manager.get_conversation(conversation_id, user_id) or {}
        messages = memory.get('messages', [])
        
        # Add user message
        messages.append({
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        # Add assistant response if provided
        if assistant_response is not None:
            messages.append({
                'role': 'assistant',
                'content': assistant_response,
                'timestamp': datetime.utcnow().isoformat()
            })
        
        # Update memory
        memory_manager.save_conversation(
            conversation_id=conversation_id,
            user_id=user_id,
            messages=messages,
            metadata=memory.get('metadata', {})
        )
    
    def _get_formatted_history(
        self,
        conversation_id: str,
        user_id: str,
        max_messages: int = 10
    ) -> List[Dict[str, str]]:
        """Get formatted conversation history for agent input."""
        memory = memory_manager.get_conversation(conversation_id, user_id) or {}
        messages = memory.get('messages', [])
        
        # Return last N messages, keeping the most recent
        recent_messages = messages[-max_messages:]
        
        return [
            {
                'role': msg['role'],
                'content': msg['content']
            }
            for msg in recent_messages
        ]
