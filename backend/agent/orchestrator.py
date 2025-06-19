from typing import Dict, Any, List, Optional, Tuple, Type
from datetime import datetime
import json
import logging
import importlib
import time
import asyncio

# LangChain imports
from langchain.agents import AgentExecutor
from langchain.agents.agent import AgentFinish, AgentAction, AgentStep
from langchain.tools import BaseTool as LangChainBaseTool
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# Local imports
from .memory.memory_manager import memory_manager
from .monitoring.agent_monitor import agent_monitor
from .types import BaseAgent, AgentConfig

# Lazy imports to avoid circular dependencies
_domain_agents = None

def _get_domain_agents():
    global _domain_agents
    if _domain_agents is None:
        from .domain_agents import create_domain_agents
        _domain_agents = create_domain_agents()
    return _domain_agents

logger = logging.getLogger(__name__)

class AgentOrchestrator:
    def __init__(self, base_agents: Dict[str, AgentExecutor] = None):
        """Initialize the orchestrator with base agents and domain agents.
        
        Args:
            base_agents: Optional dictionary of base agents. If not provided,
                       an empty dict will be used.
        """
        self.base_agents = base_agents or {}
        self.domain_agents = _get_domain_agents()
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
        supporting_agents: List[str] = None,
        fallback_agent: str = None
    ) -> Dict[str, Any]:
        """Route the query to the appropriate agent with enhanced error handling and fallback support.
        
        Args:
            agent_name: Primary agent to handle the request
            user_input: The user's input text
            context: Additional context including any files or metadata
            conversation_id: ID of the current conversation
            user_id: ID of the current user
            supporting_agents: List of additional agents that can assist if the primary agent needs help
            fallback_agent: Optional fallback agent to try if all other agents fail
            
        Returns:
            Dict containing the response and metadata
        """
        logger.info(f"[Orchestrator] Routing to agent: {agent_name}" +
                   (f" with {len(supporting_agents or [])} supporting agents" if supporting_agents else ""))
        
        supporting_agents = supporting_agents or []
        start_time = time.time()
        
        # Prepare context with routing information
        context = context.copy()
        context['routing'] = {
            'primary_agent': agent_name,
            'supporting_agents': supporting_agents,
            'attempt_time': datetime.utcnow().isoformat()
        }
        
        # Special handling for specific agent types
        if agent_name == 'researcher':
            context['search_performed'] = True
            context['max_iterations'] = min(context.get('max_iterations', 8), 10)
            logger.debug("[Orchestrator] Enabled search for researcher agent")
        
        try:
            # Try the primary agent first
            if agent_name in self.domain_agents:
                logger.debug(f"[Orchestrator] Using domain agent: {agent_name}")
                response = await self._process_with_domain_agent(
                    agent_name=agent_name,
                    user_input=user_input,
                    context=context,
                    conversation_id=conversation_id,
                    user_id=user_id
                )
            else:
                logger.debug(f"[Orchestrator] Using base agent: {agent_name}")
                response = await self._process_with_base_agent(
                    agent_name=agent_name,
                    user_input=user_input,
                    context=context,
                    conversation_id=conversation_id,
                    user_id=user_id
                )
            
            # Check if the response indicates a need for assistance
            needs_assistance = (
                not response.get('success', False) or
                response.get('needs_assistance', False) or
                (isinstance(response.get('output'), str) and 
                 any(phrase in response['output'].lower() for phrase in 
                     ['i don\'t know', 'i don\'t have enough information', 'i need help']))
            )
            
            # If we have supporting agents and the primary agent needs help
            if supporting_agents and needs_assistance:
                logger.info(f"[Orchestrator] Primary agent {agent_name} needs assistance, trying {len(supporting_agents)} supporting agents")
                
                for i, support_agent in enumerate(supporting_agents, 1):
                    if support_agent != agent_name:  # Don't call the same agent again
                        logger.info(f"[Orchestrator] Trying supporting agent {i}/{len(supporting_agents)}: {support_agent}")
                        
                        # Update context with previous attempt info
                        support_context = {
                            **context,
                            'previous_agent': agent_name,
                            'previous_response': response,
                            'attempt_count': i,
                            'is_fallback': False
                        }
                        
                        try:
                            support_response = await self._route_to_agent(
                                agent_name=support_agent,
                                user_input=user_input,
                                context=support_context,
                                conversation_id=conversation_id,
                                user_id=user_id,
                                supporting_agents=[a for a in supporting_agents if a != support_agent]  # Don't retry agents
                            )
                            
                            # If the supporting agent was successful, use its response
                            if support_response.get('success', False):
                                logger.info(f"[Orchestrator] Supporting agent {support_agent} provided a successful response")
                                return support_response
                                
                        except Exception as e:
                            logger.error(f"[Orchestrator] Error in supporting agent {support_agent}: {str(e)}", exc_info=True)
                            continue
            
            # If we have a fallback agent and all else failed
            if fallback_agent and (not response.get('success', False) or needs_assistance):
                logger.info(f"[Orchestrator] All agents failed, trying fallback agent: {fallback_agent}")
                try:
                    fallback_response = await self._route_to_agent(
                        agent_name=fallback_agent,
                        user_input=user_input,
                        context={
                            **context,
                            'is_fallback': True,
                            'previous_attempts': [agent_name] + supporting_agents
                        },
                        conversation_id=conversation_id,
                        user_id=user_id
                    )
                    if fallback_response.get('success', False):
                        return fallback_response
                except Exception as e:
                    logger.error(f"[Orchestrator] Error in fallback agent {fallback_agent}: {str(e)}", exc_info=True)
            
            # Add routing metadata to the response
            if isinstance(response, dict):
                if 'metadata' not in response:
                    response['metadata'] = {}
                response['metadata'].update({
                    'routing': {
                        'primary_agent': agent_name,
                        'supporting_agents_attempted': supporting_agents,
                        'processing_time_seconds': time.time() - start_time,
                        'used_fallback': fallback_agent if (not response.get('success', False) or needs_assistance) else None
                    }
                })
            
            return response
            
        except Exception as e:
            error_msg = f"Error in agent routing for {agent_name}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Try to use a fallback agent if available
            if fallback_agent and fallback_agent != agent_name:
                logger.info(f"[Orchestrator] Error in primary agent, trying fallback: {fallback_agent}")
                try:
                    return await self._route_to_agent(
                        agent_name=fallback_agent,
                        user_input=user_input,
                        context={
                            **context,
                            'previous_error': str(e),
                            'is_fallback': True
                        },
                        conversation_id=conversation_id,
                        user_id=user_id
                    )
                except Exception as fallback_error:
                    logger.error(f"[Orchestrator] Fallback agent also failed: {str(fallback_error)}", exc_info=True)
            
            # If all else fails, return an error response
            return {
                'success': False,
                'output': "I encountered an error while processing your request. Please try again.",
                'error': error_msg,
                'agent_used': agent_name,
                'metadata': {
                    'processing_time_seconds': time.time() - start_time,
                    'error_type': type(e).__name__,
                    'attempted_fallback': bool(fallback_agent and fallback_agent != agent_name)
                }
            }
    
    async def _process_with_base_agent(
        self,
        agent_name: str,
        user_input: str,
        context: Dict[str, Any],
        conversation_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """Process a query with one of the base agents with enhanced error handling and logging.
        
        Args:
            agent_name: Name of the agent to use
            user_input: The user's input text
            context: Additional context including any files or metadata
            conversation_id: ID of the current conversation
            user_id: ID of the current user
            
        Returns:
            Dict containing the agent's response and metadata
        """
        logger.info(f"[Orchestrator] Processing with base agent: {agent_name}")
        start_time = time.time()
        
        # Validate agent exists
        if agent_name not in self.base_agents:
            error_msg = f"Unknown base agent: {agent_name}. Available agents: {list(self.base_agents.keys())}"
            logger.error(error_msg)
            return {
                'success': False,
                'output': f"I couldn't find the '{agent_name}' agent to handle your request.",
                'error': error_msg,
                'agent_used': agent_name,
                'metadata': {
                    'processing_time_seconds': time.time() - start_time,
                    'error_type': 'agent_not_found'
                }
            }
            
        agent = self.base_agents[agent_name]
        
        try:
            # Handle special cases
            if context.get('has_image', False) and agent_name == 'responder':
                logger.info("[Orchestrator] Detected image in context, using image analysis")
                return await self._handle_image_analysis(user_input, context, conversation_id, user_id)
            
            # Prepare the input with conversation history
            history = self._get_formatted_history(conversation_id, user_id)
            
            # Prepare context for the agent
            process_kwargs = {
                'input_text': user_input,
                'chat_history': history,
                'conversation_id': conversation_id,
                'user_id': user_id,
                **context
            }
            
            # Special handling for researcher agent
            if agent_name == 'researcher':
                if not context.get('search_performed', False):
                    process_kwargs['input_text'] = f"[Research required] {user_input}"
                process_kwargs['max_iterations'] = min(context.get('max_iterations', 8), 10)
            
            # Log the processing details
            logger.info(f"[Orchestrator] Invoking {agent_name} agent with input: {user_input[:200]}...")
            logger.debug(f"[Orchestrator] Agent context: {json.dumps({k: str(v)[:200] for k, v in process_kwargs.items()}, indent=2)}")
            
            # Set a timeout for agent processing
            try:
                # Process with the agent
                response = await asyncio.wait_for(
                    agent.process(**process_kwargs),
                    timeout=60  # 60 second timeout
                )
                
                # Log the response
                processing_time = time.time() - start_time
                logger.info(f"[Orchestrator] {agent_name} agent completed in {processing_time:.2f}s")
                
                # Add processing metadata
                if isinstance(response, dict):
                    if 'metadata' not in response:
                        response['metadata'] = {}
                    response['metadata'].update({
                        'processing_time_seconds': processing_time,
                        'agent_used': agent_name,
                        'model': getattr(agent, 'model_name', 'unknown')
                    })
                
                return response
                
            except asyncio.TimeoutError:
                error_msg = f"{agent_name} agent timed out after {time.time() - start_time:.2f}s"
                logger.error(error_msg)
                return {
                    'success': False,
                    'output': f"The {agent_name} agent took too long to respond. Please try again with a more specific query.",
                    'error': error_msg,
                    'agent_used': agent_name,
                    'metadata': {
                        'processing_time_seconds': time.time() - start_time,
                        'error_type': 'timeout'
                    }
                }
            
        except Exception as e:
            error_msg = f"Error in {agent_name} agent: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                'success': False,
                'output': f"I encountered an error while processing your request with the {agent_name} agent.",
                'error': error_msg,
                'agent_used': agent_name,
                'metadata': {
                    'processing_time_seconds': time.time() - start_time,
                    'error_type': type(e).__name__
                }
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
