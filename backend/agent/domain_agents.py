from typing import Dict, Any, List, Optional, Type
from datetime import datetime, timedelta
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
import json
import re
import logging

from .memory.memory_manager import memory_manager
from .monitoring.agent_monitor import agent_monitor
from .registry import AgentRegistry, AgentInitializationError
from ..llm.config import settings

logger = logging.getLogger(__name__)

# Initialize the agent registry
agent_registry = AgentRegistry()

class DomainAgent:
    """Base class for all domain-specific agents."""
    
    # Class variables for agent registration
    agent_name: str = None
    agent_description: str = "A domain-specific agent"
    default_temperature: float = 0.7
    
    def __init__(self, name: str = None, system_prompt: str = None, tools: list = None, **kwargs):
        """Initialize the domain agent.
        
        Args:
            name: Optional name for the agent instance (defaults to class agent_name)
            system_prompt: Optional system prompt (defaults to class docstring)
            tools: List of tools to make available to the agent
            **kwargs: Additional arguments to pass to the LLM
        """
        self.name = name or self.agent_name or self.__class__.__name__.lower()
        self.system_prompt = system_prompt or self.__doc__ or ""
        self.tools = tools or []
        self.llm = self._create_llm(**kwargs)
        self.agent = self._create_agent()
        
        # Register the agent class if it has a name
        if self.agent_name and not agent_registry.is_registered(self.agent_name):
            agent_registry.register(self.agent_name, self.__class__)
    
    @classmethod
    def register(cls):
        """Class method to register the agent with the registry."""
        if not cls.agent_name:
            raise ValueError("Agent class must define an agent_name")
        agent_registry.register(cls.agent_name, cls)
        return cls
    
    def _create_llm(self, **kwargs):
        """Create the language model for this agent.
        
        Args:
            **kwargs: Additional arguments to pass to the LLM
            
        Returns:
            An instance of the language model
        """
        llm_kwargs = {
            "openai_api_base": "https://openrouter.ai/api/v1",
            "openai_api_key": settings.openrouter_api_key,
            "model_name": settings.chat_model,
            "temperature": self.default_temperature,
            **kwargs
        }
        return ChatOpenAI(**llm_kwargs)
    
    def _create_agent(self) -> AgentExecutor:
        """Create the agent executor with proper configuration.
        
        Returns:
            An instance of AgentExecutor configured for this agent
            
        Raises:
            AgentInitializationError: If agent creation fails
        """
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad")
            ])
            
            agent = create_openai_tools_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=prompt
            )
            
            return AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=settings.debug_mode,
                handle_parsing_errors=True,
                max_iterations=5,
                early_stopping_method="generate",
                max_execution_time=30  # seconds
            )
        except Exception as e:
            logger.error(f"Failed to create agent {self.name}: {str(e)}", exc_info=True)
            raise AgentInitializationError(f"Failed to create agent {self.name}: {str(e)}") from e
    
    async def process(
        self, 
        input_text: str, 
        context: Dict[str, Any], 
        conversation_id: str,
        user_id: str = "default"
    ) -> Dict[str, Any]:
        """Process input with this agent."""
        with agent_monitor.track_request(self.name) as tracker:
            try:
                # Get conversation history
                memory = memory_manager.get_conversation(conversation_id, user_id) or {}
                chat_history = memory.get('messages', [])
                
                # Process with agent
                result = await self.agent.ainvoke({
                    "input": input_text,
                    "chat_history": chat_history,
                    "context": json.dumps(context)
                })
                
                # Update agent state in memory
                self._update_agent_state(conversation_id, user_id, {
                    "last_used": datetime.utcnow().isoformat(),
                    "input": input_text,
                    "output": result["output"]
                })
                
                return {
                    "success": True,
                    "output": result["output"],
                    "agent": self.name
                }
                
            except Exception as e:
                agent_monitor.record_error(
                    self.name,
                    error_type=type(e).__name__,
                    error_message=str(e)
                )
                raise
    
    def _update_agent_state(
        self, 
        conversation_id: str, 
        user_id: str, 
        state: Dict[str, Any]
    ) -> None:
        """Update this agent's state in the conversation memory."""
        memory_manager.update_agent_state(
            conversation_id=conversation_id,
            user_id=user_id,
            agent_name=self.name,
            state=state
        )

def create_domain_agents() -> Dict[str, DomainAgent]:
    """Create and return all registered domain agents.
    
    This function initializes all registered domain agents and returns them
    in a dictionary keyed by their agent names.
    
    Returns:
        Dict[str, DomainAgent]: Dictionary mapping agent names to agent instances
        
    Raises:
        AgentInitializationError: If any agent fails to initialize
    """
    try:
        # This will automatically register all decorated agents
        from . import registry  # Import to trigger registration
        
        agents = {}
        for agent_name in agent_registry.get_all_agent_names():
            try:
                agents[agent_name] = agent_registry.get_agent(agent_name)
                logger.info(f"Successfully initialized agent: {agent_name}")
            except Exception as e:
                logger.error(f"Failed to initialize agent {agent_name}: {str(e)}", exc_info=True)
                if settings.strict_mode:
                    raise AgentInitializationError(f"Failed to initialize agent {agent_name}") from e
        
        return agents
        
    except Exception as e:
        logger.critical(f"Critical error initializing domain agents: {str(e)}", exc_info=True)
        raise

@DomainAgent.register
class TaskManagementAgent(DomainAgent):
    """Agent for managing tasks and reminders.
    
    This agent handles all task-related operations including creation, updates, 
    and queries about tasks and reminders.
    """
    
    agent_name = "task_manager"
    agent_description = "Manages tasks, reminders, and to-do lists"
    default_temperature = 0.3  # Lower temperature for more deterministic task management
    
    def __init__(self, **kwargs):
        """Initialize the task management agent."""
        # System prompt will be taken from the class docstring
        super().__init__(**kwargs)
        
        # Initialize any task-specific state
        self.task_storage = {}
    
    async def process(self, input_text: str, context: Dict[str, Any], conversation_id: str, user_id: str = "default") -> Dict[str, Any]:
        """Process a task-related query.
        
        Args:
            input_text: The user's input text
            context: Additional context including any files or metadata
            conversation_id: ID of the current conversation
            user_id: ID of the current user
            
        Returns:
            Dict containing the response and metadata
        """
        try:
            # Get or create user's task list
            if user_id not in self.task_storage:
                self.task_storage[user_id] = []
                
            # Process with the base agent
            result = await super().process(input_text, context, conversation_id, user_id)
            
            # Add any task-specific processing here
            
            return result
            
        except Exception as e:
            logger.error(f"Error in TaskManagementAgent: {str(e)}", exc_info=True)
            return {
                "success": False,
                "output": "I encountered an error while processing your task. Please try again.",
                "error": str(e)
            }
    
    async def process(self, input_text: str, context: Dict[str, Any], conversation_id: str, user_id: str = "default") -> Dict[str, Any]:
        # Extract task-related information
        task_info = self._extract_task_info(input_text)
        
        # Get existing tasks
        memory = memory_manager.get_conversation(conversation_id, user_id) or {}
        tasks = memory.get('metadata', {}).get('tasks', [])
        
        # Process task command
        if "list" in input_text.lower():
            return await self._list_tasks(tasks)
        elif any(cmd in input_text.lower() for cmd in ["create", "add", "new"]):
            return await self._create_task(input_text, task_info, tasks, conversation_id, user_id)
        elif any(cmd in input_text.lower() for cmd in ["complete", "done", "finish"]):
            return await self._update_task_status(input_text, tasks, "completed", conversation_id, user_id)
        else:
            return await super().process(input_text, context, conversation_id, user_id)
    
    def _extract_task_info(self, text: str) -> Dict[str, Any]:
        # Simple regex pattern to extract task details
        # This can be enhanced with more sophisticated NLP
        return {"description": text}
    
    async def _list_tasks(self, tasks: List[Dict]) -> Dict[str, Any]:
        if not tasks:
            return {"success": True, "output": "You have no tasks.", "agent": self.name}
        
        task_list = "\n".join(
            f"{i+1}. {task['description']} "
            f"[Due: {task.get('due_date', 'No due date')}] "
            f"- {task.get('status', 'pending')}"
            for i, task in enumerate(tasks)
        )
        return {"success": True, "output": f"Your tasks:\n{task_list}", "agent": self.name}
    
    async def _create_task(
        self, 
        input_text: str, 
        task_info: Dict, 
        tasks: List[Dict], 
        conversation_id: str, 
        user_id: str
    ) -> Dict[str, Any]:
        new_task = {
            "id": len(tasks) + 1,
            "description": task_info["description"],
            "created_at": datetime.utcnow().isoformat(),
            "status": "pending"
        }
        
        # Update tasks in memory
        memory = memory_manager.get_conversation(conversation_id, user_id) or {}
        metadata = memory.get('metadata', {})
        metadata.setdefault('tasks', []).append(new_task)
        
        memory_manager.save_conversation(
            conversation_id=conversation_id,
            user_id=user_id,
            messages=memory.get('messages', []),
            metadata=metadata
        )
        
        return {
            "success": True, 
            "output": f"Task created: {new_task['description']}",
            "agent": self.name
        }
    
    async def _update_task_status(
        self, 
        input_text: str, 
        tasks: List[Dict], 
        status: str,
        conversation_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        # Simple implementation - can be enhanced with better task matching
        task_idx = None
        for i, task in enumerate(tasks):
            if task["description"].lower() in input_text.lower():
                task_idx = i
                break
        
        if task_idx is not None:
            tasks[task_idx]["status"] = status
            tasks[task_idx]["completed_at"] = datetime.utcnow().isoformat()
            
            # Update tasks in memory
            memory = memory_manager.get_conversation(conversation_id, user_id) or {}
            metadata = memory.get('metadata', {})
            metadata['tasks'] = tasks
            
            memory_manager.save_conversation(
                conversation_id=conversation_id,
                user_id=user_id,
                messages=memory.get('messages', []),
                metadata=metadata
            )
            
            return {
                "success": True, 
                "output": f"Marked task as {status}: {tasks[task_idx]['description']}",
                "agent": self.name
            }
        
        return {
            "success": False, 
            "output": "Couldn't find a matching task to update.",
            "agent": self.name
        }

@DomainAgent.register
class CreativeWritingAgent(DomainAgent):
    """Agent specialized in creative writing and storytelling.
    
    This agent helps with all forms of creative writing including stories, 
    poems, character development, and narrative structures. It can adapt to 
    different styles, genres, and tones based on user preferences.
    """
    
    agent_name = "creative_writer"
    agent_description = "Specialized in creative writing and storytelling"
    default_temperature = 0.8  # Higher temperature for more creative outputs
    
    def __init__(self, **kwargs):
        """Initialize the creative writing agent."""
        super().__init__(**kwargs)
        self.writing_styles = ["narrative", "poetic", "descriptive", "conversational"]
        self.current_style = "narrative"
    
    async def process(self, input_text: str, context: Dict[str, Any], conversation_id: str, user_id: str = "default") -> Dict[str, Any]:
        """Process a creative writing request.
        
        Args:
            input_text: The user's input text
            context: Additional context including any files or metadata
            conversation_id: ID of the current conversation
            user_id: ID of the current user
            
        Returns:
            Dict containing the response and metadata
        """
        try:
            # Check for style changes in the input
            for style in self.writing_styles:
                if f"write in {style} style" in input_text.lower():
                    self.current_style = style
                    return {
                        "success": True,
                        "output": f"Switched to {style} writing style.",
                        "metadata": {"style": style}
                    }
            
            # Process with the base agent
            result = await super().process(input_text, context, conversation_id, user_id)
            
            # Add style information to the response
            if result.get("success", False):
                result["metadata"] = result.get("metadata", {})
                result["metadata"]["writing_style"] = self.current_style
                
            return result
            
        except Exception as e:
            logger.error(f"Error in CreativeWritingAgent: {str(e)}", exc_info=True)
            return {
                "success": False,
                "output": "I encountered an error while processing your creative request. Please try again.",
                "error": str(e)
            }

@DomainAgent.register
class AnalysisAgent(DomainAgent):
    """Agent for deep analysis and research.
    
    This agent specializes in analyzing information, identifying patterns, 
    and providing structured insights. It can handle complex analytical tasks,
    compare different options, and present findings in a clear, organized manner.
    """
    
    agent_name = "analyst"
    agent_description = "Specialized in analysis, research, and structured insights"
    default_temperature = 0.3  # Lower temperature for more focused outputs
    
    def __init__(self, **kwargs):
        """Initialize the analysis agent."""
        super().__init__(**kwargs)
        self.analysis_methods = ["swot", "pestle", "cost_benefit", "pros_cons"]
    
    async def process(self, input_text: str, context: Dict[str, Any], conversation_id: str, user_id: str = "default") -> Dict[str, Any]:
        """Process an analysis request.
        
        Args:
            input_text: The user's input text
            context: Additional context including any files or metadata
            conversation_id: ID of the current conversation
            user_id: ID of the current user
            
        Returns:
            Dict containing the response and metadata
        """
        try:
            # Check for specific analysis method in the input
            analysis_type = None
            for method in self.analysis_methods:
                if method in input_text.lower().replace(" ", "_"):
                    analysis_type = method
                    break
            
            # Process with the base agent
            result = await super().process(input_text, context, conversation_id, user_id)
            
            # Add analysis metadata to the response
            if result.get("success", False):
                result["metadata"] = result.get("metadata", {})
                if analysis_type:
                    result["metadata"]["analysis_type"] = analysis_type
                
            return result
            
        except Exception as e:
            logger.error(f"Error in AnalysisAgent: {str(e)}", exc_info=True)
            return {
                "success": False,
                "output": "I encountered an error while analyzing your request. Please try again.",
                "error": str(e)
            }

@DomainAgent.register
class ContextAwareChatAgent(DomainAgent):
    """Agent that maintains context and manages conversation flow.
    
    This agent is responsible for maintaining conversation context, managing 
    topic changes, and ensuring coherent, relevant responses. It tracks 
    conversation history and can reference previous interactions.
    """
    
    agent_name = "context_manager"
    agent_description = "Manages conversation context and flow"
    default_temperature = 0.5  # Balanced temperature for conversational responses
    
    def __init__(self, **kwargs):
        """Initialize the context-aware chat agent."""
        super().__init__(**kwargs)
        self.conversation_history = {}
    
    async def process(self, input_text: str, context: Dict[str, Any], conversation_id: str, user_id: str = "default") -> Dict[str, Any]:
        """Process a chat message with context awareness.
        
        Args:
            input_text: The user's input text
            context: Additional context including any files or metadata
            conversation_id: ID of the current conversation
            user_id: ID of the current user
            
        Returns:
            Dict containing the response and metadata
        """
        try:
            # Get or initialize conversation history
            if conversation_id not in self.conversation_history:
                self.conversation_history[conversation_id] = []
            
            # Add user message to history
            self.conversation_history[conversation_id].append({"role": "user", "content": input_text})
            
            # Get conversation memory
            memory = memory_manager.get_conversation(conversation_id, user_id) or {}
            chat_history = memory.get('messages', [])
            
            # Analyze context
            context_analysis = self._analyze_context(input_text, chat_history, context)
            
            # Process with the base agent
            result = await super().process(input_text, context, conversation_id, user_id)
            
            if result.get("success", False):
                # Add assistant response to history
                self.conversation_history[conversation_id].append({
                    "role": "assistant", 
                    "content": result.get("output", "")
                })
                
                # Add context metadata to the response
                result["metadata"] = result.get("metadata", {})
                result["metadata"]["conversation_length"] = len(self.conversation_history[conversation_id])
                result["metadata"]["context_analysis"] = context_analysis
                
                # Add conversation history to context for next turn
                context["conversation_history"] = self.conversation_history[conversation_id][-10:]  # Last 10 messages
            
            return result
            
        except Exception as e:
            logger.error(f"Error in ContextAwareChatAgent: {str(e)}", exc_info=True)
            return {
                "success": False,
                "output": "I encountered an error while processing your message. Please try again.",
                "error": str(e)
            }
    
    def _analyze_context(
        self, 
        input_text: str, 
        chat_history: List[Dict], 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze the conversation context.
        
        Args:
            input_text: The current user input
            chat_history: List of previous messages in the conversation
            context: Additional context data
            
        Returns:
            Dict containing analysis results including topic changes and suggestions
        """
        analysis = {
            'topic_changed': False,
            'needs_clarification': False,
            'suggested_actions': [],
            'topics': set(),
            'entities': set()
        }
        
        # Simple keyword extraction for topic tracking
        input_lower = input_text.lower()
        
        # Check for topic changes by comparing with recent history
        if chat_history and len(chat_history) > 1:
            # Get the last few messages (excluding current input)
            recent_messages = [msg.get('content', '') for msg in chat_history[-3:]]
            recent_text = ' '.join(recent_messages).lower()
            
            # Simple keyword-based topic change detection
            recent_words = set(word for word in recent_text.split() if len(word) > 3)
            current_words = set(word for word in input_lower.split() if len(word) > 3)
            
            # If less than 30% of words match, consider it a topic change
            if recent_words and current_words:
                overlap = len(recent_words.intersection(current_words)) / len(current_words)
                analysis['topic_changed'] = overlap < 0.3
        
        # Check for ambiguous references that might need clarification
        ambiguous_pronouns = {'it', 'that', 'this', 'they', 'them', 'those'}
        if any(pronoun in input_lower.split() for pronoun in ambiguous_pronouns):
            # Check if the pronoun refers to something in recent context
            analysis['needs_clarification'] = True
            analysis['suggested_actions'].append('request_clarification')
        
        # Extract potential entities (simple implementation)
        # In a real system, you'd use NER here
        potential_entities = [word for word in input_text.split() if word[0].isupper() and len(word) > 2]
        if potential_entities:
            analysis['entities'].update(potential_entities)
        
        # Check for questions that might need clarification
        question_words = {'who', 'what', 'when', 'where', 'why', 'how', 'which'}
        if any(input_lower.startswith(word) for word in question_words):
            analysis['suggested_actions'].append('provide_detailed_response')
        
        return analysis
