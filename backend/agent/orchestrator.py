from typing import Dict, Any, List, Optional, Tuple, Type, Union, Callable
import datetime
import json
import logging
import importlib
import asyncio
from functools import wraps
import time
import asyncio
import re
from dateutil import parser as date_parser
import dateutil.relativedelta
import pytz

# LangChain imports
from langchain.agents import AgentExecutor
from langchain.agents.agent import AgentFinish, AgentAction, AgentStep
from langchain.tools import BaseTool as LangChainBaseTool
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# Local imports
from .memory.memory_manager import memory_manager
from .monitoring.agent_monitor import agent_monitor
from .types import BaseAgent, AgentConfig
from .tools.brave_search import BraveSearchTool

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
        self.last_interaction_time = datetime.datetime.utcnow()
    
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
        start_time = datetime.datetime.utcnow()
        
        try:
            # Update conversation history (without the potentially large image data)
            history_context = context.copy()
            if 'image_data' in history_context:
                history_context['has_image'] = True
                del history_context['image_data']
                
            await self._update_history(
                user_input=user_input,
                conversation_id=conversation_id,
                user_id=user_id
            )
            
            # --- Entity Extraction and Context Update ---
            entities = self._extract_entities(user_input)
            if 'metadata' not in context:
                context['metadata'] = {}
            if 'entities' not in context['metadata']:
                context['metadata']['entities'] = {}
            # Merge new entities with previous ones
            for k, v in entities.items():
                if v:
                    context['metadata']['entities'][k] = v
            
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
            await self._update_history(
                user_input=user_input,
                assistant_response=response['output'],
                conversation_id=conversation_id,
                user_id=user_id
            )
            
            # 4. Update metrics
            processing_time = (datetime.datetime.utcnow() - start_time).total_seconds()
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
    
    async def process_message_stream(
        self,
        user_input: str,
        conversation_id: str,
        user_id: str,
        message_type: str = "text",
        **kwargs
    ):
        """Process a message and stream the response back."""
        try:
            # 1. Select agent
            agent_decision = await self._select_agent(
                user_input=user_input,
                context=kwargs,
                conversation_id=conversation_id,
                user_id=user_id
            )
            primary_agent = agent_decision['primary_agent']

            # Yield typing start indicator
            yield {"type": "typing_start", "agent": primary_agent}

            # 2. Route to the appropriate agent's streaming method if it exists
            if primary_agent in self.domain_agents and hasattr(self.domain_agents[primary_agent], 'process_stream'):
                agent = self.domain_agents[primary_agent]
                async for chunk in agent.process_stream(user_input, kwargs, conversation_id, user_id):
                    yield chunk
            elif primary_agent in self.base_agents and hasattr(self.base_agents[primary_agent], 'process_stream'):
                agent = self.base_agents[primary_agent]
                async for chunk in agent.process_stream(user_input, kwargs, conversation_id, user_id):
                    yield chunk
            else:
                # Fallback to non-streaming for agents that don't support it
                response = await self._route_to_agent(
                    agent_name=primary_agent,
                    user_input=user_input,
                    context=kwargs,
                    conversation_id=conversation_id,
                    user_id=user_id
                )
                yield {
                    "type": "message",
                    "content": response.get('output', 'No response generated.'),
                    "agent": primary_agent
                }

        except Exception as e:
            logger.error(f"Error in process_message_stream: {str(e)}", exc_info=True)
            yield {
                "type": "error",
                "error": f"A streaming error occurred: {str(e)}"
            }
        finally:
            # Ensure a message_complete is sent
            yield {
                "type": "message_complete",
                "conversation_id": conversation_id
            }
    
    async def _select_agent(
        self, 
        user_input: str,
        context: Dict[str, Any],
        conversation_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """Determine which agent(s) should handle the query.
        
        Args:
            user_input: The user's input text
            context: Additional context including any files or metadata
            conversation_id: ID of the current conversation
            user_id: ID of the current user
            
        Returns:
            Dict containing the primary agent and any supporting agents
        """
        try:
            # Get conversation context asynchronously
            memory_coro = memory_manager.get_conversation(conversation_id, user_id)
            memory = await memory_coro if hasattr(memory_coro, '__await__') else memory_coro or {}
            
            # Check for domain-specific intents
            domain_agent = await self._identify_domain(user_input, context, memory)
            
            if domain_agent:
                return {
                    'primary_agent': domain_agent,
                    'supporting_agents': []
                }
            
            # Default to general response agent
            return {
                'primary_agent': 'responder',
                'supporting_agents': []
            }
            
        except Exception as e:
            logger.error(f"Error in _select_agent: {str(e)}", exc_info=True)
            # Fallback to responder agent on error
            return {
                'primary_agent': 'responder',
                'supporting_agents': []
            }
    
    async def _identify_domain(
        self, 
        user_input: str, 
        context: Dict[str, Any],
        memory: Dict[str, Any]
    ) -> Optional[str]:
        """Identify the appropriate domain agent for the query.
        
        Args:
            user_input: The user's input text
            context: Additional context including any files or metadata
            memory: Conversation memory and context
            
        Returns:
            Optional[str]: The name of the domain agent to use, or None for default
        """
        try:
            # Use enhanced router for intent classification
            from .enhanced_router import EnhancedRouter
            router = EnhancedRouter()
            
            # Classify intent
            intent_match = await router.classify_intent(user_input, context)
            intent = intent_match.intent
            
            # Map intents to domain agents
            intent_to_agent = {
                'task_manager': ['reminder', 'calendar', 'task'],
                'creative_writer': ['creative_writing', 'story', 'narrative'],
                'analyst': ['analysis', 'research', 'investigation'],
                'context_manager': ['conversation', 'context', 'follow_up'],
                'emotion_agent': ['emotion', 'feeling', 'mood', 'empathy'],
                'creativity_agent': ['creativity', 'brainstorm', 'ideas', 'innovation'],
                'prediction_agent': ['prediction', 'forecast', 'trend', 'pattern'],
                'learning_agent': ['learning', 'improve', 'adapt', 'feedback'],
                'quantum_agent': ['quantum', 'superposition', 'entanglement', 'quantum_computing'],
                'search': ['researcher']
            }
            
            # Check for specific domain keywords
            user_input_lower = user_input.lower()
            
            # Search and research requests
            if any(word in user_input_lower for word in ['search', 'find', 'look up', 'google', 'research', 'investigate', 'what is', 'who is', 'where is', 'when is', 'how to']):
                return 'researcher'
            
            # Emotion detection
            if any(word in user_input_lower for word in ['feel', 'emotion', 'mood', 'sad', 'happy', 'angry', 'worried', 'excited', 'frustrated']):
                return 'emotion_agent'
            
            # Creativity requests
            if any(word in user_input_lower for word in ['creative', 'brainstorm', 'ideas', 'innovative', 'story', 'narrative', 'inspiration']):
                return 'creativity_agent'
            
            # Prediction requests
            if any(word in user_input_lower for word in ['predict', 'forecast', 'future', 'trend', 'pattern', 'what will happen', 'outcome']):
                return 'prediction_agent'
            
            # Learning requests
            if any(word in user_input_lower for word in ['learn', 'improve', 'adapt', 'feedback', 'better', 'optimize', 'enhance']):
                return 'learning_agent'
            
            # Quantum requests
            if any(word in user_input_lower for word in ['quantum', 'superposition', 'entanglement', 'tunneling', 'quantum computing', 'qubit']):
                return 'quantum_agent'
            
            # Task management
            if any(word in user_input_lower for word in ['remind', 'task', 'todo', 'schedule', 'appointment', 'meeting']):
                return 'task_manager'
            
            # Creative writing
            if any(word in user_input_lower for word in ['write', 'story', 'creative', 'narrative', 'character', 'plot']):
                return 'creative_writer'
            
            # Analysis
            if any(word in user_input_lower for word in ['analyze', 'research', 'investigate', 'examine', 'study']):
                return 'analyst'
            
            # Context management
            if any(word in user_input_lower for word in ['remember', 'context', 'previous', 'earlier', 'conversation']):
                return 'context_manager'
            
            # Check intent match from enhanced router
            if intent and hasattr(intent, 'value'):
                intent_value = intent.value
                
                # Map intent values to agents
                if intent_value in ['emotion']:
                    return 'emotion_agent'
                elif intent_value in ['creativity']:
                    return 'creativity_agent'
                elif intent_value in ['prediction']:
                    return 'prediction_agent'
                elif intent_value in ['learning']:
                    return 'learning_agent'
                elif intent_value in ['quantum']:
                    return 'quantum_agent'
                elif intent_value in ['search']:
                    return 'researcher'
                elif intent_value in ['reminder', 'calendar']:
                    return 'task_manager'
                elif intent_value in ['creative_writing']:
                    return 'creative_writer'
                elif intent_value in ['analysis']:
                    return 'analyst'
                elif intent_value in ['context']:
                    return 'context_manager'
            
            # Check conversation history for context
            if memory and 'messages' in memory:
                recent_messages = memory['messages'][-5:]  # Last 5 messages
                for message in recent_messages:
                    if 'agent_used' in message:
                        # Continue with the same agent if it was working well
                        return message['agent_used']
            
            # Default to context manager for general conversation
            return 'context_manager'
            
        except Exception as e:
            logger.error(f"Error in domain identification: {str(e)}", exc_info=True)
            return 'context_manager'  # Safe fallback
    
    def _is_realtime_query(self, user_input: str) -> bool:
        """Detect if the query is about real-time/current information (news, weather, scores, finance, live events, etc.)."""
        user_input_lower = user_input.lower().strip()
        # Core keywords and phrases
        realtime_keywords = [
            'latest', 'breaking', 'today', 'current', 'now', 'recent', 'news', 'score', 'scores', 'weather', 'temperature',
            'forecast', 'update', 'trend', 'headline', 'event', 'ongoing', 'live', 'result', 'results', 'match', 'report',
            'stock', 'stocks', 'market', 'price', 'prices', 'crypto', 'bitcoin', 'ethereum', 'exchange rate', 'currency',
            'sports', 'game', 'games', 'tournament', 'fixture', 'fixtures', 'schedule', 'standing', 'standings', 'table',
            'winner', 'loser', 'draw', 'tie', 'goal', 'run', 'wicket', 'innings', 'quarter', 'half-time', 'full-time',
            'breaking news', 'just in', 'announced', 'released', 'alert', 'emergency', 'advisory', 'traffic', 'accident',
            'earthquake', 'storm', 'rain', 'flood', 'fire', 'disaster', 'crash', 'shutdown', 'outage', 'power cut',
            'covid', 'pandemic', 'case count', 'infection', 'vaccine', 'epidemic', 'epicenter', 'lockdown', 'restriction',
            'election', 'vote', 'poll', 'result', 'results', 'winner', 'loser', 'turnout', 'exit poll', 'referendum',
            'award', 'oscar', 'nobel', 'winner', 'nominee', 'laureate', 'medal', 'gold', 'silver', 'bronze',
            'weather today', 'temperature now', 'live score', 'live update', 'live results', 'who won', 'who is winning',
            'who scored', 'who scored the goal', 'who scored the run', 'who took the wicket', 'who is leading',
            'what is the score', 'what is the weather', 'what is the temperature', 'what happened', 'what is happening',
            'what is trending', 'what is new', 'what is the latest', 'show me the latest', 'show me the news',
            'tell me the news', 'tell me the latest', 'give me the latest', 'give me the news',
        ]
        # Regex patterns for common real-time question forms
        realtime_patterns = [
            r"what(?:'s| is) the (score|weather|temperature|news|result|update|trend|price|market|exchange rate|crypto|winner|standings|table|case count|poll|vote|turnout|medal|award|traffic|alert|advisory|emergency|situation|happening|latest)",
            r"who (won|is winning|scored|is leading|took the wicket|scored the goal|scored the run)",
            r"show me (the|all)? (latest|news|results|scores|updates|trends|prices|standings|fixtures|schedule|weather|temperature)",
            r"tell me (the|all)? (latest|news|results|scores|updates|trends|prices|standings|fixtures|schedule|weather|temperature)",
            r"give me (the|all)? (latest|news|results|scores|updates|trends|prices|standings|fixtures|schedule|weather|temperature)",
            r".*\b(live|breaking|just in|ongoing|current|today|now)\b.*",
        ]
        # Keyword/phrase match
        for kw in realtime_keywords:
            if kw in user_input_lower:
                return True
        # Regex pattern match
        for pat in realtime_patterns:
            if re.search(pat, user_input_lower):
                return True
        return False
    
    def _extract_date(self, text: str) -> datetime.datetime:
        """Try to extract a date from a text snippet, URL, or natural language expression."""
        now = datetime.datetime.now(pytz.utc)
        text_lower = text.lower()
        # Handle natural language expressions
        if any(kw in text_lower for kw in ['just now', 'right now', 'moments ago', 'seconds ago']):
            return now
        if 'minute ago' in text_lower or 'minutes ago' in text_lower:
            try:
                mins = int(re.search(r'(\d+) minute', text_lower).group(1))
                return now - datetime.timedelta(minutes=mins)
            except Exception:
                return now - datetime.timedelta(minutes=5)
        if 'hour ago' in text_lower or 'hours ago' in text_lower:
            try:
                hrs = int(re.search(r'(\d+) hour', text_lower).group(1))
                return now - datetime.timedelta(hours=hrs)
            except Exception:
                return now - datetime.timedelta(hours=1)
        if 'today' in text_lower:
            return now.replace(hour=0, minute=0, second=0, microsecond=0)
        if 'yesterday' in text_lower:
            return (now - datetime.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        if 'this morning' in text_lower:
            return now.replace(hour=8, minute=0, second=0, microsecond=0)
        if 'last night' in text_lower:
            return (now - datetime.timedelta(days=1)).replace(hour=22, minute=0, second=0, microsecond=0)
        if 'this afternoon' in text_lower:
            return now.replace(hour=15, minute=0, second=0, microsecond=0)
        if 'this evening' in text_lower:
            return now.replace(hour=19, minute=0, second=0, microsecond=0)
        # Look for date patterns (e.g., 2024-06-23, June 23, 2024, etc.)
        date_patterns = [
            r'(\d{4}-\d{2}-\d{2})',  # 2024-06-23
            r'(\d{2}/\d{2}/\d{4})',  # 06/23/2024
            r'(\d{1,2} [A-Za-z]+ \d{4})',  # 23 June 2024
            r'([A-Za-z]+ \d{1,2}, \d{4})',  # June 23, 2024
            r'(\d{1,2} [A-Za-z]+)',  # 23 June
            r'([A-Za-z]+ \d{1,2})',  # June 23
        ]
        for pat in date_patterns:
            match = re.search(pat, text)
            if match:
                try:
                    # If year is missing, assume current year
                    date_str = match.group(1)
                    if re.match(r'\d{1,2} [A-Za-z]+$', date_str) or re.match(r'[A-Za-z]+ \d{1,2}$', date_str):
                        date_str += f' {now.year}'
                    return date_parser.parse(date_str, fuzzy=True, default=now)
                except Exception:
                    continue
        # Fallback: if 'live' or 'current' in text, treat as now
        if 'live' in text_lower or 'current' in text_lower:
            return now
        return None
    
    def _remove_duplicate_words(self, text: str) -> str:
        """Remove consecutive duplicate words and phrases from a string."""
        import re
        # Remove consecutive duplicate words
        text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text, flags=re.IGNORECASE)
        # Remove consecutive duplicate phrases (up to 3 words)
        text = re.sub(r'\b(\w+ \w+)( \1\b)+', r'\1', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(\w+ \w+ \w+)( \1\b)+', r'\1', text, flags=re.IGNORECASE)
        return text

    def _should_force_realtime(self, user_input: str) -> bool:
        """Detect if the user is clarifying for real-time data in a follow-up."""
        # Expanded list of clarifiers and expressions
        realtime_clarifiers = [
            'today', 'live', 'current', 'ongoing', 'now', 'latest', 'recent', 'right now', 'just now',
            'this moment', 'this minute', 'this hour', 'this evening', 'this morning', 'this afternoon',
            'this week', 'this month', 'this year', 'tonight', 'this season', 'this game', 'this match',
            'newest', 'fresh', 'breaking', 'update', 'updates', 'real-time', 'real time', 'up-to-date',
            'as of now', 'as of today', 'as of this moment', 'as of this minute', 'as of this hour',
            'any update', 'any news', 'any change', 'any result', 'any score', 'any report',
            'what about now', 'what about today', 'what about live', 'what about the latest',
            'show me now', 'show me today', 'show me live', 'show me the latest',
            'tell me now', 'tell me today', 'tell me live', 'tell me the latest',
            'give me now', 'give me today', 'give me live', 'give me the latest',
            'is it live', 'is it happening', 'is it current', 'is it ongoing',
            'happening now', 'going on now', 'going on today', 'going on live',
            'currently', 'in progress', 'in real time', 'in real-time',
            'just updated', 'just released', 'just announced', 'just published',
            'breaking news', 'breaking update', 'breaking report',
        ]
        user_input_lower = user_input.lower().strip()
        # Keyword/phrase match
        for kw in realtime_clarifiers:
            if kw in user_input_lower:
                return True
        # Regex patterns for short/ambiguous real-time follow-ups
        realtime_followup_patterns = [
            r"^(and )?(now|today|live|latest|current|update|updates|news|score|scores|result|results|report|reports|happening|ongoing)[\?\.! ]*$",  # e.g., "now?", "and now?", "latest.", "update!"
            r"^what about (now|today|live|latest|current|update|updates|news|score|scores|result|results|report|reports|happening|ongoing)[\?\.! ]*$",
            r"^any (update|news|change|result|score|report)[\?\.! ]*$",
            r"^show (me )?(now|today|live|latest|current)[\?\.! ]*$",
            r"^tell (me )?(now|today|live|latest|current)[\?\.! ]*$",
            r"^give (me )?(now|today|live|latest|current)[\?\.! ]*$",
        ]
        for pat in realtime_followup_patterns:
            if re.match(pat, user_input_lower):
                return True
        return False
    
    def _extract_entities(self, text: str) -> dict:
        """
        Extracts key entities (teams, sport, event, date) from the input text.
        This can be replaced with spaCy or a more advanced NER for production.
        """
        teams = re.findall(r'\b(India|England|Australia|Pakistan|South Africa|New Zealand|Sri Lanka|Bangladesh|West Indies)\b', text, re.I)
        sport = None
        if re.search(r'cricket', text, re.I):
            sport = 'cricket'
        elif re.search(r'football|soccer', text, re.I):
            sport = 'football'
        # Add more sports/entities as needed
        event = None
        if re.search(r'match|game|score|result', text, re.I):
            event = 'match'
        # Date/time
        date = None
        if re.search(r'today|live|now|current', text, re.I):
            date = 'today'
        return {
            'teams': list(set(teams)),
            'sport': sport,
            'event': event,
            'date': date
        }

    # --- Contextual Query Builder ---
    def _build_contextual_query(self, user_input: str, context: dict) -> str:
        """
        If this is a follow-up/real-time query, merge with last known entities to build a specific Brave query.
        """
        if self._should_force_realtime(user_input) or self._is_realtime_query(user_input):
            entities = context.get('metadata', {}).get('entities', {})
            context_str = ' '.join(
                filter(None, [entities.get('sport', ''), ' vs '.join(entities.get('teams', [])), entities.get('event', ''), entities.get('date', '')])
            ).strip()
            if context_str:
                return f"{context_str} {user_input}".strip()
        return user_input

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
            'attempt_time': datetime.datetime.utcnow().isoformat()
        }
        
        # Special handling for specific agent types
        if agent_name == 'researcher':
            context['search_performed'] = True
            context['max_iterations'] = min(context.get('max_iterations', 8), 10)
            logger.debug("[Orchestrator] Enabled search for researcher agent")
        
        try:
            # Refine BraveSearchTool query for real-time requests
            force_realtime = self._is_realtime_query(user_input) or self._should_force_realtime(user_input)
            if force_realtime:
                brave_tool = BraveSearchTool()
                # --- Use context to augment Brave query for follow-ups ---
                query = self._build_contextual_query(user_input, context)
                if not any(kw in query.lower() for kw in ['today', 'live', 'current', 'now']):
                    query += ' today'
                brave_result = await brave_tool._arun(query)
                if isinstance(brave_result, dict) and 'error' not in brave_result:
                    web_results = brave_result.get('web', {}).get('results', [])
                    # Sort/filter by date if possible
                    dated_results = []
                    for item in web_results:
                        # Try to extract date from snippet, title, or url
                        date_candidates = [item.get('description', ''), item.get('title', ''), item.get('url', '')]
                        item_date = None
                        for candidate in date_candidates:
                            d = self._extract_date(candidate)
                            if d:
                                item_date = d
                                break
                        if item_date:
                            dated_results.append((item_date, item))
                    # If we found any dated results, sort by date (descending)
                    if dated_results:
                        dated_results.sort(reverse=True, key=lambda x: x[0])
                        web_results = [item for _, item in dated_results]
                    # Try to find today's result
                    today = datetime.datetime.now().date()
                    top = None
                    for item in web_results:
                        for candidate in [item.get('description', ''), item.get('title', ''), item.get('url', '')]:
                            d = self._extract_date(candidate)
                            if d and d.date() == today:
                                top = item
                                break
                        if top:
                            break
                    # If no 'today' result, offer most recent and inform user
                    if not top and web_results:
                        top = web_results[0]
                        # Find date for most recent
                        most_recent_date = None
                        for candidate in [top.get('description', ''), top.get('title', ''), top.get('url', '')]:
                            d = self._extract_date(candidate)
                            if d:
                                most_recent_date = d.date()
                                break
                        date_str = f" (most recent: {most_recent_date})" if most_recent_date else " (most recent)"
                        output = f"No results found for today. Showing the most recent result{date_str}:\n\n**{top.get('title', 'Result')}**\n{top.get('description', '')}\n{top.get('url', '')}"
                    elif top:
                        output = f"**{top.get('title', 'Result')}**\n{top.get('description', '')}\n{top.get('url', '')}"
                    else:
                        output = "No relevant results found from Brave Search."
                    # Contextual follow-up logic
                    followup = ""
                    if len(web_results) > 1:
                        q = user_input.lower()
                        titles = ' '.join([r.get('title', '').lower() for r in web_results[:5]])
                        followup_options = []
                        if any(x in q or x in titles for x in ['women', "women's"]):
                            followup_options.append("Do you want Women's or Men's match?")
                        elif 'men' in q or "men's" in titles:
                            followup_options.append("Do you want Men's or Women's match?")
                        if 'highlight' in q or 'highlight' in titles:
                            followup_options.append("Are you looking for highlights or live scorecard?")
                        if 'score' in q or 'live' in q or 'scorecard' in titles:
                            followup_options.append("Do you want the live score, full scorecard, or match summary?")
                        if not followup_options:
                            followup_options.append("Can you specify if you want news, scores, weather, or something else? Or would you like to see more results?")
                        followup = '\n\n' + ' '.join(followup_options)
                    # CLEANING STEP: Remove duplicate words/phrases from both output and followup
                    output = self._remove_duplicate_words(output)
                    followup = self._remove_duplicate_words(followup)
                    output += followup
                    return {
                        'success': True,
                        'output': output,
                        'agent_used': 'brave_search',
                        'metadata': {'tool_used': 'brave_search', 'raw_result': brave_result, 'num_results': len(web_results)}
                    }
                else:
                    error_msg = brave_result.get('error', 'Unknown error from Brave Search.') if isinstance(brave_result, dict) else str(brave_result)
                    return {
                        'success': False,
                        'output': f"Brave Search error: {error_msg}",
                        'agent_used': 'brave_search',
                        'metadata': {'tool_used': 'brave_search', 'error': error_msg}
                    }
            
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
            history = await self._get_formatted_history(conversation_id, user_id)
            
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
                agent_response = agent.process(**process_kwargs)
                
                # Ensure we await the coroutine if it is one
                if hasattr(agent_response, '__await__'):
                    response = await asyncio.wait_for(
                        agent_response,
                        timeout=60  # 60 second timeout
                    )
                else:
                    response = agent_response
                
                # Log the response
                processing_time = time.time() - start_time
                logger.info(f"[Orchestrator] {agent_name} agent completed in {processing_time:.2f}s")
                
                # Ensure response is a dictionary
                if not isinstance(response, dict):
                    response = {
                        'output': str(response) if response is not None else 'No response from agent',
                        'metadata': {}
                    }
                
                # Add processing metadata
                if 'metadata' not in response:
                    response['metadata'] = {}
                    
                # Ensure metadata values are JSON serializable
                metadata = {
                    'processing_time_seconds': float(processing_time),
                    'agent_used': agent_name,
                    'model': str(getattr(agent, 'model_name', 'unknown')),
                    'success': response.get('success', True)
                }
                response['metadata'].update(metadata)
                
                # Ensure output is a string
                if 'output' in response and not isinstance(response['output'], str):
                    if hasattr(response['output'], '__await__'):
                        response['output'] = await response['output']
                    response['output'] = str(response['output'])
                
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
                chat_history=await self._get_formatted_history(conversation_id, user_id)
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

    async def _update_history(
        self,
        user_input: str,
        assistant_response: str = None,
        conversation_id: str = "default",
        user_id: str = "default"
    ) -> None:
        """Update the conversation history in memory."""
        try:
            # Get the current conversation state
            memory = await memory_manager.get_conversation(conversation_id, user_id) or {}
            messages = memory.get('messages', [])
            
            # Add user message
            messages.append({
                'role': 'user',
                'content': user_input,
                'timestamp': datetime.datetime.utcnow().isoformat()
            })
            
            # Add assistant response if provided
            if assistant_response is not None:
                # Ensure the response is a string, not a coroutine or other non-serializable type
                if hasattr(assistant_response, '__await__'):
                    assistant_response = await assistant_response
                
                # If it's a dict, extract the output
                if isinstance(assistant_response, dict) and 'output' in assistant_response:
                    assistant_response = assistant_response['output']
                
                # Convert to string if needed
                if not isinstance(assistant_response, str):
                    assistant_response = str(assistant_response)
                
                messages.append({
                    'role': 'assistant',
                    'content': assistant_response,
                    'timestamp': datetime.datetime.utcnow().isoformat()
                })
            
            # Ensure messages are JSON serializable
            def make_serializable(obj):
                if isinstance(obj, (str, int, float, bool, type(None))):
                    return obj
                elif isinstance(obj, dict):
                    return {k: make_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [make_serializable(item) for item in obj]
                else:
                    return str(obj)
            
            serialized_messages = make_serializable(messages)
            
            # Update memory
            await memory_manager.save_conversation(
                conversation_id=conversation_id,
                user_id=user_id,
                messages=serialized_messages,
                metadata=memory.get('metadata', {})
            )
        except Exception as e:
            logger.error(f"Error updating conversation history: {str(e)}", exc_info=True)
            # Don't raise the error to avoid breaking the main flow
            pass
    
    async def _get_formatted_history(
        self,
        conversation_id: str,
        user_id: str,
        max_messages: int = 5
    ) -> str:
        """Get formatted conversation history for agent input.
        
        Args:
            conversation_id: ID of the conversation
            user_id: ID of the user
            max_messages: Maximum number of messages to return
            
        Returns:
            Formatted conversation history as a string
        """
        try:
            # Get conversation history from memory
            memory = await memory_manager.get_conversation(conversation_id, user_id) or {}
            messages = memory.get('messages', [])
            
            def safe_get(obj, key, default=''):
                """Safely get a value from a dictionary, converting to string if needed."""
                value = obj.get(key, default)
                if value is None:
                    return ''
                return str(value)
            
            formatted = []
            for msg in messages[-max_messages:]:  # Only get the most recent messages
                try:
                    if not isinstance(msg, dict):
                        continue
                    role = safe_get(msg, 'role', 'unknown')
                    content = safe_get(msg, 'content', '')
                    if not isinstance(content, str):
                        content = str(content)
                    # Truncate very long messages to prevent context overflow
                    if len(content) > 500:
                        content = content[:495] + '...'
                    formatted.append(f"{role}: {content}")
                except Exception as msg_err:
                    logger.warning(f"Error formatting message: {msg_err}", exc_info=True)
                    continue
            joined = "\n".join(formatted) if formatted else "No conversation history available."
            # Hard cap: if total exceeds 4000 chars, truncate from the start
            if len(joined) > 4000:
                joined = joined[-4000:]
            return joined
        except Exception as e:
            logger.error(f"Error getting conversation history: {str(e)}", exc_info=True)
            return "Error: Could not load conversation history."

    async def _process_with_domain_agent(
        self,
        agent_name: str,
        user_input: str,
        context: Dict[str, Any],
        conversation_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """Process a query with one of the domain agents with enhanced error handling and logging."""
        logger.info(f"[Orchestrator] Processing with domain agent: {agent_name}")
        start_time = time.time()
        # Validate agent exists
        if agent_name not in self.domain_agents:
            error_msg = f"Unknown domain agent: {agent_name}. Available agents: {list(self.domain_agents.keys())}"
            logger.error(error_msg)
            return {
                'success': False,
                'output': f"I couldn't find the '{agent_name}' domain agent to handle your request.",
                'error': error_msg,
                'agent_used': agent_name,
                'metadata': {
                    'processing_time_seconds': time.time() - start_time,
                    'error_type': 'agent_not_found'
                }
            }
        agent = self.domain_agents[agent_name]
        try:
            response = await agent.process(
                input_text=user_input,
                context=context,
                conversation_id=conversation_id,
                user_id=user_id
            )
            # Ensure response is a dictionary
            if not isinstance(response, dict):
                response = {
                    'output': str(response) if response is not None else 'No response from agent',
                    'metadata': {}
                }
            # Add processing metadata
            if 'metadata' not in response:
                response['metadata'] = {}
            response['metadata'].update({
                'processing_time_seconds': float(time.time() - start_time),
                'agent_used': agent_name,
                'success': response.get('success', True)
            })
            # Ensure output is a string
            if 'output' in response and not isinstance(response['output'], str):
                response['output'] = str(response['output'])
            return response
        except Exception as e:
            error_msg = f"Error in {agent_name} domain agent: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                'success': False,
                'output': f"I encountered an error while processing your request with the {agent_name} domain agent.",
                'error': error_msg,
                'agent_used': agent_name,
                'metadata': {
                    'processing_time_seconds': time.time() - start_time,
                    'error_type': type(e).__name__
                }
            }

# ... (rest of the code remains the same)
