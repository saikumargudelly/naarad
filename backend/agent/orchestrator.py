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
import os
from collections import OrderedDict

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
from .memory.conversation_memory import ConversationMemory

# Lazy imports to avoid circular dependencies
_domain_agents = None

def _get_domain_agents():
    global _domain_agents
    if _domain_agents is None:
        from .domain_agents import create_domain_agents
        _domain_agents = create_domain_agents()
    return _domain_agents

logger = logging.getLogger(__name__)

class OrchestratorAgent:
    def __init__(self, agents: Dict[str, Any], memory_manager, metrics=None, router=None):
        ...
    # All methods, logger lines, and comments for OrchestratorAgent

class AgentOrchestrator:
    def __init__(self, base_agents: Dict[str, AgentExecutor] = None, metrics=None, router=None):
        """Initialize the orchestrator with base agents and domain agents.
        
        Args:
            base_agents: Optional dictionary of base agents. If not provided,
                       an empty dict will be used.
            metrics: Optional metrics collector for testability/monitoring.
            router: Optional intent router for testability.
        """
        self.base_agents = base_agents or {}
        self.domain_agents = _get_domain_agents()
        self.conversation_history = []
        self.current_topic = None
        self.last_interaction_time = datetime.datetime.utcnow()
        self.metrics = metrics  # For testability/monitoring
        self.router = router    # For testability
        # self.health_status = 'ok'  # For health endpoint
        # Load semantic model for filtering
        try:
            from sentence_transformers import SentenceTransformer, util
            self._semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            self._semantic_model = None
    
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
            
            # --- Intent classification (unified) ---
            from .enhanced_router import EnhancedRouter
            router = self.router or EnhancedRouter()
            intent_match = await router.classify_intent(user_input, context)
            intent = intent_match.intent.value if hasattr(intent_match.intent, 'value') else str(intent_match.intent)
            confidence = intent_match.confidence
            entities = {k: v.value for k, v in intent_match.entities.items()}
            # Standardize context keys
            context['routing_info'] = {
                'intent': intent,
                'confidence': confidence,
                'entities': entities,
                'raw_intent_match': intent_match
            }
            context['entities'] = entities
            # Add context trace (breadcrumbs)
            trace = context.get('trace', [])
            trace.append({
                'step': 'intent_classification',
                'intent': intent,
                'confidence': confidence,
                'entities': entities,
                'timestamp': datetime.datetime.utcnow().isoformat()
            })
            context['trace'] = trace
            # --- Multi-agent collaboration logic ---
            collaborative_agents = []
            multi_agent = False
            # Check for multiple strong intents (alternatives with high confidence)
            alternatives = getattr(intent_match, 'alternatives', [])
            strong_alts = [alt for alt in alternatives if alt[1] > 0.7 and alt[0] != intent_match.intent]
            # Also check for multi-domain keywords (e.g., both "analyze" and "predict" in input)
            multi_domain_keywords = [
                ('analyze', 'predict'),
                ('research', 'analyze'),
                ('emotion', 'quality'),
                ('summarize', 'personalize'),
                # Expanded pairs for more robust multi-agent collaboration:
                ('summarize', 'analyze'),
                ('summarize', 'predict'),
                ('summarize', 'quality'),
                ('summarize', 'emotion'),
                ('summarize', 'personalize'),
                ('analyze', 'quality'),
                ('analyze', 'emotion'),
                ('analyze', 'research'),
                ('analyze', 'forecast'),
                ('analyze', 'trend'),
                ('analyze', 'classify'),
                ('predict', 'trend'),
                ('predict', 'quality'),
                ('predict', 'personalize'),
                ('predict', 'summarize'),
                ('predict', 'analyze'),
                ('predict', 'research'),
                ('predict', 'emotion'),
                ('quality', 'emotion'),
                ('quality', 'personalize'),
                ('quality', 'summarize'),
                ('quality', 'analyze'),
                ('quality', 'predict'),
                ('personalize', 'summarize'),
                ('personalize', 'analyze'),
                ('personalize', 'predict'),
                ('personalize', 'quality'),
                ('personalize', 'emotion'),
                ('emotion', 'summarize'),
                ('emotion', 'analyze'),
                ('emotion', 'predict'),
                ('emotion', 'quality'),
                ('emotion', 'personalize'),
                ('compare', 'analyze'),
                ('compare', 'summarize'),
                ('compare', 'predict'),
                ('compare', 'quality'),
                ('compare', 'emotion'),
                ('compare', 'personalize'),
                ('trend', 'forecast'),
                ('trend', 'analyze'),
                ('trend', 'predict'),
                ('trend', 'summarize'),
                ('trend', 'quality'),
                ('trend', 'personalize'),
                ('trend', 'emotion'),
                ('forecast', 'analyze'),
                ('forecast', 'predict'),
                ('forecast', 'summarize'),
                ('forecast', 'quality'),
                ('forecast', 'personalize'),
                ('forecast', 'emotion'),
                # Add more as needed for new agent types or cross-domain queries
            ]
            user_input_lower = user_input.lower()
            for kw1, kw2 in multi_domain_keywords:
                if kw1 in user_input_lower and kw2 in user_input_lower:
                    multi_agent = True
                    break
            if strong_alts:
                multi_agent = True
            if multi_agent:
                # Determine all relevant agents
                agent_names = set([await self._select_agent_by_intent(intent, context, user_input)])
                for alt_intent, alt_conf in strong_alts:
                    alt_intent_str = alt_intent.value if hasattr(alt_intent, 'value') else str(alt_intent)
                    agent_names.add(await self._select_agent_by_intent(alt_intent_str, context, user_input))
                # Remove duplicates and responder (unless only fallback)
                agent_names = [a for a in agent_names if a != 'responder'] or ['responder']
                collaborative_agents = list(agent_names)
                # Collect all agent responses in parallel
                agent_tasks = [
                    self._route_to_agent(
                        agent_name=agent_name,
                        user_input=user_input,
                        context={**context, 'collaboration_role': agent_name},
                        conversation_id=conversation_id,
                        user_id=user_id,
                        supporting_agents=[]
                    )
                    for agent_name in collaborative_agents
                ]
                agent_results = await asyncio.gather(*agent_tasks, return_exceptions=True)
                merged_output = ""
                merge_trace = []
                for agent_name, result in zip(collaborative_agents, agent_results):
                    if isinstance(result, Exception):
                        out = f"Error from {agent_name}: {str(result)}"
                    else:
                        out = result.get('output', f"No response from {agent_name}")
                    merged_output += f"\n\n--- [{agent_name.upper()} RESPONSE] ---\n{out.strip()}"
                    merge_trace.append({'step': 'agent_response', 'agent': agent_name, 'output': out, 'timestamp': datetime.datetime.utcnow().isoformat()})
                # Optionally, add a summary (could use a summarizer agent)
                merged_output = merged_output.strip()
                context['trace'].extend(merge_trace)
                context['trace'].append({'step': 'merge', 'agents': collaborative_agents, 'timestamp': datetime.datetime.utcnow().isoformat()})
                # After agent response is generated, post-process it
                if 'output' in result and isinstance(result['output'], str):
                    result['output'] = self._post_process_response(result['output'])
                return {
                    'success': True,
                    'output': merged_output,
                    'agent_used': '+'.join(collaborative_agents),
                    'metadata': {
                        'processing_time_seconds': 0,  # Could sum agent times if needed
                        'intent': intent,
                        'confidence': confidence,
                        'entities': entities,
                        'trace': context['trace'],
                        'collaborative_agents': collaborative_agents
                    }
                }
            # --- Single-agent logic (as before) ---
            # Metrics, fallback, etc. remain unchanged
            # --- Metrics: intent distribution, ambiguous queries, etc. ---
            if self.metrics:
                self.metrics.record_intent(intent, confidence)
                if confidence < 0.6:
                    self.metrics.record_ambiguous_query(user_input, intent, confidence)
            # Log low-confidence queries for review
            if confidence < 0.6:
                logger.warning(f"Low-confidence intent: '{intent}' ({confidence:.2f}) for input: {user_input}")
                # Escalate to responder with special prompt
                clarification = f"I'm not sure what you're asking. Could you clarify your request? (Detected intent: {intent}, confidence: {confidence:.2f})"
                context['trace'].append({'step': 'clarification', 'agent': 'responder', 'timestamp': datetime.datetime.utcnow().isoformat()})
                if self.metrics:
                    self.metrics.record_fallback('clarification')
                # Call responder agent with special prompt
                if 'responder' in self.base_agents:
                    agent = self.base_agents['responder']
                    response = await agent.process(
                        input_text=clarification,
                        context=context,
                        conversation_id=conversation_id,
                        user_id=user_id
                    )
                    # After agent response is generated, post-process it
                    if 'output' in response and isinstance(response['output'], str):
                        response['output'] = self._post_process_response(response['output'])
                    return {
                        'success': False,
                        'output': response.get('output', clarification),
                        'agent_used': 'responder',
                        'metadata': {
                            'processing_time_seconds': 0,
                            'intent': intent,
                            'confidence': confidence,
                            'entities': entities,
                            'trace': context['trace']
                        }
                    }
                else:
                    # After agent response is generated, post-process it
                    if 'output' in response and isinstance(response['output'], str):
                        response['output'] = self._post_process_response(response['output'])
                    return {
                        'success': False,
                        'output': clarification,
                        'agent_used': None,
                        'metadata': {
                            'processing_time_seconds': 0,
                            'intent': intent,
                            'confidence': confidence,
                            'entities': entities,
                            'trace': context['trace']
                        }
                    }
            
            # 1. Analyze context and determine the best agent(s) to handle the query
            agent_name = await self._select_agent_by_intent(intent, context, user_input)
            context['trace'].append({'step': 'agent_selection', 'agent': agent_name, 'timestamp': datetime.datetime.utcnow().isoformat()})
            
            # 2. Process with the selected agent(s)
            response = await self._route_to_agent(
                agent_name=agent_name,
                user_input=user_input,
                context=context,
                conversation_id=conversation_id,
                user_id=user_id,
                supporting_agents=[]
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
                agent_name=agent_name,
                processing_time=processing_time,
                conversation_id=conversation_id
            )
            
            # --- Metrics: agent success/fallback ---
            if self.metrics:
                self.metrics.record_agent_result(agent_name, response.get('success', False))
            
            # After agent response is generated, post-process it
            if 'output' in response and isinstance(response['output'], str):
                response['output'] = self._post_process_response(response['output'])
            return {
                'success': response.get('success', True),
                'output': response.get('output', ''),
                'agent_used': agent_name,
                'metadata': {
                    'processing_time_seconds': processing_time,
                    'intent': intent,
                    'confidence': confidence,
                    'entities': entities,
                    'trace': context['trace']
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
            # --- New intent classification and agent selection logic ---
            from .enhanced_router import EnhancedRouter
            router = self.router or EnhancedRouter()
            context = kwargs or {}
            intent_match = await router.classify_intent(user_input, context)
            intent = intent_match.intent.value if hasattr(intent_match.intent, 'value') else str(intent_match.intent)
            confidence = intent_match.confidence
            entities = {k: v.value for k, v in intent_match.entities.items()}
            # Standardize context keys
            context['routing_info'] = {
                'intent': intent,
                'confidence': confidence,
                'entities': entities,
                'raw_intent_match': intent_match
            }
            context['entities'] = entities
            trace = context.get('trace', [])
            trace.append({
                'step': 'intent_classification',
                'intent': intent,
                'confidence': confidence,
                'entities': entities,
                'timestamp': datetime.datetime.utcnow().isoformat()
            })
            context['trace'] = trace
            primary_agent = await self._select_agent_by_intent(intent, context, user_input)
            context['trace'].append({'step': 'agent_selection', 'agent': primary_agent, 'timestamp': datetime.datetime.utcnow().isoformat()})
            # Yield typing start indicator
            yield {"type": "typing_start", "agent": primary_agent}
            # 2. Route to the appropriate agent's streaming method if it exists
            if primary_agent in self.domain_agents and hasattr(self.domain_agents[primary_agent], 'process_stream'):
                agent = self.domain_agents[primary_agent]
                async for chunk in agent.process_stream(user_input, context, conversation_id, user_id):
                    yield chunk
            elif primary_agent in self.base_agents and hasattr(self.base_agents[primary_agent], 'process_stream'):
                agent = self.base_agents[primary_agent]
                async for chunk in agent.process_stream(user_input, context, conversation_id, user_id):
                    yield chunk
            else:
                # Fallback to non-streaming for agents that don't support it
                response = await self._route_to_agent(
                    agent_name=primary_agent,
                    user_input=user_input,
                    context=context,
                    conversation_id=conversation_id,
                    user_id=user_id
                )
                yield {
                    "type": "message",
                    "content": response.get('output', 'No response generated.'),
                    "agent": primary_agent
                }
            # After agent response is generated, post-process it
            if 'output' in response and isinstance(response['output'], str):
                response['output'] = self._post_process_response(response['output'])
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
    
    async def _select_agent_by_intent(self, intent: str, context: Dict[str, Any], user_input: str) -> str:
        """Select the best agent for the given intent, with minimal keyword-based fallback for 'unknown' intent only."""
        # Minimal fallback for 'unknown' intent
        if intent == "unknown":
            realtime_keywords = [
                "news", "headline", "current event", "breaking", "top", "latest", "update", "today", "now", "search", "find", "lookup", "web", "internet", "article", "report", "trend",
                "match", "game", "score", "schedule", "event", "when is", "next", "upcoming", "live", "result", "fixture", "tournament", "series", "cricket", "football", "olympics", "world cup", "sports"
            ]
            user_input_lc = user_input.lower() if isinstance(user_input, str) else ''
            if any(kw in user_input_lc for kw in realtime_keywords):
                return "researcher"
        # Standard mapping
        mapping = {
            'research': 'researcher',
            'search': 'researcher',
            'analyze': 'analyst',
            'personalize': 'personalization_agent',
            'predict': 'prediction_agent',
            'learn': 'learning_agent',
            'quantum': 'quantum_agent',
            'creativity': 'creativity_agent',
            'emotion': 'emotion_agent',
            'quality': 'quality_agent',
            'voice': 'voice_agent',
            'analytics': 'analytics_agent',
            'respond': 'responder',
        }
        return mapping.get(intent, 'responder')
    
    def _is_realtime_query(self, user_input: str) -> bool:
        logger.info(f"[DEBUG] _is_realtime_query called with: {user_input}")
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
            r"give me (me )?(now|today|live|latest|current)[\?\.! ]*$",
            r".*\b(live|breaking|just in|ongoing|current|today|now)\b.*",
        ]
        # Keyword/phrase match
        for kw in realtime_keywords:
            if kw in user_input_lower:
                logger.info(f"[DEBUG] _is_realtime_query returning True (matched keyword: {kw})")
                return True
        # Regex pattern match
        for pat in realtime_patterns:
            if re.search(pat, user_input_lower):
                logger.info(f"[DEBUG] _is_realtime_query returning True (matched pattern: {pat})")
                return True
        logger.info(f"[DEBUG] _is_realtime_query returning False")
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
        logger.info(f"[DEBUG] _should_force_realtime called with: {user_input}")
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
                logger.info(f"[DEBUG] _should_force_realtime returning True (matched clarifier: {kw})")
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
                logger.info(f"[DEBUG] _should_force_realtime returning True (matched pattern: {pat})")
                return True
        logger.info(f"[DEBUG] _should_force_realtime returning False")
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
        Dynamically build a concise, optimized query for real-time search APIs.
        - Use all extracted entities (teams, event, location, date, etc.)
        - Leverage detected intent and previous conversation context
        - Remove stopwords and redundant words
        - Truncate to 100 chars at word boundaries
        - Designed for future extensibility (e.g., custom entity types, domain-specific logic)
        """
        import datetime, re
        from collections import OrderedDict

        year = str(datetime.datetime.now().year)
        entities = context.get('metadata', {}).get('entities', {})
        intent = context.get('routing_info', {}).get('intent', '')
        prev_context = context.get('previous_user_inputs', [])

        # Gather all possible keywords/entities
        keywords = []

        # Add all entity values dynamically (future-proof: supports any entity type)
        for key, value in entities.items():
            if value:
                if isinstance(value, list):
                    keywords.extend([str(v) for v in value])
                else:
                    keywords.append(str(value))

        # Add intent if it's not generic
        if intent and intent not in ['unknown', 'general', 'other'] and intent not in keywords:
            keywords.append(intent)

        # Add previous context if available (last 2 user turns)
        for prev in prev_context[-2:]:
            prev_words = [w for w in re.findall(r'\w+', prev.lower()) if len(w) > 2]
            keywords.extend(prev_words)

        # Add user input words, excluding stopwords
        stopwords = set([
            'the', 'is', 'in', 'at', 'on', 'for', 'to', 'of', 'and', 'a', 'an', 'with', 'between',
            'show', 'me', 'check', 'find', 'tell', 'give', 'now', 'today', 'upcoming', 'please', 'can', 'could', 'would', 'should', 'when', 'where', 'how', 'what', 'which', 'who', 'whom', 'whose', 'about', 'from', 'by', 'as', 'it', 'this', 'that', 'these', 'those', 'do', 'does', 'did', 'will', 'shall', 'may', 'might', 'must', 'let', 'us', 'want', 'like', 'need', 'get', 'see', 'latest', 'current', 'recent', 'news', 'info', 'information', 'details', 'update', 'updates', 'result', 'results', 'report', 'reports', 'event', 'events', 'schedule', 'schedules', 'match', 'matches', 'game', 'games', 'score', 'scores', 'live', 'today', 'now', 'upcoming', 'next', 'year', 'month', 'week', 'day', 'date', 'time', 'vs', 'vs.'
        ])
        user_words = [w for w in re.findall(r'\w+', user_input.lower()) if w not in stopwords and len(w) > 2]
        keywords.extend(user_words)

        # Remove duplicates, preserve order
        keywords = list(OrderedDict.fromkeys(keywords))

        # Always add year if not present
        if year not in keywords:
            keywords.append(year)

        # Build query string
        query = ' '.join(keywords)
        query = query.replace('"', '').replace("'", '').strip()
        query = re.sub(r'\s+', ' ', query)

        # Truncate at word boundary
        if len(query) > 100:
            query = query[:100]
            if ' ' in query:
                query = query[:query.rfind(' ')]

        logger.info(f"[RealTime] Dynamic search query: {query}")
        return query

    # --- Context Updater for Previous User Inputs ---
    def _update_context_with_user_input(self, context: dict, user_input: str) -> dict:
        """
        Update the context with the latest user input for richer, multi-turn query construction.
        Maintains a rolling window of previous user inputs (last 5 turns).
        """
        prev_inputs = context.get('previous_user_inputs', [])
        prev_inputs.append(user_input)
        if len(prev_inputs) > 5:
            prev_inputs = prev_inputs[-5:]
        context['previous_user_inputs'] = prev_inputs
        return context

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
        try:
            logger.info(f"[Orchestrator] Routing to agent: {agent_name}" +
                       (f" with {len(supporting_agents or [])} supporting agents" if supporting_agents else ""))
            supporting_agents = supporting_agents or []
            start_time = time.time()
            context = context.copy()
            context['routing'] = {
                'primary_agent': agent_name,
                'supporting_agents': supporting_agents,
                'attempt_time': datetime.datetime.utcnow().isoformat()
            }
            # --- Robust Real-Time Routing for ALL Agents ---
            # Enhanced: Always try Brave/GNews for real-time and recommendation-related intents
            real_time_intents = [
                'weather', 'sports', 'news', 'trending', 'recommendation', 'music', 'movie', 'entertainment',
                'prediction', 'search', 'event', 'analytics', 'quality', 'calendar', 'reminder'
            ]
            intent = context.get('routing_info', {}).get('intent', '')
            force_realtime = (
                self._is_realtime_query(user_input) or
                self._should_force_realtime(user_input) or
                intent in real_time_intents
            )
            if force_realtime:
                query = self._build_contextual_query(user_input, context)
                logger.info(f"[RealTime] Real-time query constructed: {query}")
                gnews_result, brave_result = None, None
                articles, web_results = [], []
                brave_summary = None
                try:
                    from .tools.gnews_search import GNewsSearchTool
                    gnews_tool = GNewsSearchTool()
                    gnews_result = await gnews_tool._arun(query)
                    if 'error' in gnews_result:
                        logger.warning(f"[RealTime] GNews API error: {gnews_result['error']}")
                    articles = gnews_result.get('articles', []) if gnews_result else []
                except Exception as e:
                    logger.error(f"[RealTime] Exception during GNews search: {e}", exc_info=True)
                try:
                    from .tools.brave_search import BraveSearchTool
                    brave_tool = BraveSearchTool()
                    brave_result = await brave_tool._arun(query)
                    if 'error' in brave_result:
                        logger.warning(f"[RealTime] Brave API error: {brave_result['error']}")
                    web_results = brave_result.get('web', {}).get('results', []) if isinstance(brave_result, dict) else []
                    # Check for a direct answer or summary in Brave's response
                    brave_summary = brave_result.get('summary') if isinstance(brave_result, dict) else None
                    if not brave_summary and web_results:
                        # Try to extract a summary-like paragraph from the first result
                        first = web_results[0]
                        brave_summary = first.get('description') or first.get('snippet')
                except Exception as e:
                    logger.error(f"[RealTime] Exception during Brave search: {e}", exc_info=True)
                now = datetime.datetime.now()
                filtered_articles = []
                for a in articles:
                    date_str = a.get('publishedAt') or a.get('date') or ''
                    try:
                        article_date = datetime.datetime.fromisoformat(date_str.replace('Z', '+00:00')) if date_str else None
                    except Exception:
                        article_date = None
                    if article_date and (article_date.year == now.year or (now - article_date).days <= 180):
                        filtered_articles.append(a)
                filtered_results = []
                for r in web_results:
                    date_str = r.get('datePublished') or r.get('date') or ''
                    try:
                        result_date = datetime.datetime.fromisoformat(date_str.replace('Z', '+00:00')) if date_str else None
                    except Exception:
                        result_date = None
                    if result_date and (result_date.year == now.year or (now - result_date).days <= 180):
                        filtered_results.append(r)
                # Apply live filtering and semantic ranking
                live_articles = self.filter_live_results(user_input, filtered_articles)
                live_web_results = self.filter_live_results(user_input, filtered_results)
                filtered_articles = self.semantic_filter_and_rank(user_input, live_articles) if live_articles else self.semantic_filter_and_rank(user_input, filtered_articles)
                filtered_results = self.semantic_filter_and_rank(user_input, live_web_results) if live_web_results else self.semantic_filter_and_rank(user_input, filtered_results)

                # If we have a direct answer/summary from GNews, present it directly and optionally use LLM for context
                gnews_summary = None
                if filtered_articles:
                    first_article = filtered_articles[0]
                    gnews_summary = first_article.get('description') or first_article.get('content')
                if gnews_summary:
                    from langchain_groq import ChatGroq
                    from langchain_core.messages import SystemMessage, HumanMessage
                    from llm.config import settings
                    system_prompt = (
                        "You are an assistant. The following is a direct answer from a real-time news article. If helpful, add a brief, creative, or contextual expansion, but do not replace or contradict the direct answer. Only use the most recent and relevant information. If no live update is found, say so clearly."
                    )
                    messages = [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=f"User question: {user_input}\n\nDirect answer from news article:\n{gnews_summary}")
                    ]
                    llm = ChatGroq(
                        temperature=0.2,
                        model_name=settings.REASONING_MODEL,
                        groq_api_key=os.getenv('GROQ_API_KEY')
                    )
                    result = await llm.ainvoke(messages)
                    logger.info(f"[RealTime] Returning direct GNews answer with LLM context.")
                    return {
                        'success': True,
                        'output': f"{gnews_summary}\n\n{result.content.strip()}",
                        'agent_used': 'gnews_direct_answer',
                        'metadata': {
                            'tool_used': 'gnews_direct_answer',
                            'gnews_result': gnews_result,
                            'num_articles': len(filtered_articles)
                        }
                    }
                # If we have a direct answer/summary from Brave, present it directly and optionally use LLM for context
                if brave_summary:
                    from langchain_groq import ChatGroq
                    from langchain_core.messages import SystemMessage, HumanMessage
                    from llm.config import settings
                    system_prompt = (
                        "You are an assistant. The following is a direct answer from a real-time web search. If helpful, add a brief, creative, or contextual expansion, but do not replace or contradict the direct answer. Only use the most recent and relevant information. If no live update is found, say so clearly."
                    )
                    messages = [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=f"User question: {user_input}\n\nDirect answer from web search:\n{brave_summary}")
                    ]
                    llm = ChatGroq(
                        temperature=0.2,
                        model_name=settings.REASONING_MODEL,
                        groq_api_key=os.getenv('GROQ_API_KEY')
                    )
                    result = await llm.ainvoke(messages)
                    logger.info(f"[RealTime] Returning direct Brave answer with LLM context.")
                    return {
                        'success': True,
                        'output': f"{brave_summary}\n\n{result.content.strip()}",
                        'agent_used': 'brave_direct_answer',
                        'metadata': {
                            'tool_used': 'brave_direct_answer',
                            'brave_result': brave_result,
                            'num_web_results': len(filtered_results)
                        }
                    }
                # If no live or direct answer, use hybrid logic as before, but only pass the most recent and relevant results to the LLM
                context_snippets = ""
                if filtered_articles:
                    context_snippets += "\n\n--- GNews Articles ---\n" + "\n\n".join([
                        f"{a.get('title', '')}\n{a.get('description', '')}\n{a.get('url', '')}\nDate: {a.get('publishedAt', a.get('date', ''))}" for a in filtered_articles[:3]
                    ])
                if filtered_results:
                    context_snippets += "\n\n--- Brave Search Results ---\n" + "\n\n".join([
                        f"{r.get('title', '')}\n{r.get('description', '')}\n{r.get('url', '')}\nDate: {r.get('datePublished', r.get('date', ''))}" for r in filtered_results[:3]
                    ])
                from langchain_groq import ChatGroq
                from langchain_core.messages import SystemMessage, HumanMessage
                from llm.config import settings
                system_prompt = (
                    f"You are a real-time assistant. The user is asking for live or current updates: '{user_input}'. "
                    "Here are the most recent and relevant search results. "
                    "Please provide a concise, direct answer to the user's question, synthesizing only the most important information. "
                    "Do not repeat raw snippets or unrelated details. If there is no live score, state that clearly and briefly. Limit your answer to 2-3 sentences."
                )
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=f"User question: {user_input}")
                ]
                llm = ChatGroq(
                    temperature=0.2,
                    model_name=settings.REASONING_MODEL,
                    groq_api_key=os.getenv('GROQ_API_KEY')
                )
                result = await llm.ainvoke(messages)
                logger.info(f"[RealTime] Returning hybrid real-time + LLM answer (live/semantic filtered).")
                # Post-processing: remove duplicate lines, limit to 2-3 sentences, strip HTML tags
                def postprocess_response(text):
                    # Remove HTML tags
                    text = re.sub(r'<[^>]+>', '', text)
                    # Remove duplicate lines
                    lines = text.split('\n')
                    seen = set()
                    unique_lines = []
                    for line in lines:
                        line = line.strip()
                        if line and line not in seen:
                            unique_lines.append(line)
                            seen.add(line)
                    text = ' '.join(unique_lines)
                    # Limit to 2-3 sentences
                    sentences = re.split(r'(?<=[.!?]) +', text)
                    return ' '.join(sentences[:3])
                concise_output = postprocess_response(result.content.strip())
                return {
                    'success': True,
                    'output': concise_output,
                    'agent_used': 'hybrid_realtime_llm',
                    'metadata': {
                        'tool_used': 'hybrid_realtime_llm',
                        'gnews_result': gnews_result,
                        'brave_result': brave_result,
                        'num_articles': len(filtered_articles),
                        'num_web_results': len(filtered_results)
                    }
                }
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
            
        except Exception as e:
            logger.error(f"Error in agent routing for {agent_name}: {str(e)}", exc_info=True)
            return {
                'success': False,
                'output': "I encountered an error while processing your request. Please try again.",
                'error': str(e)
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
            history = await self._get_formatted_history(conversation_id, user_id, max_messages=3)
            
            # Truncate context to prevent token limit exceeded errors
            def truncate_context_data(data, max_chars=2000):
                """Truncate context data to prevent large payloads."""
                if isinstance(data, str):
                    return data[:max_chars] + "..." if len(data) > max_chars else data
                elif isinstance(data, dict):
                    truncated = {}
                    for k, v in data.items():
                        if isinstance(v, str) and len(v) > max_chars:
                            truncated[k] = v[:max_chars] + "..."
                        else:
                            truncated[k] = v
                    return truncated
                return data
            
            # Truncate context
            truncated_context = truncate_context_data(context)
            
            # Prepare context for the agent
            process_kwargs = {
                'input_text': user_input,
                'chat_history': history,
                'conversation_id': conversation_id,
                'user_id': user_id,
                **truncated_context
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
                    # More aggressive truncation: limit individual messages to 300 chars
                    if len(content) > 300:
                        content = content[:297] + '...'
                    formatted.append(f"{role}: {content}")
                except Exception as msg_err:
                    logger.warning(f"Error formatting message: {msg_err}", exc_info=True)
                    continue
            joined = "\n".join(formatted) if formatted else "No conversation history available."
            # More aggressive truncation: if total exceeds 2000 chars, truncate from the start
            if len(joined) > 2000:
                joined = joined[-2000:]
                logger.warning(f"Truncated conversation history from {len(joined) + 2000} to 2000 characters")
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

    def _post_process_response(self, response: str) -> str:
        """Clean and format agent responses for professionalism and clarity."""
        if not isinstance(response, str):
            return response
        # List of unwanted phrases or patterns to remove
        unwanted_phrases = [
            "View this free webinar from WebMD.",
            "But that changed because of concerns that chemicals in butterbur supplements may cause serious liver damage.",
            "... View this free webinar from WebMD.",
            # Add more as needed
        ]
        for phrase in unwanted_phrases:
            response = response.replace(phrase, "")
        # Remove excessive whitespace
        response = re.sub(r'\n{3,}', '\n\n', response)
        response = re.sub(r'\s{2,}', ' ', response)
        # Convert plain URLs to Markdown hyperlinks (WebMD example)
        def url_to_md(match):
            url = match.group(0)
            if "webmd.com" in url:
                return f"[WebMD: 5 Ways to Get Rid of a Headache]({url})"
            return f"[Link]({url})"
        response = re.sub(r'https?://[\w./?=&%-]+', url_to_md, response)
        return response.strip()

    # --- LLM-based relevance extraction for Brave Search output ---
    async def _extract_relevant_text_with_llm(self, user_input: str, raw_output: str) -> str:
        """Use the LLM to extract only the most relevant answer for the user's query from the raw Brave Search output."""
        from langchain_groq import ChatGroq
        from langchain_core.messages import SystemMessage, HumanMessage
        from llm.config import settings
        llm = ChatGroq(
            temperature=0.2,
            model_name=settings.REASONING_MODEL,
            groq_api_key=settings.GROQ_API_KEY
        )
        system_prompt = (
            "You are a helpful assistant. Given the user's question and a raw web search result, "
            "extract and return ONLY the most relevant, clear, and context-aware answer for the user. "
            "Remove any out-of-context, irrelevant, or distracting text. "
            "If there is a useful link, format it as a Markdown hyperlink. "
            "Do NOT include disclaimers, ads, or unrelated sentences. "
            "Respond in a professional, concise, and user-friendly way."
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"User question: {user_input}\n\nRaw web search output: {raw_output}")
        ]
        result = await llm.ainvoke(messages)
        return result.content.strip()

    def filter_live_results(self, user_query, results, live_window_minutes=60):
        now = datetime.datetime.utcnow()
        live_keywords = ['live', 'now', 'currently', 'ongoing', 'breaking', 'score', 'update', 'latest']
        filtered = []
        for r in results:
            text = (r.get('title', '') + ' ' + r.get('description', '')).lower()
            date_str = r.get('publishedAt') or r.get('date') or r.get('datePublished')
            is_live = any(kw in user_query.lower() for kw in live_keywords) or any(kw in text for kw in live_keywords)
            if date_str:
                try:
                    pub_date = datetime.datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    if is_live and (now - pub_date).total_seconds() <= live_window_minutes * 60:
                        filtered.append(r)
                except Exception:
                    continue
        return filtered

    def semantic_filter_and_rank(self, user_query, results, top_n=5):
        if not self._semantic_model:
            return results[:top_n]
        query_emb = self._semantic_model.encode(user_query, convert_to_tensor=True)
        scored = []
        for r in results:
            text = (r.get('title', '') + ' ' + r.get('description', '')).strip()
            if text:
                score = util.pytorch_cos_sim(query_emb, self._semantic_model.encode(text, convert_to_tensor=True)).item()
                scored.append((score, r))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [r for score, r in scored[:top_n]]

# ... (rest of the code remains the same)
