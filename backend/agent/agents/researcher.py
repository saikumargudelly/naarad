"""Researcher agent implementation.

This module contains the ResearcherAgent class which is specialized in finding and
gathering information from various sources using tools like BraveSearch.
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional, Union

# LangChain imports
from langchain_core.tools import BaseTool

# Local imports
from .base import BaseAgent, AgentConfig
from llm.config import settings
from agent.memory.memory_manager import MemoryManager
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)

class ResearcherAgent(BaseAgent):
    """Agent specialized in finding and gathering information from various sources.
    Modular, stateless, and uses injected memory manager for context/state.
    """
    
    def __init__(self, config: Union[AgentConfig, Dict[str, Any]], memory_manager: MemoryManager = None):
        """Initialize the researcher agent with configuration.
        
        Args:
            config: The configuration for the agent. Can be a dictionary or AgentConfig instance.
                   If None, default values will be used.
            memory_manager: The memory manager for the agent. If None, no memory manager will be used.
        """
        if not isinstance(config, (AgentConfig, dict)):
            raise ValueError("config must be an AgentConfig instance or a dictionary")
            
        if isinstance(config, dict):
            routing_info = config.get('routing_info', {})
            intent = routing_info.get('intent', 'unknown')
            confidence = routing_info.get('confidence', None)
            entities = routing_info.get('entities', {})
            routing_str = f"\n[Routing Info]\nIntent: {intent} (confidence: {confidence})\nEntities: {entities}\n" if intent != 'unknown' else ''
            system_prompt = f"""
{routing_str}
IMPORTANT: For ANY query about real-time, current events, news, or anything that may require up-to-date information, you MUST ALWAYS use the brave_search tool to perform a real-time web search. DO NOT answer from your own knowledge for these queries. If the brave_search tool is not available, inform the user and do not attempt to answer from memory.

You are an expert research assistant. Your primary responsibility is to provide users with the most accurate, up-to-date, and well-sourced information available.

INSTRUCTIONS:
1. For ANY query about news, reports, current events, recent developments, trending topics, or anything that may require up-to-date information (including but not limited to: 'latest', 'breaking', 'today', 'recent', 'news', 'report', 'update', 'trend', 'headline', 'event', 'announcement', 'release', 'find', 'search', 'web', 'internet', 'article', 'source', 'reference'), you MUST ALWAYS use the brave_search tool to perform a real-time web search. DO NOT answer from your own knowledge for these queries.
2. For product searches, use the brave_search tool and present results in this format:

===== PRODUCT SEARCH RESULTS =====
[Product Brand and Model] - [Price]
- Key features: [list 3-5 most important features]
- Where to buy: [list retailers with prices if available]
- Pros: [list 2-3 pros]
- Cons: [list 2-3 cons]
- Overall rating: [if available]

3. For general research questions:
   - Use the brave_search tool to find relevant, recent information
   - Synthesize and summarize information from multiple sources
   - Provide clear, well-structured responses
   - ALWAYS cite your sources with URLs or publication names
   - If information is conflicting, present different perspectives and cite all sources

4. If the brave_search tool is not available or fails:
   - Inform the user that real-time search is currently unavailable
   - Suggest trying again later or rephrasing the query
   - Only then, provide any relevant general knowledge you have, but clearly state it may be outdated

5. NEVER fabricate or guess recent information. If you cannot find a reliable, up-to-date answer, say so clearly.

Be thorough, objective, and always cite your sources. For all queries about current events, news, or reports, the user expects a real-time web search and cited sources.

If the query is outside your research domain or the intent confidence is low, escalate to the appropriate agent or ask the user for clarification before proceeding. If you cannot help, say so and suggest the correct agent or next step.
"""
            
            # Always define default_config first
            default_config = {
                'name': 'researcher',
                'description': 'Specialized in finding and gathering information from various sources.',
                'model_name': settings.REASONING_MODEL,
                'temperature': 0.3,  # Lower temperature for more focused, factual responses
                'system_prompt': system_prompt,
                'max_iterations': 5  # Allow for multiple search iterations
            }
            # Initialize tools list
            agent_tools = []
            # Try to import and initialize GNewsSearchTool if API key is available
            if os.getenv('GNEWS_API_KEY'):
                try:
                    from ..tools.gnews_search import GNewsSearchTool
                    agent_tools.append(GNewsSearchTool())
                except ImportError as e:
                    logger.warning(f"Failed to load GNewsSearchTool: {e}")
            # Try to import and initialize BraveSearchTool if API key is available
            if os.getenv('BRAVE_API_KEY'):
                try:
                    from ..tools.brave_search import BraveSearchTool
                    agent_tools.append(BraveSearchTool())
                except ImportError as e:
                    logger.warning(f"Failed to load BraveSearchTool: {e}")
            else:
                logger.warning("BRAVE_API_KEY not found. Brave search functionality will be disabled.")
            # If no tools were added, add a dummy tool
            if not agent_tools:
                from ..tools.dummy_tool import DummyTool
                agent_tools.append(DummyTool())
            default_config['tools'] = agent_tools
            # Now update with any provided config values
            default_config.update(config)
            config = default_config
            
        super().__init__(config)
        self.memory_manager = memory_manager
        logger.info(f"ResearcherAgent initialized with memory_manager: {bool(memory_manager)}")
        
        # Check for required API keys and services
        missing_apis = []
        
        # Check for Groq API key
        if not os.getenv('GROQ_API_KEY'):
            missing_apis.append('Groq')
            
        # Check for other required API keys
        if not os.getenv('BRAVE_API_KEY'):
            missing_apis.append('Brave Search')
        
        # Update system prompt with any missing API information
        if missing_apis:
            missing_str = ", ".join(missing_apis)
            system_prompt += f"\n\nNOTE: The following services are not available: {missing_str}. "
            system_prompt += "Some features may be limited. Please check your API configuration."
        
        # Initialize logging and metrics
        self.search_count = 0
        self.last_search_time = None
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def _truncate_for_llm(self, text: str, max_chars: int = 1500) -> str:
        """Truncate text to fit within LLM token limits.
        
        Args:
            text: The text to truncate
            max_chars: Maximum characters to allow (roughly 1500 chars ≈ 2000 tokens)
            
        Returns:
            Truncated text with ellipsis if needed
        """
        if not text or len(text) <= max_chars:
            return text
        
        # Truncate and add ellipsis
        truncated = text[:max_chars-3] + "..."
        logger.warning(f"Truncated text from {len(text)} to {len(truncated)} characters for LLM")
        return truncated
    
    def _estimate_total_payload_size(self, agent_input: Dict[str, Any]) -> int:
        """Estimate the total payload size in characters."""
        total_size = 0
        for key, value in agent_input.items():
            if isinstance(value, str):
                total_size += len(value)
            elif isinstance(value, dict):
                total_size += len(str(value))
            else:
                total_size += len(str(value))
        return total_size
    
    def _truncate_chat_history(self, chat_history: str, max_chars: int = 2000) -> str:
        """Truncate chat history to prevent token limit exceeded errors.
        
        Args:
            chat_history: The chat history string
            max_chars: Maximum characters to allow
            
        Returns:
            Truncated chat history
        """
        if not chat_history or len(chat_history) <= max_chars:
            return chat_history
        
        # Try to truncate to last N messages if it's a structured format
        if "Human:" in chat_history and "Assistant:" in chat_history:
            # Split by message boundaries and keep last few messages
            messages = chat_history.split("Human:")
            if len(messages) > 1:
                # Keep the last 3 message exchanges
                recent_messages = messages[-3:]
                truncated = "Human:".join(recent_messages)
                if len(truncated) > max_chars:
                    truncated = self._truncate_for_llm(truncated, max_chars)
                logger.warning(f"Truncated chat history to last 3 messages ({len(truncated)} chars)")
                return truncated
        
        # Fallback to simple truncation
        return self._truncate_for_llm(chat_history, max_chars)
    
    def _truncate_context(self, context: Any, max_chars: int = 1000) -> Any:
        """Truncate context to prevent token limit exceeded errors.
        
        Args:
            context: The context data (dict, string, or other)
            max_chars: Maximum characters to allow
            
        Returns:
            Truncated context
        """
        if not context:
            return context
        
        if isinstance(context, str):
            return self._truncate_for_llm(context, max_chars)
        elif isinstance(context, dict):
            # Convert dict to string and truncate
            context_str = str(context)
            if len(context_str) > max_chars:
                truncated_str = self._truncate_for_llm(context_str, max_chars)
                logger.warning(f"Truncated context dict from {len(context_str)} to {len(truncated_str)} characters")
                return truncated_str
            return context
        else:
            # For other types, convert to string and truncate
            context_str = str(context)
            return self._truncate_for_llm(context_str, max_chars)
    
    def _create_agent(self):
        """Create and configure the LangChain agent for research tasks.
        
        Returns:
            Configured agent executor
            
        Raises:
            RuntimeError: If agent creation fails
        """
        try:
            # Import here to avoid circular imports
            from langchain.agents import AgentExecutor, create_react_agent
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_groq import ChatGroq
            from langchain import hub
            
            # Get the prompt to use for the agent
            prompt = hub.pull("hwchase17/react")
            
            # Add placeholders for chat history
            prompt.template = (
                "You are a helpful assistant. Please respond to the user's request."
                + prompt.template
            )
            prompt.input_variables.extend(["chat_history"])
            prompt.template += "\n{chat_history}"

            # Initialize the language model
            llm = ChatGroq(
                temperature=self.config.temperature,
                model_name=self.config.model_name,
                groq_api_key=os.getenv('GROQ_API_KEY')
            )

            # Create the ReAct agent
            agent = create_react_agent(
                llm=llm,
                tools=self.config.tools,
                prompt=prompt,
            )
            
            # Create an agent executor
            return AgentExecutor(
                agent=agent,
                tools=self.config.tools,
                verbose=self.config.verbose,
                handle_parsing_errors=self.config.handle_parsing_errors,
                max_iterations=self.config.max_iterations,
                early_stopping_method=self.config.early_stopping_method
            )
            
        except Exception as e:
            logger.error(f"Failed to create researcher agent: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to initialize researcher agent: {str(e)}") from e
    
    def _process_impl(self, input_text: str, **kwargs) -> Dict[str, Any]:
        """Process a research query with enhanced error handling and API management.
        
        Args:
            input_text: The research query or question
            **kwargs: Additional arguments including:
                - chat_history: Previous messages in the conversation
                - conversation_id: ID of the current conversation
                - user_id: ID of the current user
                
        Returns:
            Dict containing the research results and metadata
        """
        try:
            # Preprocess the query
            processed_query = self._preprocess_query(input_text)

            # Only pass expected keys to the agent
            agent_input = {"input": processed_query}
            
            # Truncate chat_history to prevent token limit exceeded errors
            if "chat_history" in kwargs:
                agent_input["chat_history"] = self._truncate_chat_history(kwargs["chat_history"])
            
            # Truncate context to prevent token limit exceeded errors
            if "context" in kwargs:
                agent_input["context"] = self._truncate_context(kwargs["context"])

            # Final payload size check - more aggressive for 8b model
            total_payload_size = self._estimate_total_payload_size(agent_input)
            logger.warning(f"LLM model: {self.config.model_name}, payload size: {total_payload_size}")
            logger.warning(f"agent_input keys: {list(agent_input.keys())}, input length: {len(agent_input.get('input', ''))}, chat_history length: {len(agent_input.get('chat_history', ''))}, context length: {len(str(agent_input.get('context', '')))}")
            if total_payload_size > 4000:  # Very conservative limit for 8b model
                logger.warning(f"Payload size {total_payload_size} exceeds limit, applying emergency truncation")
                # Emergency truncation: keep only essential fields
                agent_input = {
                    "input": agent_input.get("input", "")[:500],
                    "chat_history": "",
                    "context": ""
                }

            # Execute the agent
            try:
                result = self.agent.invoke(agent_input)
                # Post-process the results
                return self._postprocess_research(result)
            except Exception as llm_error:
                if "413" in str(llm_error) or "too large" in str(llm_error).lower() or "token" in str(llm_error).lower():
                    logger.warning(f"LLM token limit exceeded, trying with minimal context: {llm_error}")
                    # Try again with minimal context
                    minimal_input = {"input": processed_query, "chat_history": ""}
                    result = self.agent.invoke(minimal_input)
                    return self._postprocess_research(result)
                else:
                    raise llm_error
        except Exception as e:
            logger.error(f"Error in research process: {str(e)}", exc_info=True)
            return {
                'output': f"I encountered an error while processing your research request: {str(e)}",
                'metadata': {
                    'error': str(e),
                    'success': False
                }
            }

    def _is_realtime_query(self, input_text: str, chat_history: str = "") -> bool:
        """Detect if the query is about news, sports, schedules, or real-time events, using context if available."""
        realtime_patterns = [
            r"\b(news|headline|current event|breaking|top|latest|update|today|now|report|trend|event|announcement|release|find|search|web|internet|article|source|reference)\b",
            r"\b(match|game|tournament|schedule|fixture|cricket|football|soccer|test match|series|event|score|live|result|when is|date|time|upcoming|next|sports|athlete|player|team|league|cup|olympic|world cup|championship|draw|round|quarterfinal|semifinal|final|kickoff|start|begin|play|vs\.?|versus|playing 11|lineup|squad|team selection|probable|expected|roster|squad list|captain|coach|manager|injury|transfer|auction|draft|substitute|bench|formation|strategy|batting order|bowling order|pitch report|weather|umpire|referee|stadium|venue|broadcast|telecast|stream|tickets|fan|supporter|crowd|attendance|press conference|media|announcement|statement|rumor|speculation|prediction|preview|review|analysis|statistic|record|milestone|history|highlight|summary|recap|reaction|opinion|comment|quote|interview|exclusive|breaking news|live update|minute by minute|as it happens|real time|today|tonight|tomorrow|this week|this month|this year|now|soon|imminent|forthcoming|upcoming|future|next|latest|recent|current|ongoing|scheduled|postponed|delayed|cancelled|rescheduled|confirmed|official|provisional|tentative|finalized|announced|declared|named|listed|shortlisted|selected|picked|chosen|called up|drafted|signed|joined|left|released|retired|comeback|return|debut|first appearance|farewell|last match|final match|goodbye|tribute|honor|award|recognition|celebration|ceremony|event)\b"
        ]
        input_text_lower = input_text.lower()
        for pattern in realtime_patterns:
            if re.search(pattern, input_text_lower):
                return True
        # Check if previous user message in chat_history was real-time
        if chat_history:
            # Try to extract last user message from chat_history (string or list)
            last_user_msg = ""
            if isinstance(chat_history, str):
                # Assume format: 'Human: ...\nAssistant: ...\nHuman: ...'
                user_msgs = [m for m in chat_history.split('Human:') if m.strip()]
                if user_msgs:
                    last_user_msg = user_msgs[-1].strip().split('Assistant:')[0].strip()
            elif isinstance(chat_history, list):
                # Assume list of dicts with 'role' and 'content'
                user_msgs = [m.get('content', '') for m in chat_history if m.get('role', '').lower() == 'user']
                if user_msgs:
                    last_user_msg = user_msgs[-1]
            if last_user_msg:
                for pattern in realtime_patterns:
                    if re.search(pattern, last_user_msg.lower()):
                        # If current query is contextually related (e.g., contains 'playing 11', 'lineup', etc.)
                        followup_patterns = [
                            r"\b(playing 11|lineup|squad|team selection|probable|expected|roster|squad list|batting order|bowling order|strategy|formation|injury|substitute|bench|captain|coach|manager|prediction|preview|analysis|statistic|record|milestone|highlight|summary|recap|reaction|opinion|comment|quote|interview|exclusive|breaking news|live update|minute by minute|as it happens|real time|today|tonight|tomorrow|this week|this month|this year|now|soon|imminent|forthcoming|upcoming|future|next|latest|recent|current|ongoing|scheduled|postponed|delayed|cancelled|rescheduled|confirmed|official|provisional|tentative|finalized|announced|declared|named|listed|shortlisted|selected|picked|chosen|called up|drafted|signed|joined|left|released|retired|comeback|return|debut|first appearance|farewell|last match|final match|goodbye|tribute|honor|award|recognition|celebration|ceremony|event)\b"
                        ]
                        for fpat in followup_patterns:
                            if re.search(fpat, input_text_lower):
                                return True
        return False

    async def _synthesize_answer_with_llm(self, user_question: str, articles: list) -> str:
        """Use the LLM to synthesize a concise, accurate answer from news articles."""
        from langchain_groq import ChatGroq
        from langchain_core.messages import SystemMessage, HumanMessage
        from llm.config import settings
        if not articles:
            return "No relevant news articles found."
        # Sort articles by relevance: prioritize those whose title/description best match the question
        def relevance_score(article):
            text = (article.get('title', '') + ' ' + article.get('description', '')).lower()
            score = 0
            for word in user_question.lower().split():
                if word in text:
                    score += 1
            return score
        sorted_articles = sorted(articles, key=relevance_score, reverse=True)
        # Concatenate top 5 snippets for LLM synthesis
        context_snippets = "\n".join(
            f"{a.get('title', '')}: {a.get('description', '')} (Source: {a.get('url', '')})"
            for a in sorted_articles[:5]
        )
        system_prompt = (
            "You are a world-class research assistant. Given the user's question and a set of news search results, "
            "synthesize a single, concise, and accurate answer. If a date, location, or key fact is mentioned, include it. "
            "Always cite the most relevant source link. Do not list all articles—just answer the question as directly as possible."
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"User question: {user_question}\n\nNews search results:\n{context_snippets}")
        ]
        llm = ChatGroq(
            temperature=0.2,
            model_name=settings.REASONING_MODEL,
            groq_api_key=os.getenv('GROQ_API_KEY')
        )
        result = await llm.ainvoke(messages)
        return result.content.strip()

    async def _extract_topic_with_llm(self, chat_history: str, input_text: str) -> str:
        """Use the LLM to extract the main topic from the conversation."""
        from langchain_groq import ChatGroq
        from langchain_core.messages import SystemMessage, HumanMessage
        from llm.config import settings
        system_prompt = (
            "You are an expert conversation analyst. Given the following conversation and the latest user message, "
            "extract the main topic or subject being discussed in a single phrase. If the topic is ambiguous, return your best guess."
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Conversation so far:\n{chat_history}\n\nLatest user message:\n{input_text}\n\nMain topic:")
        ]
        llm = ChatGroq(
            temperature=0.2,
            model_name=settings.REASONING_MODEL,
            groq_api_key=os.getenv('GROQ_API_KEY')
        )
        result = await llm.ainvoke(messages)
        return result.content.strip()

    def _is_semantic_followup(self, input_text: str, chat_history: str, threshold: float = 0.7) -> bool:
        """Detect if the current query is a semantic follow-up to the last user message."""
        if not chat_history:
            return False
        last_user_msg = ""
        if isinstance(chat_history, str):
            user_msgs = [m for m in chat_history.split('Human:') if m.strip()]
            if user_msgs:
                last_user_msg = user_msgs[-1].strip().split('Assistant:')[0].strip()
        elif isinstance(chat_history, list):
            user_msgs = [m.get('content', '') for m in chat_history if m.get('role', '').lower() == 'user']
            if user_msgs:
                last_user_msg = user_msgs[-1]
        if not last_user_msg:
            return False
        emb1 = self.embedding_model.encode(input_text, convert_to_tensor=True)
        emb2 = self.embedding_model.encode(last_user_msg, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(emb1, emb2).item()
        return similarity > threshold

    async def process(self, input_text: str, context: Dict[str, Any] = None, conversation_id: str = None, user_id: str = None, conversation_memory=None, **kwargs) -> Dict[str, Any]:
        logger.info(f"ResearcherAgent.process called | input_text: {input_text} | conversation_id: {conversation_id} | user_id: {user_id}")
        try:
            chat_history = kwargs.get('chat_history', '')
            # Use advanced context from conversation_memory
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
            # Use robust check for real-time queries (news, sports, events, follow-ups)
            is_realtime = self._is_realtime_query(input_text, chat_history)
            is_semantic_followup = self._is_semantic_followup(input_text, chat_history)
            realtime_intents = {'get_news', 'get_schedule', 'get_team_lineup', 'get_stats', 'get_event', 'get_result', 'get_update'}
            if is_realtime or is_semantic_followup or (intent and intent in realtime_intents):
                logger.info("[ResearcherAgent] Delegating real-time query to orchestrator's routing logic.")
                return {
                    'output': "[Orchestrator handles real-time routing and response. See orchestrator logs for details.]",
                    'metadata': {
                        'delegated_to_orchestrator': True,
                        'success': None,
                        'topic': topic,
                        'intent': intent
                    }
                }
            # For non-realtime queries, use LLM with full context
            context_snippets = "\n".join([
                f"{m['role'].capitalize()}: {m['content']}" for m in conversation_memory.messages[-6:]
            ]) if conversation_memory else ""
            system_prompt = (
                "You are a world-class research assistant. Use the conversation context, topic, and intent to answer the user's question as accurately and helpfully as possible. "
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
            logger.error(f"Error in research process: {str(e)}", exc_info=True)
            return {
                'output': f"I encountered an error while processing your research request: {str(e)}",
                'metadata': {
                    'error': str(e),
                    'success': False
                }
            }
    
    def _preprocess_query(self, query: str) -> str:
        """Preprocess the research query for better search results.
        
        Args:
            query: The original user query
                
        Returns:
            Processed query string optimized for search
        """
        # Remove any leading/trailing whitespace
        query = query.strip()
        
        # Add context if it's a product search
        if any(term in query.lower() for term in ['best', 'compare', 'review', 'price']):
            query = f"latest {query} with prices and reviews"
            
        return query
    
    def _postprocess_research(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process the research results for better presentation.
        
        Args:
            result: The raw result from the agent
            
        Returns:
            Processed result with enhanced formatting
        """
        try:
            output = result.get('output', '')
            sources = result.get('sources', [])
            # Add a header if not present
            if not output.startswith('====='):
                output = f"===== RESEARCH RESULTS =====\n\n{output}"
            # Ensure proper formatting
            if not output.endswith('\n'):
                output += '\n'
            if not output.endswith('====='):
                output += "\n===== END OF RESULTS ====="
            # Contextual follow-up if multiple sources/results
            if isinstance(sources, list) and len(sources) > 1:
                followup = self._contextual_followup(self.last_query if hasattr(self, 'last_query') else '', sources, domain='news')
                output += f"\n\n{followup}"
            return {
                'output': output,
                'metadata': {
                    'sources': sources,
                    'success': True
                }
            }
        except Exception as e:
            logger.error(f"Error post-processing research results: {str(e)}", exc_info=True)
            return {
                'output': str(result.get('output', 'Research completed')),
                'metadata': {
                    'success': True,
                    'warning': 'Results may not be properly formatted'
                }
            }
