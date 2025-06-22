"""Researcher agent implementation.

This module contains the ResearcherAgent class which is specialized in finding and
gathering information from various sources using tools like BraveSearch.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Union

# LangChain imports
from langchain_core.tools import BaseTool

# Local imports
from .base import BaseAgent, AgentConfig
from llm.config import settings

logger = logging.getLogger(__name__)

class ResearcherAgent(BaseAgent):
    """Agent specialized in finding and gathering information from various sources.
    
    This agent is designed to handle research-intensive tasks by leveraging available tools
    to gather, analyze, and synthesize information from multiple sources. It includes robust
    error handling for API limits and missing dependencies.
    """
    
    def __init__(self, config: Union[AgentConfig, Dict[str, Any]]):
        """Initialize the researcher agent with configuration.
        
        Args:
            config: The configuration for the agent. Can be a dictionary or AgentConfig instance.
                   If None, default values will be used.
        """
        if not isinstance(config, (AgentConfig, dict)):
            raise ValueError("config must be an AgentConfig instance or a dictionary")
            
        if isinstance(config, dict):
            # Define research-specific system prompt with clear instructions
            system_prompt = """
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

Be thorough, objective, and always cite your sources. For all queries about current events, news, or reports, the user expects a real-time web search and cited sources."""
            
            # Set default values if not provided
            default_config = {
                'name': 'researcher',
                'description': 'Specialized in finding and gathering information from various sources.',
                'model_name': settings.REASONING_MODEL,
                'temperature': 0.3,  # Lower temperature for more focused, factual responses
                'system_prompt': system_prompt,
                'max_iterations': 5  # Allow for multiple search iterations
            }
            # Update with any provided config values
            default_config.update(config)
            config = default_config
            
        super().__init__(config)
        
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
        
        # Initialize tools list
        agent_tools = []
        
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
            
        # Update the config with tools
        if isinstance(config, dict):
            config['tools'] = agent_tools
        else:
            config.tools = agent_tools
        
        # Initialize logging and metrics
        self.search_count = 0
        self.last_search_time = None
    
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
            if "chat_history" in kwargs:
                agent_input["chat_history"] = kwargs["chat_history"]
            if "context" in kwargs:
                agent_input["context"] = kwargs["context"]

            # Execute the agent
            result = self.agent.invoke(agent_input)
            # Post-process the results
            return self._postprocess_research(result)
        except Exception as e:
            logger.error(f"Error in research process: {str(e)}", exc_info=True)
            return {
                'output': f"I encountered an error while processing your research request: {str(e)}",
                'metadata': {
                    'error': str(e),
                    'success': False
                }
            }

    async def process(self, input_text: str, context: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """Async process method for research agent."""
        try:
            processed_query = self._preprocess_query(input_text)
            agent_input = {"input": processed_query}
            if "chat_history" in kwargs:
                agent_input["chat_history"] = kwargs["chat_history"]
            if context is not None:
                agent_input["context"] = context
            result = await self.agent.ainvoke(agent_input)
            return self._postprocess_research(result)
        except Exception as e:
            logger.error(f"Async error in research process: {str(e)}", exc_info=True)
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
