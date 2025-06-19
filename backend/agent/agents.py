from typing import Dict, Any, List, Optional, Type, TypeVar, Union, Sequence, Tuple
from dataclasses import dataclass, field
import logging
import os
import json
import time
import asyncio
import traceback
from datetime import datetime
from dotenv import load_dotenv

# LangChain imports
from langchain.agents import AgentExecutor, create_react_agent, tool as langchain_tool
from langchain.agents.agent import AgentFinish, AgentAction, AgentStep
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseLanguageModel
from langchain_community.llms import Ollama

# Local imports
from .types import AgentConfig, BaseAgent

def create_react_agent(
    llm: BaseLanguageModel,
    tools: Sequence[BaseTool],
    prompt: ChatPromptTemplate
) -> AgentExecutor:
    """Create a ReAct agent with the given LLM and tools.
    
    Args:
        llm: The language model to use
        tools: List of tools the agent can use
        prompt: The prompt template to use
        
    Returns:
        An AgentExecutor instance
    """
    # Format the tools for the prompt
    tool_names = [tool.name for tool in tools]
    tool_descriptions = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])
    
    # Create the agent
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
            "tool_names": lambda x: ", ".join(tool_names),
            "tool_descriptions": lambda x: tool_descriptions,
        }
        | prompt
        | llm
        | ReActSingleInputOutputParser()
    )
    
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

from llm.config import settings

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Load environment variables
load_dotenv()

# Type variable for agent classes
AgentT = TypeVar('AgentT', bound='BaseAgent')

@dataclass
class AgentConfig:
    """Configuration for an agent.
    
    Models are configured to use OpenRouter with the following mapping:
    - Language Reasoning: mistralai/Mixtral-8x7B-Instruct-v0.1
    - Personality Chat: nousresearch/nous-hermes-2-mixtral-8x7b-dpo
    - Default: mistralai/Mixtral-8x7B-Instruct-v0.1
    """
    name: str
    description: str
    model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"  # Default model via OpenRouter
    temperature: float = 0.7
    system_prompt: str = ""
    tools: List[Any] = field(default_factory=list)
    max_iterations: int = 10
    verbose: bool = True

class BaseAgent:
    """Base class for all agents."""
    
    def __init__(self, config: AgentConfig):
        """Initialize the agent with configuration."""
        self.config = config
        self.agent = self._create_agent()
    
    @property
    def name(self) -> str:
        """Get the agent's name."""
        return self.config.name
    
    @property
    def description(self) -> str:
        """Get the agent's description."""
        return self.config.description
    
    def _create_agent(self) -> AgentExecutor:
        """Create and return a configured agent with proper tool handling."""
        from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
        from langchain.chains import LLMChain
        from langchain_core.prompts import PromptTemplate
        from langchain_openai import ChatOpenAI
        import os
        import logging

        logger = logging.getLogger(__name__)
        
        try:
            # Get tool names for the prompt
            tool_names = [tool.name for tool in self.config.tools] if self.config.tools else []
            
            # Create the LLM instance with proper configuration
            llm = ChatOpenAI(
                model_name=self.config.model_name,
                temperature=self.config.temperature,
                openai_api_base="https://openrouter.ai/api/v1",
                openai_api_key=os.getenv("OPENROUTER_API_KEY"),
                max_retries=3,
                request_timeout=60,
                streaming=False
            )
            
            # Set custom headers for OpenRouter
            headers = {
                "HTTP-Referer": "https://github.com/saikumargudelly/naarad",
                "X-Title": "Naarad"
            }
            if hasattr(llm, 'client') and hasattr(llm.client, '_client'):
                llm.client._client.default_headers.update(headers)
            
            # Create the prompt template with clear instructions
            prefix = f"""{self.config.system_prompt}

            You have access to the following tools: {', '.join(tool_names) if tool_names else 'No tools available'}

            Use the following format:

            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{', '.join(tool_names) if tool_names else 'No tools available'}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question"""

            suffix = """Begin!

            Question: {input}
            {agent_scratchpad}"""
            
            # Create the prompt template
            prompt = ZeroShotAgent.create_prompt(
                self.config.tools,
                prefix=prefix,
                suffix=suffix,
                input_variables=["input", "agent_scratchpad"]
            )
            
            # Create the LLM chain
            llm_chain = LLMChain(
                llm=llm,
                prompt=prompt,
                verbose=self.config.verbose
            )
            
            # Create the agent
            agent = ZeroShotAgent(
                llm_chain=llm_chain,
                tools=self.config.tools,
                verbose=self.config.verbose,
                handle_parsing_errors=True
            )
            
            # Create the agent executor with proper configuration
            agent_executor = AgentExecutor.from_agent_and_tools(
                agent=agent,
                tools=self.config.tools,
                verbose=self.config.verbose,
                max_iterations=min(self.config.max_iterations, 10),  # Cap at 10 iterations
                handle_parsing_errors=True,
                return_intermediate_steps=True,
                early_stopping_method="generate"
            )
            
            logger.info(f"Successfully created {self.name} agent with {len(tool_names)} tools")
            return agent_executor
            
        except Exception as e:
            logger.error(f"Error creating {self.name} agent: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to initialize {self.name} agent: {str(e)}")
    
    async def process(self, input_text: str, **kwargs) -> Dict[str, Any]:
        """Process input using the agent with enhanced error handling and logging.
        
        Args:
            input_text: The input text to process
            **kwargs: Additional arguments including:
                - chat_history: List of previous messages in the conversation
                - intermediate_steps: List of (action, observation) tuples
                - conversation_id: ID of the current conversation
                - user_id: ID of the current user
                
        Returns:
            Dict containing the response and metadata
        """
        start_time = time.time()
        logger.info(f"[{self.name}] Processing input: {input_text[:100]}...")
        
        try:
            # Clean up and validate input
            if not input_text or not isinstance(input_text, str):
                raise ValueError("Input text must be a non-empty string")
                
            # Log the full input and kwargs for debugging
            logger.debug(f"[{self.name}] Full input: {input_text}")
            logger.debug(f"[{self.name}] Additional kwargs: {json.dumps({k: str(v)[:200] for k, v in kwargs.items()}, indent=2)}")
            
            # Prepare the input for the agent
            invoke_kwargs = {
                k: v for k, v in kwargs.items()
                if k not in ['headers', 'model_kwargs', 'callbacks']
            }
            
            # Add conversation history if available
            if 'chat_history' in invoke_kwargs and not invoke_kwargs['chat_history']:
                del invoke_kwargs['chat_history']
            
            # Log tools available to the agent
            tool_names = [t.name for t in self.config.tools] if hasattr(self.config, 'tools') and self.config.tools else []
            logger.info(f"[{self.name}] Available tools: {tool_names}")
            
            # Process with the agent
            try:
                logger.info(f"[{self.name}] Invoking agent with input: {input_text[:200]}...")
                
                # Add a timeout to prevent hanging
                timeout_seconds = 60  # 1 minute timeout
                
                async def run_agent():
                    return await self.agent.ainvoke({
                        'input': input_text,
                        **invoke_kwargs
                    })
                
                # Run with timeout
                try:
                    result = await asyncio.wait_for(run_agent(), timeout=timeout_seconds)
                except asyncio.TimeoutError:
                    raise TimeoutError(f"Agent execution timed out after {timeout_seconds} seconds")
                
                logger.info(f"[{self.name}] Agent execution completed successfully")
                logger.debug(f"[{self.name}] Raw agent response: {json.dumps(str(result)[:500], indent=2)}")
                
            except Exception as e:
                logger.error(f"[{self.name}] Error during agent execution: {str(e)}", exc_info=True)
                raise RuntimeError(f"Agent execution failed: {str(e)}")
            
            # Process the agent response
            try:
                # Extract output from different response formats
                output = None
                
                if hasattr(result, 'output') and result.output is not None:
                    output = result.output
                elif hasattr(result, 'content'):
                    output = result.content
                elif hasattr(result, 'text'):
                    output = result.text
                elif isinstance(result, (str, int, float, bool)):
                    output = str(result)
                elif isinstance(result, dict):
                    if 'output' in result:
                        output = result['output']
                    elif 'response' in result:
                        output = result['response']
                    elif 'message' in result:
                        output = result['message']
                    elif 'messages' in result and result['messages']:
                        last_msg = result['messages'][-1]
                        if hasattr(last_msg, 'content'):
                            output = last_msg.content
                        elif isinstance(last_msg, dict) and 'content' in last_msg:
                            output = last_msg['content']
                        else:
                            output = str(last_msg)
                    else:
                        output = json.dumps(result, default=str)
                
                # Ensure output is a non-empty string
                if not output or not isinstance(output, str):
                    output = str(output) if output is not None else ""
                
                # Clean up the output
                output = output.strip()
                
                if not output:
                    logger.warning(f"[{self.name}] Empty response from agent")
                    output = "I received your message but didn't generate a response. Could you try rephrasing your question?"
                
                # Log processing time
                processing_time = time.time() - start_time
                logger.info(f"[{self.name}] Processing completed in {processing_time:.2f}s")
                
                # Prepare the response
                response = {
                    'success': True,
                    'output': output,
                    'agent_used': self.name,
                    'metadata': {
                        'model': self.config.model_name,
                        'temperature': self.config.temperature,
                        'processing_time_seconds': processing_time,
                        'tools_used': tool_names,
                        'response_length': len(output)
                    }
                }
                
                logger.debug(f"[{self.name}] Final response: {json.dumps({k: str(v)[:200] for k, v in response.items()}, indent=2)}")
                return response
                
            except Exception as e:
                logger.error(f"[{self.name}] Error processing agent response: {str(e)}\nRaw response: {str(result)[:1000]}", exc_info=True)
                raise RuntimeError(f"Failed to process agent response: {str(e)}")
            
        except Exception as e:
            # Handle all other exceptions
            processing_time = time.time() - start_time
            error_msg = f"Error in {self.name} agent after {processing_time:.2f}s: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            return {
                'success': False,
                'output': f"I encountered an error while processing your request with {self.name}. Please try again or rephrase your question.",
                'error': str(e),
                'agent_used': self.name,
                'metadata': {
                    'model': self.config.model_name,
                    'processing_time_seconds': processing_time,
                    'error_type': type(e).__name__
                }
            }

class ResearcherAgent(BaseAgent):
    """Agent specialized in finding and gathering information from various sources.
    
    This agent is designed to handle research-intensive tasks by leveraging available tools
    to gather, analyze, and synthesize information from multiple sources.
    """
    
    def __init__(self, tools: List[Any] = None):
        from llm.config import settings
        import os
        
        # Define research-specific system prompt
        system_prompt = """You are a highly skilled research assistant with expertise in gathering and analyzing 
        information from various sources. Your primary responsibilities include:
        
        1. Conducting thorough research using available tools
        2. Verifying information from multiple sources
        3. Providing clear, well-structured responses with proper citations
        4. Distinguishing between facts, opinions, and speculations
        5. Being transparent about the limitations of the information found
        
        When responding:
        - Always cite your sources when possible
        - Clearly indicate when information is from a specific source vs general knowledge
        - If information is contradictory across sources, present multiple viewpoints
        - Be concise but thorough in your responses
        - Use bullet points or numbered lists for better readability when appropriate
        - If you're unsure about something, say so rather than guessing
        """
        
        # Check for required API keys
        self.has_required_apis = True
        missing_apis = []
        
        # Check for OpenRouter API key
        if not os.getenv('OPENROUTER_API_KEY'):
            missing_apis.append('OpenRouter')
            self.has_required_apis = False
            
        # Check for other required API keys
        if not os.getenv('BRAVE_API_KEY'):
            missing_apis.append('Brave Search')
            
        if missing_apis:
            system_prompt += "\n\nNOTE: The following API keys are missing: " + ", ".join(missing_apis) + ". "
            system_prompt += "Some features may be limited. Please provide the missing API keys for full functionality."
        
        # Configure the agent
        config = AgentConfig(
            name="researcher",
            description="Specialized in finding and gathering information from various sources.",
            model_name=settings.REASONING_MODEL if os.getenv('OPENROUTER_API_KEY') else 'gpt-3.5-turbo',
            temperature=0.3,  # Lower temperature for more focused, deterministic responses
            system_prompt=system_prompt,
            tools=tools or [],
            max_iterations=8,  # Allow more iterations for thorough research
            verbose=True
        )
        
        # Initialize the base agent
        super().__init__(config)
        
        # Add research-specific logging
        self.logger = logging.getLogger(f"ResearcherAgent")
        self.search_count = 0
        self.missing_apis = missing_apis
        
    async def process(self, input_text: str, **kwargs) -> Dict[str, Any]:
        """Process a research query with enhanced research capabilities.
        
        Args:
            input_text: The research query or question
            **kwargs: Additional arguments including:
                - chat_history: Previous messages in the conversation
                - conversation_id: ID of the current conversation
                - user_id: ID of the current user
                
        Returns:
            Dict containing the research results and metadata
        """
        start_time = time.time()
        self.logger.info(f"Processing research query: {input_text[:200]}...")
        
        try:
            # Check for missing APIs
            if not self.has_required_apis:
                missing_apis = ", ".join(self.missing_apis)
                return {
                    'success': False,
                    'output': f"I'm unable to process this request because the following required API keys are missing: {missing_apis}. "
                             "Please provide the missing API keys in the .env file to enable full functionality.",
                    'agent_used': 'researcher',
                    'missing_apis': self.missing_apis
                }
            
            # Pre-process the input
            query = self._preprocess_query(input_text)
            
            # Track search operations
            self.search_count += 1
            
            # Add research context to the input
            research_context = {
                'research_goal': query,
                'search_depth': 'comprehensive',
                'require_citations': True,
                'max_sources': 3,
                'current_step': 'initial_research',
                'search_count': self.search_count
            }
            
            # Add research context to the kwargs
            kwargs['research_context'] = research_context
            
            # Process with the base agent
            try:
                result = await super().process(query, **kwargs)
                
                # Post-process the response
                if result.get('success', False):
                    result = self._postprocess_research(result)
                
                # Log completion
                processing_time = time.time() - start_time
                self.logger.info(f"Research completed in {processing_time:.2f}s")
                
                return result
                
            except Exception as e:
                self.logger.error(f"Error in base agent processing: {str(e)}", exc_info=True)
                return {
                    'success': False,
                    'output': "I encountered an error while processing your request. This might be due to missing or invalid API keys. "
                             "Please check your configuration and try again.",
                    'error': str(e),
                    'agent_used': 'researcher'
                }
            
        except Exception as e:
            self.logger.error(f"Research error: {str(e)}", exc_info=True)
            return {
                'success': False,
                'output': "I encountered an error while conducting research. Please try again with a more specific query.",
                'error': str(e),
                'agent_used': 'researcher'
            }
    
    def _preprocess_query(self, query: str) -> str:
        """Preprocess the research query for better results."""
        # Clean and normalize the query
        query = query.strip()
        
        # Add research-specific instructions if not present
        if not any(phrase in query.lower() for phrase in ['research', 'find', 'search', 'look up']):
            query = f"Research and provide detailed information about: {query}"
            
        return query
    
    def _postprocess_research(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process the research results for better presentation."""
        if not result.get('success', False):
            return result
            
        output = result.get('output', '')
        
        # Ensure the response includes citations if sources were used
        if 'sources' in result.get('metadata', {}) and 'source:' not in output.lower():
            sources = result['metadata']['sources']
            if sources:
                output += "\n\nSources:"
                for i, source in enumerate(sources, 1):
                    output += f"\n{i}. {source}"
        
        # Update the result with processed output
        result['output'] = output
        return result

class AnalystAgent(BaseAgent):
    """Agent specialized in analyzing information and providing insights."""
    
    def __init__(self, tools: List[Any] = None):
        from llm.config import settings
        config = AgentConfig(
            name="analyst",
            description="Specialized in analyzing information and providing insights.",
            model_name=settings.REASONING_MODEL,  # Use configured reasoning model
            temperature=0.5,
            system_prompt="""You are an analytical assistant. Your job is to analyze information, identify patterns, 
            and provide clear, insightful analysis. Consider multiple perspectives and provide balanced viewpoints. 
            Highlight key findings and their implications.""",
            tools=tools or [],
            max_iterations=5
        )
        super().__init__(config)

class ResponderAgent(BaseAgent):
    """Agent specialized in generating friendly and helpful responses."""
    
    def __init__(self, tools: List[Any] = None):
        from llm.config import settings
        config = AgentConfig(
            name="responder",
            description="Specialized in generating friendly and helpful responses.",
            model_name=settings.CHAT_MODEL,  # Use configured chat model
            temperature=0.8,
            system_prompt="""You are a friendly and helpful AI assistant. Your goal is to provide clear, 
            concise, and helpful responses. Be polite, empathetic, and engaging in your communication. 
            If you don't know something, be honest about it.""",
            tools=tools or [],
            max_iterations=3
        )
        super().__init__(config)

class QualityAgent(BaseAgent):
    """Agent specialized in refining and improving responses."""
    
    def __init__(self, tools: List[Any] = None):
        from llm.config import settings
        config = AgentConfig(
            name="quality",
            description="Specialized in refining and improving responses for quality and clarity.",
            model_name=settings.REASONING_MODEL,  # Use configured reasoning model
            temperature=0.3,
            system_prompt="""You are a quality assurance specialist. Your job is to review and improve responses 
            for clarity, conciseness, accuracy, and tone. Ensure the response is well-structured, free of errors, 
            and effectively addresses the user's query.""",
            tools=tools or [],
            max_iterations=2
        )
        super().__init__(config)

def create_base_agents(tools: List[Any] = None) -> Dict[str, BaseAgent]:
    """
    Create and return a dictionary of base agents.
    
    Args:
        tools: List of tools to make available to the agents
        
    Returns:
        Dict mapping agent names to agent instances
    """
    agents = {
        "researcher": ResearcherAgent(tools=tools),
        "analyst": AnalystAgent(tools=tools),
        "responder": ResponderAgent(tools=tools),
        "quality": QualityAgent(tools=tools)
    }
    
    logger.info(f"Created {len(agents)} base agents")
    return agents

def get_agent_class(agent_name: str) -> Optional[Type[BaseAgent]]:
    """
    Get an agent class by name.
    
    Args:
        agent_name: Name of the agent class to retrieve
        
    Returns:
        The agent class if found, None otherwise
    """
    agent_classes = {
        'researcher': ResearcherAgent,
        'analyst': AnalystAgent,
        'responder': ResponderAgent,
        'quality': QualityAgent
    }
    return agent_classes.get(agent_name.lower())
