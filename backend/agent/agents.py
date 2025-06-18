from typing import Dict, Any, List, Optional, Type, TypeVar, Union
from dataclasses import dataclass, field
import logging
import os
import traceback
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.llms import Ollama

# Create a simple React agent
def create_react_agent(llm, tools, prompt):
    return ReActChain(llm=llm, tools=tools)

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
        """Create and return a configured agent."""
        from langchain.agents import AgentExecutor, Tool, ZeroShotAgent, initialize_agent
        from langchain.agents.agent_types import AgentType
        from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS
        from langchain.chains import LLMChain
        from langchain_core.prompts import PromptTemplate
        from langchain_openai import ChatOpenAI
        from typing import List, Any, Dict, Union, Optional, Type, Tuple, Sequence
        
        # Get tool names for the prompt
        tool_names = ", ".join([tool.name for tool in self.config.tools])
        
        # Create a prompt template for the ReAct agent
        prompt = PromptTemplate.from_template(
            """{system_prompt}

            Current conversation:
            {chat_history}
            
            Question: {input}
            {agent_scratchpad}"""
        )
        
        # Create the LLM instance with proper header handling
        headers = {
            "HTTP-Referer": "https://github.com/your-github-username/your-repo-name",
            "X-Title": "Naarad"
        }
        
        # Create the LLM instance without headers first
        llm = ChatOpenAI(
            model_name=self.config.model_name,
            temperature=self.config.temperature,
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        
        # Manually set the headers on the client
        if hasattr(llm, 'client') and hasattr(llm.client, '_client'):
            llm.client._client.default_headers.update(headers)
        
        # Create the prompt with the system message and tool instructions
        prefix = f"""{self.config.system_prompt}

        You have access to the following tools:"""
        
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
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        
        # Create the agent
        agent = ZeroShotAgent(llm_chain=llm_chain, tools=self.config.tools)
        
        # Create the agent executor
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.config.tools,
            verbose=self.config.verbose,
            max_iterations=self.config.max_iterations,
            handle_parsing_errors=True
        )
        
        return agent_executor
    
    async def process(self, input_text: str, **kwargs) -> Dict[str, Any]:
        """Process input using the agent."""
        try:
            logger.info(f"Processing input with {self.name} agent: {input_text[:100]}...")
            
            # Log the full input for debugging
            logger.debug(f"Full input text: {input_text}")
            logger.debug(f"Additional kwargs: {kwargs}")
            
            # Ensure we're not passing any OpenAI-specific parameters
            invoke_kwargs = {k: v for k, v in kwargs.items() 
                           if k not in ['headers', 'model_kwargs']}
            
            # Import required message types
            from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, FunctionMessage
            from langchain.agents.format_scratchpad import format_log_to_messages
            from langchain.agents.output_parsers import ReActSingleInputOutputParser
            
            # Log the tools available to the agent
            logger.debug(f"Agent {self.name} has {len(self.config.tools)} tools: {[t.name for t in self.config.tools]}")
            
            # Prepare the input with required variables
            input_data = {
                "input": input_text,
                "agent_scratchpad": [],  # Will be populated with proper message types
                **{k: v for k, v in invoke_kwargs.items() if k != 'intermediate_steps'}
            }
            
            # Import required message types
            from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, FunctionMessage
            
            # Import required message types
            from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
            
            # Initialize agent_scratchpad as an empty list
            agent_scratchpad = []
            
            # Handle intermediate steps if present
            if 'intermediate_steps' in invoke_kwargs and invoke_kwargs['intermediate_steps']:
                logger.debug(f"Processing with {len(invoke_kwargs['intermediate_steps'])} intermediate steps")
                try:
                    # Convert intermediate steps to messages
                    steps = invoke_kwargs['intermediate_steps']
                    for action, observation in steps:
                        # Add the action message
                        if isinstance(action, tuple) and len(action) == 2:
                            action_msg = f"Action: {action[0]}\nAction Input: {action[1]}"
                        else:
                            action_msg = f"Action: {action}"
                        agent_scratchpad.append(AIMessage(content=action_msg))
                        
                        # Add the observation message
                        if isinstance(observation, str):
                            agent_scratchpad.append(
                                AIMessage(content=f"Observation: {observation}")
                            )
                        else:
                            agent_scratchpad.append(
                                AIMessage(content=f"Observation: {str(observation)}")
                            )
                    
                    logger.debug(f"Formatted {len(agent_scratchpad)} messages in agent_scratchpad")
                    
                except Exception as e:
                    logger.error(f"Error formatting agent_scratchpad: {str(e)}", exc_info=True)
                    agent_scratchpad = [AIMessage(content="")]
            else:
                # Initialize with empty message if no intermediate steps
                agent_scratchpad = [AIMessage(content="")]
            
            # Set the formatted agent_scratchpad in input_data
            input_data['agent_scratchpad'] = agent_scratchpad
                
            # If there's chat history, add it to the input
            if 'chat_history' in invoke_kwargs:
                logger.debug(f"Including {len(invoke_kwargs['chat_history'])} messages from chat history")
                input_data['chat_history'] = invoke_kwargs['chat_history']
            
            # Log the input data being sent to the agent
            logger.debug(f"Agent input data: {input_data}")
            
            # Invoke the agent
            logger.info("Invoking agent...")
            try:
                result = await self.agent.ainvoke(input_data)
                logger.info("Agent invocation complete")
                logger.debug(f"Raw agent response type: {type(result)}")
                logger.debug(f"Raw agent response: {result}")
                
                # Log the structure of the result for debugging
                if hasattr(result, '__dict__'):
                    logger.debug(f"Result attributes: {vars(result)}")
                elif isinstance(result, dict):
                    logger.debug(f"Result keys: {result.keys()}")
                
            except Exception as e:
                logger.error(f"Error invoking agent: {str(e)}", exc_info=True)
                raise
            
            # Extract the output based on different possible response formats
            output = None
            
            try:
                if isinstance(result, dict):
                    logger.debug("Processing dictionary response")
                    # Handle dictionary response
                    if 'output' in result:
                        output = result['output']
                        logger.debug(f"Found output in 'output' key: {output}")
                    elif 'response' in result:
                        output = result['response']
                        logger.debug(f"Found output in 'response' key: {output}")
                    elif 'message' in result:
                        output = result['message']
                        logger.debug(f"Found output in 'message' key: {output}")
                    elif 'messages' in result and result['messages']:
                        # Get the last message content if it exists
                        last_msg = result['messages'][-1]
                        if hasattr(last_msg, 'content'):
                            output = last_msg.content
                            logger.debug(f"Found output in message.content: {output}")
                        elif isinstance(last_msg, dict) and 'content' in last_msg:
                            output = last_msg['content']
                            logger.debug(f"Found output in message dict content: {output}")
                        else:
                            output = str(last_msg)
                            logger.debug(f"Converted last message to string: {output}")
                    else:
                        output = str(result)
                        logger.debug(f"Converted entire result to string: {output}")
                elif hasattr(result, 'output') and result.output is not None:
                    output = result.output
                    logger.debug(f"Found output in result.output: {output}")
                elif hasattr(result, 'content'):
                    output = result.content
                    logger.debug(f"Found output in result.content: {output}")
                elif hasattr(result, 'text'):
                    output = result.text
                    logger.debug(f"Found output in result.text: {output}")
                elif isinstance(result, str):
                    output = result
                    logger.debug(f"Result is already a string: {output}")
                else:
                    output = str(result)
                    logger.debug(f"Converted result to string: {output}")
                
                # Ensure output is a string
                if not isinstance(output, str):
                    logger.debug(f"Converting output to string from type: {type(output)}")
                    output = str(output)
                    
                if not output.strip():
                    logger.warning("Empty response from agent")
                    raise ValueError("Empty response from agent")
                    
            except Exception as e:
                logger.error(f"Error extracting output from agent response: {str(e)}\nRaw response type: {type(result)}\nRaw response: {result}", exc_info=True)
                output = f"I encountered an error processing the response: {str(e)}"
                
            # If output is empty, provide a default response
            if not output.strip():
                output = "I received your message but didn't generate a response. Could you try rephrasing your question?"
                
            logger.info(f"Agent {self.name} processing complete. Response length: {len(output) if output else 0} characters")
            
            return {
                'success': True,
                'output': output,
                'agent_used': self.name,
                'metadata': {
                    'model': self.config.model_name,
                    'temperature': self.config.temperature,
                    'tools_used': [tool.name for tool in self.config.tools] if hasattr(self.config, 'tools') else []
                }
            }
            
        except Exception as e:
            logger.error(f"Error in {self.name} agent: {str(e)}", exc_info=True)
            return {
                'success': False,
                'output': f"I encountered an error while processing your request with {self.name}.",
                'error': str(e),
                'agent_used': self.name,
                'traceback': str(traceback.format_exc())
            }

class ResearcherAgent(BaseAgent):
    """Agent specialized in finding and gathering information."""
    
    def __init__(self, tools: List[Any] = None):
        from llm.config import settings
        config = AgentConfig(
            name="researcher",
            description="Specialized in finding and gathering information from various sources.",
            model_name=settings.REASONING_MODEL,  # Use configured reasoning model
            temperature=0.7,
            system_prompt="""You are a research assistant. Your job is to find accurate and relevant information to answer questions. 
            Use the available tools to search for information. Be thorough and objective in your research. 
            Always cite your sources and provide links when available.""",
            tools=tools or [],
            max_iterations=5
        )
        super().__init__(config)

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
