from typing import Dict, Any, List, Optional, Type, TypeVar
from dataclasses import dataclass, field
import logging
import os
from dotenv import load_dotenv

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Type variable for agent classes
AgentT = TypeVar('AgentT', bound='BaseAgent')

@dataclass
class AgentConfig:
    """Configuration for an agent.
    
    Models are configured to use OpenRouter with the following mapping:
    - Language Reasoning: groq/mixtral-8x7b-32768 (GROQ Mixtral)
    - Personality Chat: nousresearch/nous-hermes-2-mixtral-8x7b-dpo
    - Default: groq/mixtral-8x7b-32768 (GROQ Mixtral)
    """
    name: str
    description: str
    model_name: str = "groq/mixtral-8x7b-32768"  # Default to GROQ Mixtral via OpenRouter
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
        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.config.system_prompt),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Initialize the language model
        llm = ChatOpenAI(
            model_name=self.config.model_name,
            temperature=self.config.temperature,
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
            headers={
                "HTTP-Referer": "https://naarad-ai.com",
                "X-Title": "Naarad AI Assistant"
            }
        )
        
        # Create the agent
        agent = create_openai_tools_agent(llm, self.config.tools, prompt)
        
        # Return the executor
        return AgentExecutor(
            agent=agent, 
            tools=self.config.tools, 
            verbose=self.config.verbose,
            max_iterations=self.config.max_iterations,
            handle_parsing_errors=True
        )
    
    async def process(self, input_text: str, **kwargs) -> Dict[str, Any]:
        """Process input using the agent."""
        try:
            result = await self.agent.ainvoke({"input": input_text, **kwargs})
            return {
                'success': True,
                'output': result.get('output', ''),
                'metadata': {
                    'agent': self.name,
                    'model': self.config.model_name,
                    'tokens_used': len(input_text) + len(result.get('output', '')),  # Rough estimate
                }
            }
        except Exception as e:
            logger.error(f"Error in {self.name} agent: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'agent': self.name
            }

class ResearcherAgent(BaseAgent):
    """Agent specialized in finding and gathering information."""
    
    def __init__(self, tools: List[Any] = None):
        config = AgentConfig(
            name="researcher",
            description="Specialized in finding and gathering information from various sources.",
            model_name="groq/mixtral-8x7b-32768",  # GROQ Mixtral for research
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
        config = AgentConfig(
            name="analyst",
            description="Specialized in analyzing information and providing insights.",
            model_name="groq/mixtral-8x7b-32768",  # GROQ Mixtral for analysis
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
        config = AgentConfig(
            name="responder",
            description="Specialized in generating friendly and helpful responses.",
            model_name="nousresearch/nous-hermes-2-mixtral-8x7b-dpo",  # Nous Hermes for personality
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
        config = AgentConfig(
            name="quality",
            description="Specialized in refining and improving responses for quality and clarity.",
            model_name="groq/mixtral-8x7b-32768",  # GROQ Mixtral for quality control
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
