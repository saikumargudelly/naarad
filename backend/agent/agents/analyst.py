"""Analyst agent implementation.

This module contains the AnalystAgent class which is specialized in analyzing
information and providing insights.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Union, Type

# LangChain imports
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough

# Local imports
from .base import BaseAgent, AgentConfig
from ..tools.dummy_tool import DummyTool  # Import dummy tool
from llm.config import settings
from agent.memory.memory_manager import MemoryManager

logger = logging.getLogger(__name__)

class AnalystAgent(BaseAgent):
    """Agent specialized in analyzing information and providing insights.
    Modular, stateless, and uses injected memory manager for context/state.
    """
    
    def __init__(self, config: Union[AgentConfig, Dict[str, Any]], memory_manager: MemoryManager = None):
        """Initialize the analyst agent with configuration.
        
        Args:
            config: The configuration for the agent. Can be a dictionary or AgentConfig instance.
                   If None, default values will be used.
            memory_manager: The memory manager for context and state management
        """
        if not isinstance(config, (AgentConfig, dict)):
            raise ValueError("config must be an AgentConfig instance or a dictionary")
            
        if isinstance(config, dict):
            # Accept routing_info from context if present
            routing_info = config.get('routing_info', {})
            intent = routing_info.get('intent', 'unknown')
            confidence = routing_info.get('confidence', None)
            entities = routing_info.get('entities', {})
            routing_str = f"\n[Routing Info]\nIntent: {intent} (confidence: {confidence})\nEntities: {entities}\n" if intent != 'unknown' else ''
            # Always define system_prompt first
            system_prompt = f"""
{routing_str}
You are an expert analytical assistant. Your responsibilities are:

1. Thoroughly analyze the provided information, breaking down complex data into clear, actionable insights.
2. Always consider multiple perspectives and highlight key findings, trends, and implications.
3. Use available tools for data analysis when appropriate, and clearly explain your methodology and reasoning.
4. Structure your responses with clear headings, bullet points, and concise summaries for maximum clarity.
5. If information is incomplete or ambiguous, state your assumptions and suggest follow-up questions or next steps.
6. Always cite sources or reference data when possible.
7. For ambiguous or broad queries, ask clarifying questions before proceeding with analysis.
8. Never fabricate data or analysis; be transparent about any limitations or uncertainties.

If the query is outside your analytical domain or the intent confidence is low, escalate to the appropriate agent or ask the user for clarification before proceeding. If you cannot help, say so and suggest the correct agent or next step.
"""
            default_config = {
                'name': 'analyst',
                'description': 'Specialized in analyzing information and providing insights.',
                'model_name': settings.REASONING_MODEL,
                'temperature': 0.5,
                'system_prompt': system_prompt,
                'max_iterations': 5
            }
            # If no tools are provided, use a dummy tool
            if 'tools' not in config or not config['tools']:
                from ..tools.dummy_tool import DummyTool
                default_config['tools'] = [DummyTool()]
            # Now update with any provided config values
            default_config.update(config)
            config = default_config
            
        super().__init__(config)
        self.memory_manager = memory_manager
        logger.info(f"AnalystAgent initialized with memory_manager: {bool(memory_manager)}")
    
    def _create_agent(self):
        """Create and configure the LangChain agent for analysis tasks.
        
        Returns:
            Configured agent executor
            
        Raises:
            RuntimeError: If agent creation fails
        """
        try:
            from langchain.agents import AgentExecutor, create_react_agent
            from langchain_core.prompts import ChatPromptTemplate
            
            # Check for required environment variable
            groq_api_key = os.getenv('GROQ_API_KEY')
            if not groq_api_key:
                logger.warning("GROQ_API_KEY not found in environment variables. Using a fallback model.")
                from langchain_openai import ChatOpenAI
                
                llm = ChatOpenAI(
                    temperature=self.config.temperature,
                    model_name='gpt-3.5-turbo',
                    openai_api_key=os.getenv('OPENAI_API_KEY')
                )
            else:
                from langchain_groq import ChatGroq
                llm = ChatGroq(
                    temperature=self.config.temperature,
                    model_name=self.config.model_name,
                    groq_api_key=groq_api_key
                )
            
            # Define the prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.config.system_prompt),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="intermediate_steps"),
            ])
            
            # Get tools or use a dummy tool
            tools = self.config.tools or [DummyTool()]
            
            # Create the agent
            agent = create_react_agent(
                llm=llm,
                tools=tools,
                prompt=prompt
            )
            
            # Create the agent executor
            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=self.config.verbose,
                max_iterations=self.config.max_iterations,
                handle_parsing_errors=True,
                return_intermediate_steps=True
            )
            
            return agent_executor
            
        except ImportError as e:
            logger.error(f"Required package not found: {str(e)}")
            # Fallback to a simple agent without external dependencies
            from langchain.llms.fake import FakeListLLM
            from langchain.agents import AgentExecutor, create_react_agent
            from langchain_core.prompts import ChatPromptTemplate
            
            logger.warning("Falling back to a simple agent implementation")
            llm = FakeListLLM(responses=["I'm a simple analyst agent. For a better experience, please install the required dependencies."])
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.config.system_prompt),
                ("human", "{input}"),
            ])
            
            agent = create_react_agent(
                llm=llm,
                tools=[DummyTool()],
                prompt=prompt
            )
            
            return AgentExecutor(
                agent=agent,
                tools=[DummyTool()],
                verbose=self.config.verbose,
                handle_parsing_errors=True
            )
            
        except Exception as e:
            logger.error(f"Failed to create analyst agent: {str(e)}", exc_info=True)
            # Return a minimal working agent that won't fail
            from langchain.llms.fake import FakeListLLM
            from langchain.agents import AgentExecutor, create_react_agent
            from langchain_core.prompts import ChatPromptTemplate
            
            logger.warning("Using fallback agent due to initialization error")
            llm = FakeListLLM(responses=["I'm having trouble analyzing this right now. Please try again later."])
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.config.system_prompt),
                ("human", "{input}"),
            ])
            
            agent = create_react_agent(
                llm=llm,
                tools=[DummyTool()],
                prompt=prompt
            )
            
            return AgentExecutor(
                agent=agent,
                tools=[DummyTool()],
                verbose=self.config.verbose,
                handle_parsing_errors=True
            )
    
    async def process(self, input_text: str, context: Dict[str, Any] = None, conversation_id: str = None, user_id: str = None, conversation_memory=None, **kwargs) -> Dict[str, Any]:
        logger.info(f"AnalystAgent.process called | input_text: {input_text} | conversation_id: {conversation_id} | user_id: {user_id}")
        try:
            chat_history = kwargs.get('chat_history', '')
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
            # Compose a context-aware prompt
            context_snippets = "\n".join([
                f"{m['role'].capitalize()}: {m['content']}" for m in conversation_memory.messages[-6:]
            ]) if conversation_memory else ""
            system_prompt = (
                "You are an analyst assistant. Use the conversation context, topic, and intent to answer the user's question as accurately and helpfully as possible. "
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
            logger.error(f"Async error in analyst process: {str(e)}", exc_info=True)
            return {
                'output': f"I encountered an error while processing your analyst request: {str(e)}",
                'metadata': {
                    'error': str(e),
                    'success': False
                }
            }
