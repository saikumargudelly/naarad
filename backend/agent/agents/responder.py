"""Responder agent implementation.

This module contains the ResponderAgent class which is specialized in generating
friendly and helpful responses to user queries.
"""

import logging
import random
import os
from typing import Any, Dict, List, Optional, Union, Literal

# LangChain imports
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough

# Local imports
from .base import BaseAgent, AgentConfig
from llm.config import settings
from agent.memory.memory_manager import MemoryManager

logger = logging.getLogger(__name__)

class ResponderAgent(BaseAgent):
    """Agent specialized in generating friendly and helpful responses.
    Modular, stateless, and uses injected memory manager for context/state.
    """
    
    def __init__(self, config: Dict[str, Any], memory_manager: MemoryManager = None):
        """Initialize the responder agent with configuration.
        
        Args:
            config: The configuration for the agent. Must be a dictionary.
            memory_manager: The memory manager for the agent.
        """
        routing_info = config.get('routing_info', {}) if isinstance(config, dict) else {}
        intent = routing_info.get('intent', 'unknown')
        confidence = routing_info.get('confidence', None)
        entities = routing_info.get('entities', {})
        routing_str = f"\n[Routing Info]\nIntent: {intent} (confidence: {confidence})\nEntities: {entities}\n" if intent != 'unknown' else ''
        system_prompt = f"""
{routing_str}
You are a friendly, helpful, and context-aware AI assistant. Your primary goals are:

1. Provide clear, concise, and helpful responses tailored to the user's context and previous conversation.
2. Be polite, empathetic, and engaging, adapting your tone to the user's mood or situation.
3. For simple queries (greetings, basic facts, etc.), respond directly and conversationally without using tools.
4. For complex, ambiguous, or research-based queries, escalate to the appropriate agent or suggest using research/analysis tools.
5. Always be honest about the limits of your knowledge and capabilities.
6. If unsure, ask clarifying questions or suggest next steps to help the user get what they need.
7. Never fabricate information; be transparent about uncertainties or lack of knowledge.
8. Summarize or recap context when helpful, especially in multi-turn conversations.

Guidelines:
- Keep responses conversational and natural.
- Be concise but thorough.
- Admit when you don't know something or when a tool/agent is needed.
- For factual questions you're unsure about, say so rather than guessing.
- For queries about your capabilities, explain what you can do and when you escalate to other agents.
- For complex queries, suggest using research or analysis tools if needed.

If the query is outside your responder domain or the intent confidence is low, escalate to the appropriate agent or ask the user for clarification before proceeding. If you cannot help, say so and suggest the correct agent or next step.
"""
        # Common greetings and simple queries that don't need tools
        self.simple_queries = {
            # Greetings
            "hello": ["Hi there!", "Hello!", "Hi! How can I help you today?"],
            "hi": ["Hello!", "Hi there!", "Hey!"],
            "hey": ["Hi!", "Hello!", "Hey there!"],
            "good morning": ["Good morning!", "Morning! How can I assist you today?"],
            "good afternoon": ["Good afternoon!", "Afternoon! How can I help?"],
            "good evening": ["Good evening!", "Evening! How can I assist you?"],
            
            # Thanks
            "thank": ["You're welcome!", "Happy to help!", "No problem!"],
            "thanks": ["You're welcome!", "Anytime!", "Glad I could help!"],
            "appreciate": ["You're very welcome!", "Happy to assist!", "My pleasure!"],
            
            # Goodbyes
            "bye": ["Goodbye!", "See you later!", "Take care!"],
            "goodbye": ["Goodbye!", "Have a great day!", "See you next time!"],
            "see you": ["See you!", "Take care!", "Have a great day!"],
            
            # About you
            "who are you": ["I'm Naarad, your AI assistant. I'm here to help answer your questions and assist with various tasks.",
                          "I'm Naarad, an AI assistant designed to help with information and tasks. How can I assist you today?"],
            "what can you do": ["I can help with answering questions, finding information, comparing products, and more. What would you like to know?",
                              "I can assist with research, answer questions, and help with various tasks. Just let me know what you need!"],
            "your name": ["I'm Naarad, your AI assistant!", "You can call me Naarad. How can I help you today?"],
            
            # Help
            "help": ["I'm here to help! What do you need assistance with?", 
                   "How can I assist you today? I can help with questions, research, and more."],
            
            # General responses
            "how are you": ["I'm just a computer program, so I don't have feelings, but I'm here and ready to help you!",
                          "I'm functioning well, thank you for asking! How can I assist you today?"],
            "what's up": ["Not much, just here to help you! What can I do for you today?",
                        "Just waiting to assist you! What's on your mind?"]
        }
        
        # Set default values if not provided
        default_config = {
            'name': 'responder',
            'description': 'Specialized in generating friendly and helpful responses to user queries.',
            'model_name': settings.CHAT_MODEL,
            'temperature': 0.7,  # Slightly higher temperature for more varied responses
            'system_prompt': system_prompt,
            'max_iterations': 3  # Fewer iterations for faster responses
        }
        # Update with any provided config values
        default_config.update(config)
        config = default_config
        super().__init__(config)
        self.memory_manager = memory_manager
        logger.info(f"ResponderAgent initialized with memory_manager: {bool(memory_manager)}")
        
    def _is_simple_query(self, query: str) -> bool:
        """Check if the query is simple enough to handle without tools.
        
        Args:
            query: The user's input query
            
        Returns:
            bool: True if this is a simple query that can be handled directly
        """
        if not query or not isinstance(query, str):
            return False
            
        query = query.lower().strip()
        
        # Check against known simple queries
        if any(simple in query for simple in self.simple_queries):
            return True
            
        # Check for capability questions
        if any(term in query for term in ['what can you do', 'help', 'capabilities']):
            return True
            
        return False
    
    def _get_simple_response(self, query: str) -> str:
        """Generate a response for simple queries without using tools.
        
        Args:
            query: The user's input query
            
        Returns:
            str: A suitable response
        """
        query = query.lower().strip()
        
        # Check for exact matches first
        for key, responses in self.simple_queries.items():
            if key in query:
                return random.choice(responses)
        
        # Check for partial matches
        for key, responses in self.simple_queries.items():
            if any(word in query.split() for word in key.split()):
                return random.choice(responses)
                
        # Default response for other simple queries
        return "I'm here to help! Is there something specific you'd like to know?"
    
    def _create_agent(self):
        """Create and configure the LangChain agent for response generation.
        
        Returns:
            Configured agent executor
        """
        try:
            # Import here to avoid circular imports
            from langchain_groq import ChatGroq
            from langchain.chains import LLMChain
            from langchain.prompts import PromptTemplate
            
            # Initialize the language model
            llm = ChatGroq(
                temperature=self.config.temperature,
                model_name=self.config.model_name,
                groq_api_key=os.getenv('GROQ_API_KEY')
            )
            
            # Create a simple LLM chain for response generation
            prompt = PromptTemplate(
                input_variables=["input"],
                template="""You are a helpful AI assistant. Respond to the user in a friendly and helpful manner.
                
                User: {input}
                Assistant:"""
            )
            
            return LLMChain(llm=llm, prompt=prompt)
            
        except Exception as e:
            logger.error(f"Failed to create responder agent: {str(e)}", exc_info=True)
            # Fallback to a simple LLM chain instead of a complex agent
            from langchain_community.llms import FakeListLLM
            from langchain.chains import LLMChain
            from langchain.prompts import PromptTemplate
            
            logger.warning("Using fallback responder agent")
            llm = FakeListLLM(responses=["I'm here to help! How can I assist you today?"])
            
            # Create a simple LLM chain for the fallback
            prompt = PromptTemplate(
                input_variables=["input"],
                template="You are a helpful AI assistant. User: {input}\nAssistant: I'm here to help! How can I assist you today?"
            )
            
            return LLMChain(llm=llm, prompt=prompt)
    
    async def process(self, input_text: str, context: Dict[str, Any] = None, conversation_id: str = None, user_id: str = None, conversation_memory=None, **kwargs) -> Dict[str, Any]:
        logger.info(f"ResponderAgent.process called | input_text: {input_text} | conversation_id: {conversation_id} | user_id: {user_id}")
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
                "You are a helpful, context-aware assistant. Use the conversation context, topic, and intent to answer the user's question as accurately and helpfully as possible. "
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
            logger.error(f"Async error in responder process: {str(e)}", exc_info=True)
            return {
                'output': f"I encountered an error while processing your request: {str(e)}",
                'metadata': {
                    'error': str(e),
                    'success': False
                }
            }
