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

logger = logging.getLogger(__name__)

class ResponderAgent(BaseAgent):
    """Agent specialized in generating friendly and helpful responses.
    
    This agent handles general conversations and simple queries without requiring
    external tools. It's optimized for quick responses to common queries.
    """
    
    def __init__(self, config: Union[AgentConfig, Dict[str, Any]]):
        """Initialize the responder agent with configuration.
        
        Args:
            config: The configuration for the agent. Can be a dictionary or AgentConfig instance.
                   If None, default values will be used.
        """
        if not isinstance(config, (AgentConfig, dict)):
            raise ValueError("config must be an AgentConfig instance or a dictionary")
            
        if isinstance(config, dict):
            system_prompt = """You are a friendly and helpful AI assistant. Your primary goals are:
            1. Provide clear, concise, and helpful responses
            2. Be polite, empathetic, and engaging in your communication
            3. Be honest about the limits of your knowledge
            4. For simple queries (greetings, basic facts, etc.), respond directly without using tools
            5. Only use tools when explicitly needed for specific information
            
            Guidelines:
            - Keep responses conversational and natural
            - Be concise but thorough
            - Admit when you don't know something
            - For factual questions you're unsure about, say so rather than guessing
            - For simple greetings or thanks, respond appropriately without using tools
            - For queries about your capabilities, explain what you can do
            - For complex queries that require research, suggest using specific search terms
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
        
        # Initialize simple_queries if not set in config
        if not hasattr(self, 'simple_queries') or not self.simple_queries:
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
    
    def _process_impl(self, input_text: str, **kwargs) -> Dict[str, Any]:
        """Process input with the responder agent.
        
        Args:
            input_text: The input text to respond to
            **kwargs: Additional keyword arguments
            
        Returns:
            Dict containing the response and metadata
        """
        try:
            # Convert input to lowercase for case-insensitive matching
            input_lower = input_text.lower().strip()
            
            # Check for simple queries first
            for key, responses in self.simple_queries.items():
                if key in input_lower:
                    return {
                        'output': random.choice(responses),
                        'metadata': {
                            'agent': self.config.name,
                            'model': self.config.model_name,
                            'source': 'simple_query',
                            'confidence': 0.9
                        }
                    }
            
            # For other queries, generate a response using the agent
            agent = self.agent
            result = agent.invoke({
                'input': input_text,
                'chat_history': kwargs.get('chat_history', [])
            })
            
            # Extract the output from the result
            if isinstance(result, dict) and 'output' in result:
                output = result['output']
            elif isinstance(result, str):
                output = result
            else:
                output = str(result)
            
        except Exception as e:
            logger.error(f"Error in responder agent: {str(e)}", exc_info=True)
            return {
                'output': "I'm sorry, I encountered an error while processing your request. Please try again.",
                'error': str(e),
                'metadata': {
                    'agent': self.name,
                    'error': str(e),
                    'success': False
                }
            }
