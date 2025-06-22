"""
Agent package initialization.

This package contains all agent implementations and related components.
"""

# Import all agent classes to make them available at package level
from .base import BaseAgent, AgentConfig
from .researcher import ResearcherAgent
from .analyst import AnalystAgent
from .responder import ResponderAgent
from .quality import QualityAgent

# Import new futuristic agents
from .voice_agent import VoiceAgent, SpeechRecognitionTool, TextToSpeechTool
from .personalization_agent import PersonalizationAgent, UserPreferenceTool

# Import factory functions
from ..factory import create_agent, AgentManager

__all__ = [
    # Base classes
    'BaseAgent',
    'AgentConfig',
    
    # Core agent implementations
    'ResearcherAgent',
    'AnalystAgent',
    'ResponderAgent',
    'QualityAgent',
    
    # Futuristic agents
    'VoiceAgent',
    'SpeechRecognitionTool',
    'TextToSpeechTool',
    'PersonalizationAgent',
    'UserPreferenceTool',
    
    # Factory functions
    'create_agent',
    'AgentManager'
]
