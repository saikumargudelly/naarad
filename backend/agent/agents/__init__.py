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

# Import factory functions
from ..factory import create_agent, AgentManager

__all__ = [
    # Base classes
    'BaseAgent',
    'AgentConfig',
    
    # Agent implementations
    'ResearcherAgent',
    'AnalystAgent',
    'ResponderAgent',
    'QualityAgent',
    
    # Factory functions
    'create_agent',
    'AgentManager'
]
