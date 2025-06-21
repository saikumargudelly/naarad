"""Agents package - main entry point.

This module provides access to all agent implementations and related components.
It's maintained for backward compatibility. New code should import directly from
the specific modules in the agents package.
"""

import logging
from typing import Dict, Any, List, Optional, Type, TypeVar, Union

# Import from the new modular structure
from .agents.base import AgentConfig, BaseAgent
from .agents.researcher import ResearcherAgent
from .agents.analyst import AnalystAgent
from .agents.responder import ResponderAgent
from .agents.quality import QualityAgent
from .factory import create_agent, create_default_agents, get_agent_class, AgentManager

# Set up logging
logger = logging.getLogger(__name__)

# Re-export types and classes for backward compatibility
AgentT = TypeVar('AgentT', bound=BaseAgent)

# Re-export agent classes
__all__ = [
    'AgentConfig',
    'BaseAgent',
    'ResearcherAgent',
    'AnalystAgent',
    'ResponderAgent',
    'QualityAgent',
    'create_agent',
    'create_default_agents',
    'get_agent_class',
    'AgentManager'
]
