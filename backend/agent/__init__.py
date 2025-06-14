"""Agent module for the Naarad AI assistant.

This module contains the core agent implementations for the Naarad AI assistant,
including the base agent, domain-specific agents, and the agent orchestrator.
"""

# Import core agent components
from .agents import (
    BaseAgent,
    ResearcherAgent,
    AnalystAgent,
    ResponderAgent,
    QualityAgent
)

# Import domain agents
from .domain_agents import (
    DomainAgent,
    TaskManagementAgent,
    CreativeWritingAgent,
    AnalysisAgent,
    ContextAwareChatAgent,
    create_domain_agents
)

# Import orchestrator
from .orchestrator import AgentOrchestrator

# Import registry
from .registry import AgentRegistry, AgentInitializationError

# Export public API
__all__ = [
    # Core agents
    'BaseAgent',
    'ResearcherAgent',
    'AnalystAgent',
    'ResponderAgent',
    'QualityAgent',
    
    # Domain agents
    'DomainAgent',
    'TaskManagementAgent',
    'CreativeWritingAgent',
    'AnalysisAgent',
    'ContextAwareChatAgent',
    'create_domain_agents',
    
    # Orchestrator
    'AgentOrchestrator',
    
    # Registry
    'AgentRegistry',
    'AgentInitializationError'
]
