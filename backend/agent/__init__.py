"""Agent module for the Naarad AI assistant.

This module contains the core agent implementations for the Naarad AI assistant,
including the base agent, domain-specific agents, and the agent orchestrator.
"""

# Import agent registry first to avoid circular imports
from .agent_registry import agent_registry, AgentRegistry, AgentInitializationError

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
    'agent_registry',
    'AgentRegistry',
    'AgentInitializationError'
]
