"""Agent module for the Naarad AI assistant.

This module contains the core agent implementations for the Naarad AI assistant,
including the base agent, domain-specific agents, and the agent orchestrator.
"""

# Import types first to avoid circular imports
from .types import AgentInitializationError, AgentConfig, BaseAgent

# Import agent registry (must come after types)
from .registry import AgentRegistry, agent_registry

# Import core agent components (lazy import in functions)
# Import domain agents (lazy import in functions)
# Import orchestrator (lazy import in functions)

# Export public API
__all__ = [
    # Core types
    'AgentInitializationError',
    'AgentConfig',
    'BaseAgent',
    
    # Core agents (imported lazily)
    'ResearcherAgent',
    'AnalystAgent',
    'ResponderAgent',
    'QualityAgent',
    
    # Domain agents (imported lazily)
    'DomainAgent',
    'TaskManagementAgent',
    'CreativeWritingAgent',
    'AnalysisAgent',
    'ContextAwareChatAgent',
    'create_domain_agents',
    
    # Orchestrator (imported lazily)
    'AgentOrchestrator',
    
    # Registry
    'agent_registry',
    'AgentRegistry',
    'AgentInitializationError'
]
