"""
This module provides singleton instances of core application dependencies,
ensuring they are created lazily (i.e., only when first needed).
"""
import logging
from agent.naarad_agent import NaaradAgent
from agent.agents.voice_agent import VoiceAgent
from agent.agents.analytics_agent import AnalyticsAgent
from agent.agents.personalization_agent import PersonalizationAgent

logger = logging.getLogger(__name__)

# --- Lazy-loaded singleton for NaaradAgent ---
_naarad_agent_instance = None

def get_naarad_agent():
    """Get the singleton instance of the NaaradAgent, creating it if it doesn't exist."""
    global _naarad_agent_instance
    if _naarad_agent_instance is None:
        logger.info("Creating NaaradAgent singleton instance for the first time...")
        _naarad_agent_instance = NaaradAgent()
    return _naarad_agent_instance
# --------------------------------------------- 

# --- Lazy-loaded singleton for VoiceAgent ---
_voice_agent_instance = None

def get_voice_agent():
    """Get the singleton instance of the VoiceAgent."""
    global _voice_agent_instance
    if _voice_agent_instance is None:
        logger.info("Creating VoiceAgent singleton instance for the first time...")
        _voice_agent_instance = VoiceAgent({
            "name": "voice_agent",
            "description": "Voice processing agent",
            "model_name": "llama3-70b-8192"
        })
    return _voice_agent_instance
# --------------------------------------------

# --- Lazy-loaded singleton for AnalyticsAgent ---
_analytics_agent_instance = None

def get_analytics_agent():
    """Get the singleton instance of the AnalyticsAgent."""
    global _analytics_agent_instance
    if _analytics_agent_instance is None:
        logger.info("Creating AnalyticsAgent singleton instance for the first time...")
        _analytics_agent_instance = AnalyticsAgent({
            "name": "analytics_agent",
            "description": "Data analysis and insights agent",
            "model_name": "llama3-70b-8192"
        })
    return _analytics_agent_instance
# ------------------------------------------------

# --- Lazy-loaded singleton for PersonalizationAgent ---
_personalization_agent_instance = None

def get_personalization_agent():
    """Get the singleton instance of the PersonalizationAgent."""
    global _personalization_agent_instance
    if _personalization_agent_instance is None:
        logger.info("Creating PersonalizationAgent singleton instance for the first time...")
        _personalization_agent_instance = PersonalizationAgent({
            "name": "personalization_agent",
            "description": "User preference learning and personalization agent",
            "model_name": "llama3-70b-8192"
        })
    return _personalization_agent_instance
# ---------------------------------------------------- 