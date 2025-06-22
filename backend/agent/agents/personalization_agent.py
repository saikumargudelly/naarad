"""Personalization Agent for learning user preferences and providing personalized experiences."""

from typing import Dict, Any, Optional, Union, List, Set
import asyncio
import logging
import json
import hashlib
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# LangChain imports
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseChatModel

# Local imports
from .base import BaseAgent, AgentConfig
from llm.config import settings

logger = logging.getLogger(__name__)

class UserPreferenceTool(BaseTool):
    """Tool for learning and managing user preferences."""
    
    name: str = "user_preference"
    description: str = "Learn and manage user preferences for personalized experiences"
    
    def _run(self, user_id: str, action: str, data: Dict[str, Any] = None) -> str:
        """Manage user preferences.
        
        Args:
            user_id: Unique identifier for the user
            action: Action to perform (learn, get, update, analyze)
            data: Additional data for the action
            
        Returns:
            str: Result of the preference operation
        """
        try:
            if action == "learn":
                return self._learn_preferences(user_id, data)
            elif action == "get":
                return self._get_preferences(user_id)
            elif action == "update":
                return self._update_preferences(user_id, data)
            elif action == "analyze":
                return self._analyze_preferences(user_id)
            else:
                return f"Unknown action: {action}"
                
        except Exception as e:
            logger.error(f"Error in user preference tool: {e}")
            return f"Error managing preferences: {str(e)}"
    
    def _learn_preferences(self, user_id: str, data: Dict[str, Any]) -> str:
        """Learn user preferences from interaction data."""
        try:
            # Extract preference signals from data
            preferences = self._extract_preferences(data)
            
            # Store preferences (in a real app, this would be in a database)
            # For now, we'll use a simple in-memory storage
            if not hasattr(self, '_user_preferences'):
                self._user_preferences = {}
            
            if user_id not in self._user_preferences:
                self._user_preferences[user_id] = {
                    'topics': Counter(),
                    'interaction_style': Counter(),
                    'response_length': [],
                    'time_of_day': Counter(),
                    'tools_used': Counter(),
                    'last_updated': datetime.utcnow().isoformat()
                }
            
            # Update preferences
            user_prefs = self._user_preferences[user_id]
            
            # Update topic preferences
            if 'topics' in preferences:
                for topic, weight in preferences['topics'].items():
                    user_prefs['topics'][topic] += weight
            
            # Update interaction style
            if 'interaction_style' in preferences:
                for style, weight in preferences['interaction_style'].items():
                    user_prefs['interaction_style'][style] += weight
            
            # Update response length preference
            if 'response_length' in preferences:
                user_prefs['response_length'].append(preferences['response_length'])
                # Keep only last 10 responses for average
                if len(user_prefs['response_length']) > 10:
                    user_prefs['response_length'] = user_prefs['response_length'][-10:]
            
            # Update time of day preference
            if 'time_of_day' in preferences:
                user_prefs['time_of_day'][preferences['time_of_day']] += 1
            
            # Update tools used
            if 'tools_used' in preferences:
                for tool in preferences['tools_used']:
                    user_prefs['tools_used'][tool] += 1
            
            user_prefs['last_updated'] = datetime.utcnow().isoformat()
            
            return f"Successfully learned preferences for user {user_id}"
            
        except Exception as e:
            logger.error(f"Error learning preferences: {e}")
            return f"Error learning preferences: {str(e)}"
    
    def _extract_preferences(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract preference signals from interaction data."""
        preferences = {}
        
        # Extract topics from message content
        if 'message' in data:
            message = data['message'].lower()
            
            # Simple topic extraction (in production, use NLP)
            topics = {
                'technology': ['tech', 'programming', 'code', 'software', 'computer'],
                'science': ['science', 'research', 'experiment', 'study'],
                'business': ['business', 'company', 'market', 'finance', 'money'],
                'health': ['health', 'medical', 'fitness', 'exercise', 'diet'],
                'entertainment': ['movie', 'music', 'game', 'entertainment', 'fun'],
                'education': ['learn', 'study', 'education', 'course', 'school']
            }
            
            found_topics = []
            for topic, keywords in topics.items():
                if any(keyword in message for keyword in keywords):
                    found_topics.append(topic)
            
            if found_topics:
                preferences['topics'] = {topic: 1 for topic in found_topics}
        
        # Extract interaction style
        if 'message' in data:
            message = data['message']
            if len(message) < 50:
                preferences['interaction_style'] = {'concise': 1}
            elif len(message) > 200:
                preferences['interaction_style'] = {'detailed': 1}
            else:
                preferences['interaction_style'] = {'balanced': 1}
        
        # Extract response length preference
        if 'response_length' in data:
            preferences['response_length'] = data['response_length']
        
        # Extract time of day
        current_hour = datetime.utcnow().hour
        if 6 <= current_hour < 12:
            time_of_day = 'morning'
        elif 12 <= current_hour < 17:
            time_of_day = 'afternoon'
        elif 17 <= current_hour < 21:
            time_of_day = 'evening'
        else:
            time_of_day = 'night'
        preferences['time_of_day'] = time_of_day
        
        # Extract tools used
        if 'tools_used' in data:
            preferences['tools_used'] = data['tools_used']
        
        return preferences
    
    def _get_preferences(self, user_id: str) -> str:
        """Get user preferences."""
        if not hasattr(self, '_user_preferences') or user_id not in self._user_preferences:
            return "No preferences found for this user"
        
        prefs = self._user_preferences[user_id]
        
        # Format preferences for display
        result = []
        result.append(f"User Preferences for {user_id}:")
        
        if prefs['topics']:
            top_topics = prefs['topics'].most_common(3)
            result.append(f"Favorite topics: {', '.join([topic for topic, _ in top_topics])}")
        
        if prefs['interaction_style']:
            top_style = prefs['interaction_style'].most_common(1)[0][0]
            result.append(f"Preferred interaction style: {top_style}")
        
        if prefs['response_length']:
            avg_length = np.mean(prefs['response_length'])
            result.append(f"Preferred response length: {avg_length:.0f} words")
        
        if prefs['time_of_day']:
            top_time = prefs['time_of_day'].most_common(1)[0][0]
            result.append(f"Most active time: {top_time}")
        
        if prefs['tools_used']:
            top_tools = prefs['tools_used'].most_common(3)
            result.append(f"Most used tools: {', '.join([tool for tool, _ in top_tools])}")
        
        return "\n".join(result)
    
    def _update_preferences(self, user_id: str, data: Dict[str, Any]) -> str:
        """Update user preferences."""
        return self._learn_preferences(user_id, data)
    
    def _analyze_preferences(self, user_id: str) -> str:
        """Analyze user preferences and provide insights."""
        if not hasattr(self, '_user_preferences') or user_id not in self._user_preferences:
            return "No preferences to analyze"
        
        prefs = self._user_preferences[user_id]
        
        insights = []
        insights.append("User Preference Analysis:")
        
        # Topic diversity
        topic_count = len(prefs['topics'])
        if topic_count > 5:
            insights.append("- User shows diverse interests across many topics")
        elif topic_count > 2:
            insights.append("- User has moderate topic diversity")
        else:
            insights.append("- User has focused interests in specific topics")
        
        # Interaction style consistency
        if prefs['interaction_style']:
            top_style = prefs['interaction_style'].most_common(1)[0]
            total_interactions = sum(prefs['interaction_style'].values())
            consistency = top_style[1] / total_interactions
            
            if consistency > 0.8:
                insights.append("- User has very consistent interaction style")
            elif consistency > 0.6:
                insights.append("- User has moderately consistent interaction style")
            else:
                insights.append("- User has varied interaction styles")
        
        # Time patterns
        if prefs['time_of_day']:
            top_time = prefs['time_of_day'].most_common(1)[0][0]
            insights.append(f"- User is most active during {top_time}")
        
        return "\n".join(insights)

class PersonalizationAgent(BaseAgent):
    """Agent specialized in personalization and user preference learning."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the personalization agent with configuration.
        Args:
            config: The configuration for the agent. Must be a dictionary.
        """
        # Set default values if not provided
        default_config = {
            'name': 'personalization_agent',
            'description': 'User preference learning and personalization agent',
            'model_name': settings.REASONING_MODEL,
            'temperature': 0.2,
            'system_prompt': """You are a personalization agent. Your job is to learn user preferences, adapt responses, and provide personalized experiences.\n\nYou can analyze user interactions, update preferences, and validate learning algorithms.""",
            'max_iterations': 5
        }
        # Update with any provided config values
        default_config.update(config)
        config = default_config
        super().__init__(config)
        
        # Add personalization-specific tools
        self.user_preference = UserPreferenceTool()
        
        # Personalization-specific configuration
        self.preference_learning_enabled = True
        self.personalization_threshold = 0.7  # Minimum confidence for personalization
        
        logger.info("Personalization Agent initialized")
    
    async def learn_from_interaction(
        self, 
        user_id: str,
        interaction_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Learn from user interaction to improve personalization."""
        
        try:
            if not self.preference_learning_enabled:
                return {"success": True, "message": "Preference learning disabled"}
            
            # Learn preferences from interaction
            result = self.user_preference._run(user_id, "learn", interaction_data)
            
            return {
                "success": True,
                "message": result,
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error learning from interaction: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_personalized_response(
        self, 
        user_id: str,
        base_response: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Get a personalized version of a response based on user preferences."""
        
        try:
            # Get user preferences
            prefs_result = self.user_preference._run(user_id, "get")
            
            if "No preferences found" in prefs_result:
                return {
                    "success": True,
                    "response": base_response,
                    "personalized": False,
                    "reason": "No user preferences available"
                }
            
            # Personalize response based on preferences
            personalized_response = self._personalize_response(
                base_response, prefs_result, context
            )
            
            return {
                "success": True,
                "response": personalized_response,
                "personalized": True,
                "preferences_used": self._extract_preference_signals(prefs_result),
                "original_response": base_response
            }
            
        except Exception as e:
            logger.error(f"Error personalizing response: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": base_response  # Fallback to original response
            }
    
    def _personalize_response(self, response: str, preferences: str, context: Dict[str, Any]) -> str:
        """Personalize a response based on user preferences."""
        personalized = response
        
        # Adjust response length based on preference
        if "concise" in preferences.lower():
            # Make response more concise
            sentences = response.split('. ')
            if len(sentences) > 2:
                personalized = '. '.join(sentences[:2]) + '.'
        elif "detailed" in preferences.lower():
            # Make response more detailed
            if len(response) < 200:
                personalized = response + " Would you like me to elaborate on any specific aspect?"
        
        # Add topic-specific personalization
        if "technology" in preferences.lower():
            if "tech" not in response.lower() and "technology" not in response.lower():
                personalized += " From a technical perspective, this involves several key considerations."
        
        # Add time-based personalization
        current_hour = datetime.utcnow().hour
        if 6 <= current_hour < 12:
            personalized = "Good morning! " + personalized
        elif 17 <= current_hour < 21:
            personalized = "Good evening! " + personalized
        
        return personalized
    
    def _extract_preference_signals(self, preferences: str) -> List[str]:
        """Extract key preference signals from preferences string."""
        signals = []
        
        if "concise" in preferences.lower():
            signals.append("prefers_concise_responses")
        if "detailed" in preferences.lower():
            signals.append("prefers_detailed_responses")
        if "technology" in preferences.lower():
            signals.append("interested_in_technology")
        if "morning" in preferences.lower():
            signals.append("morning_user")
        if "evening" in preferences.lower():
            signals.append("evening_user")
        
        return signals
    
    async def get_user_insights(self, user_id: str) -> Dict[str, Any]:
        """Get insights about user behavior and preferences."""
        
        try:
            # Get preference analysis
            analysis = self.user_preference._run(user_id, "analyze")
            
            # Get current preferences
            preferences = self.user_preference._run(user_id, "get")
            
            return {
                "success": True,
                "analysis": analysis,
                "preferences": preferences,
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting user insights: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user preferences in a format suitable for the API."""
        
        try:
            # Get current preferences
            preferences_result = self.user_preference._run(user_id, "get")
            
            # Get analysis for insights
            analysis_result = self.user_preference._run(user_id, "analyze")
            
            # Parse preferences into structured format
            preferences = {}
            if "No preferences found" not in preferences_result:
                # Extract structured data from the text result
                lines = preferences_result.split('\n')
                for line in lines:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip().lower().replace(' ', '_')
                        value = value.strip()
                        preferences[key] = value
            
            return {
                "success": True,
                "preferences": preferences,
                "learning_insights": analysis_result,
                "confidence_scores": {
                    "topic_preference": 0.8 if preferences else 0.0,
                    "style_preference": 0.7 if preferences else 0.0,
                    "time_preference": 0.6 if preferences else 0.0
                },
                "last_updated": datetime.utcnow().isoformat(),
                "user_id": user_id
            }
            
        except Exception as e:
            logger.error(f"Error getting user preferences: {e}")
            return {
                "success": False,
                "error": str(e),
                "preferences": {},
                "learning_insights": "",
                "confidence_scores": {},
                "last_updated": datetime.utcnow().isoformat(),
                "user_id": user_id
            }
    
    def enable_preference_learning(self, enabled: bool = True):
        """Enable or disable preference learning."""
        self.preference_learning_enabled = enabled
        logger.info(f"Preference learning {'enabled' if enabled else 'disabled'}")
    
    def set_personalization_threshold(self, threshold: float):
        """Set the confidence threshold for personalization."""
        self.personalization_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"Personalization threshold set to {self.personalization_threshold}") 