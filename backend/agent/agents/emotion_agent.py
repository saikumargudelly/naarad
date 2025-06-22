"""Emotion Agent for detecting and responding to user emotions in real-time."""

from typing import Dict, Any, List, Optional, Tuple
import logging
import json
import re
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from .base import BaseAgent, AgentConfig

logger = logging.getLogger(__name__)

class EmotionType(Enum):
    """Supported emotion types."""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    NEUTRAL = "neutral"
    EXCITEMENT = "excitement"
    ANXIETY = "anxiety"
    CONFUSION = "confusion"
    FRUSTRATION = "frustration"
    SATISFACTION = "satisfaction"

@dataclass
class EmotionDetection:
    """Represents detected emotion with confidence and intensity."""
    emotion: EmotionType
    confidence: float
    intensity: float  # 0.0 to 1.0
    triggers: List[str]
    context: Dict[str, Any]

class EmotionAgent(BaseAgent):
    """Agent specialized in emotion detection and emotionally intelligent responses."""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.emotion_patterns = self._initialize_emotion_patterns()
        self.emotion_history = []
        self.response_templates = self._load_response_templates()
        
    def _initialize_emotion_patterns(self) -> Dict[EmotionType, List[Tuple[re.Pattern, float]]]:
        """Initialize regex patterns for emotion detection."""
        return {
            EmotionType.JOY: [
                (re.compile(r'\b(?:happy|joy|excited|great|wonderful|amazing|fantastic|awesome|brilliant|excellent)\b', re.IGNORECASE), 0.8),
                (re.compile(r'[!]{2,}|😊|😄|😃|😁|🎉|🎊|🎈'), 0.9),
                (re.compile(r'\b(?:love|adore|enjoy|pleased|delighted|thrilled|ecstatic)\b', re.IGNORECASE), 0.7),
            ],
            EmotionType.SADNESS: [
                (re.compile(r'\b(?:sad|depressed|unhappy|miserable|down|blue|melancholy|sorrowful|grief|heartbroken)\b', re.IGNORECASE), 0.8),
                (re.compile(r'😢|😭|😔|💔|😞|😥'), 0.9),
                (re.compile(r'\b(?:miss|lost|gone|alone|lonely|empty|hopeless)\b', re.IGNORECASE), 0.7),
            ],
            EmotionType.ANGER: [
                (re.compile(r'\b(?:angry|mad|furious|rage|irritated|annoyed|frustrated|outraged|livid|enraged)\b', re.IGNORECASE), 0.8),
                (re.compile(r'[!]{3,}|😠|😡|🤬|💢'), 0.9),
                (re.compile(r'\b(?:hate|despise|loathe|disgusted|sick of|tired of)\b', re.IGNORECASE), 0.7),
            ],
            EmotionType.FEAR: [
                (re.compile(r'\b(?:scared|afraid|terrified|frightened|panicked|worried|anxious|nervous|concerned|fearful)\b', re.IGNORECASE), 0.8),
                (re.compile(r'😨|😰|😱|😳|😟'), 0.9),
                (re.compile(r'\b(?:what if|oh no|help|danger|emergency|urgent)\b', re.IGNORECASE), 0.6),
            ],
            EmotionType.SURPRISE: [
                (re.compile(r'\b(?:wow|omg|oh my|unbelievable|incredible|shocking|amazing|astonishing|stunned|speechless)\b', re.IGNORECASE), 0.8),
                (re.compile(r'😲|😯|😱|🤯|😳'), 0.9),
                (re.compile(r'\b(?:really\?|seriously\?|no way|you\'re kidding)\b', re.IGNORECASE), 0.7),
            ],
            EmotionType.EXCITEMENT: [
                (re.compile(r'\b(?:can\'t wait|so excited|looking forward|thrilled|eager|enthusiastic|pumped|stoked)\b', re.IGNORECASE), 0.8),
                (re.compile(r'🎉|🎊|🎈|🚀|⚡|🔥'), 0.9),
                (re.compile(r'\b(?:finally|at last|yes|absolutely|definitely)\b', re.IGNORECASE), 0.6),
            ],
            EmotionType.ANXIETY: [
                (re.compile(r'\b(?:stress|overwhelmed|pressure|deadline|worried|concerned|uncertain|doubt|hesitant)\b', re.IGNORECASE), 0.8),
                (re.compile(r'😰|😟|😕|🤔|😬'), 0.8),
                (re.compile(r'\b(?:not sure|maybe|if only|wish|hope)\b', re.IGNORECASE), 0.6),
            ],
            EmotionType.FRUSTRATION: [
                (re.compile(r'\b(?:frustrated|annoyed|irritated|fed up|sick of|tired of|had enough|give up)\b', re.IGNORECASE), 0.8),
                (re.compile(r'😤|😫|😩|😖|😣'), 0.8),
                (re.compile(r'\b(?:why|how come|this is ridiculous|impossible|useless)\b', re.IGNORECASE), 0.7),
            ],
        }
    
    def _load_response_templates(self) -> Dict[EmotionType, List[str]]:
        """Load emotionally appropriate response templates."""
        return {
            EmotionType.JOY: [
                "I'm so glad you're feeling happy! 😊 What's bringing you joy today?",
                "Your enthusiasm is contagious! 🎉 Tell me more about what's making you excited.",
                "It's wonderful to see you in such good spirits! What would you like to explore or celebrate?"
            ],
            EmotionType.SADNESS: [
                "I can sense you're feeling down. I'm here to listen and support you. Would you like to talk about what's troubling you?",
                "It's okay to feel sad sometimes. You're not alone, and I'm here to help you through this.",
                "I hear the sadness in your words. Let's work through this together. What's on your mind?"
            ],
            EmotionType.ANGER: [
                "I can see you're frustrated, and that's completely valid. Let's take a moment to breathe and work through this together.",
                "Your feelings are important. Would you like to talk about what's causing this frustration?",
                "I understand you're upset. Let's address this calmly and find a solution together."
            ],
            EmotionType.FEAR: [
                "I can sense your concern. You're safe here, and we can work through this together. What's worrying you?",
                "It's natural to feel anxious sometimes. Let's break this down and find a way forward.",
                "I'm here to help you through this. Let's talk about what's causing your worry."
            ],
            EmotionType.SURPRISE: [
                "Wow! That's quite unexpected! 😲 Tell me more about what surprised you.",
                "That's definitely a plot twist! What are your thoughts on this development?",
                "I can feel your surprise! This is quite the revelation. How are you processing this?"
            ],
            EmotionType.EXCITEMENT: [
                "Your excitement is absolutely infectious! 🚀 I can't wait to hear more about this!",
                "This sounds amazing! Your enthusiasm is making me excited too! What's next?",
                "I love your energy! Let's channel this excitement into something great!"
            ],
            EmotionType.ANXIETY: [
                "I can sense your anxiety, and that's completely understandable. Let's work through this step by step.",
                "It's okay to feel uncertain. We can explore this together and find clarity.",
                "I'm here to support you through this. Let's break down what's causing your concern."
            ],
            EmotionType.FRUSTRATION: [
                "I can feel your frustration, and it's completely valid. Let's work through this together.",
                "I understand this is challenging. Let's take a different approach and find a solution.",
                "Your frustration is understandable. Let's address this systematically and find a way forward."
            ],
        }
    
    async def process(self, input_text: str, context: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """Process input with emotion detection and emotionally intelligent response."""
        try:
            emotions = self._detect_emotions(input_text, context or {})
            primary_emotion = self._get_primary_emotion(emotions)
            self._update_emotion_history(primary_emotion, input_text)
            response = self._generate_emotionally_intelligent_response(
                primary_emotion, input_text, context or {}
            )
            emotional_insights = self._analyze_emotional_patterns()
            # Contextual follow-up if multiple emotions/triggers
            triggers = primary_emotion.triggers if hasattr(primary_emotion, 'triggers') else []
            followup = ''
            if len(emotions) > 2 or (isinstance(triggers, list) and len(triggers) > 2):
                followup = self._contextual_followup(input_text, triggers if triggers else [e.emotion.value for e in emotions], domain='emotion')
                response += f"\n\n{followup}"
            return {
                "success": True,
                "output": response,
                "emotion_detection": {
                    "primary_emotion": primary_emotion.emotion.value,
                    "confidence": primary_emotion.confidence,
                    "intensity": primary_emotion.intensity,
                    "all_emotions": [e.emotion.value for e in emotions],
                    "triggers": primary_emotion.triggers
                },
                "emotional_insights": emotional_insights,
                "agent": "emotion_agent"
            }
        except Exception as e:
            logger.error(f"Error in emotion processing: {str(e)}", exc_info=True)
            return {
                "success": False,
                "output": "I'm here to support you. How can I help?",
                "error": str(e),
                "agent": "emotion_agent"
            }
    
    def _detect_emotions(self, text: str, context: Dict[str, Any]) -> List[EmotionDetection]:
        """Detect emotions in the input text."""
        emotions = []
        
        # Check each emotion pattern
        for emotion_type, patterns in self.emotion_patterns.items():
            confidence = 0.0
            triggers = []
            
            for pattern, weight in patterns:
                matches = pattern.findall(text)
                if matches:
                    confidence += weight
                    triggers.extend(matches)
            
            # Normalize confidence
            if confidence > 0:
                confidence = min(confidence / len(patterns), 1.0)
                intensity = self._calculate_intensity(text, emotion_type, confidence)
                
                emotions.append(EmotionDetection(
                    emotion=emotion_type,
                    confidence=confidence,
                    intensity=intensity,
                    triggers=triggers,
                    context=context
                ))
        
        # If no emotions detected, default to neutral
        if not emotions:
            emotions.append(EmotionDetection(
                emotion=EmotionType.NEUTRAL,
                confidence=0.5,
                intensity=0.3,
                triggers=[],
                context=context
            ))
        
        return emotions
    
    def _calculate_intensity(self, text: str, emotion: EmotionType, confidence: float) -> float:
        """Calculate emotion intensity based on text characteristics."""
        intensity = confidence
        
        # Adjust based on text length and punctuation
        if len(text) > 100:
            intensity *= 0.8  # Longer texts might be more complex
        
        # Check for emphasis markers
        emphasis_patterns = [
            (r'[!]{2,}', 0.2),  # Multiple exclamation marks
            (r'\b(?:very|really|extremely|absolutely|completely)\b', 0.15),  # Intensifiers
            (r'[A-Z]{3,}', 0.1),  # ALL CAPS
        ]
        
        for pattern, boost in emphasis_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                intensity = min(intensity + boost, 1.0)
        
        return intensity
    
    def _get_primary_emotion(self, emotions: List[EmotionDetection]) -> EmotionDetection:
        """Get the primary emotion based on confidence and intensity."""
        if not emotions:
            return EmotionDetection(
                emotion=EmotionType.NEUTRAL,
                confidence=0.5,
                intensity=0.3,
                triggers=[],
                context={}
            )
        
        # Score emotions by confidence * intensity
        scored_emotions = [
            (emotion, emotion.confidence * emotion.intensity)
            for emotion in emotions
        ]
        
        # Return the emotion with the highest score
        return max(scored_emotions, key=lambda x: x[1])[0]
    
    def _generate_emotionally_intelligent_response(
        self, 
        emotion: EmotionDetection, 
        input_text: str, 
        context: Dict[str, Any]
    ) -> str:
        """Generate an emotionally appropriate response."""
        templates = self.response_templates.get(emotion.emotion, [])
        
        if templates:
            # Select template based on context and emotion intensity
            if emotion.intensity > 0.8:
                # High intensity - use more supportive templates
                template = templates[0] if templates else "I understand how you're feeling."
            elif emotion.intensity > 0.5:
                # Medium intensity - use balanced templates
                template = templates[1] if len(templates) > 1 else templates[0]
            else:
                # Low intensity - use lighter templates
                template = templates[-1] if templates else "I'm here to help."
        else:
            template = "I'm here to support you. How can I help?"
        
        # Personalize based on context
        if context.get('conversation_history'):
            # Reference previous emotions if relevant
            recent_emotions = self._get_recent_emotions(5)
            if recent_emotions and recent_emotions[0].emotion != emotion.emotion:
                template += f" I notice your mood has shifted from {recent_emotions[0].emotion.value} to {emotion.emotion.value}. "
        
        return template
    
    def _update_emotion_history(self, emotion: EmotionDetection, text: str):
        """Update emotion history for pattern analysis."""
        self.emotion_history.append({
            'emotion': emotion.emotion.value,
            'confidence': emotion.confidence,
            'intensity': emotion.intensity,
            'timestamp': datetime.utcnow().isoformat(),
            'text_length': len(text)
        })
        
        # Keep only recent history (last 50 interactions)
        if len(self.emotion_history) > 50:
            self.emotion_history = self.emotion_history[-50:]
    
    def _get_recent_emotions(self, count: int = 5) -> List[EmotionDetection]:
        """Get recent emotions from history."""
        recent = self.emotion_history[-count:] if self.emotion_history else []
        return [
            EmotionDetection(
                emotion=EmotionType(entry['emotion']),
                confidence=entry['confidence'],
                intensity=entry['intensity'],
                triggers=[],
                context={}
            )
            for entry in recent
        ]
    
    def _analyze_emotional_patterns(self) -> Dict[str, Any]:
        """Analyze emotional patterns over time."""
        if not self.emotion_history:
            return {}
        
        # Calculate emotion frequency
        emotion_counts = {}
        for entry in self.emotion_history:
            emotion = entry['emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Find dominant emotion
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else None
        
        # Calculate average intensity
        avg_intensity = sum(entry['intensity'] for entry in self.emotion_history) / len(self.emotion_history)
        
        # Detect emotional volatility (frequent changes)
        emotion_changes = 0
        for i in range(1, len(self.emotion_history)):
            if self.emotion_history[i]['emotion'] != self.emotion_history[i-1]['emotion']:
                emotion_changes += 1
        
        volatility = emotion_changes / max(len(self.emotion_history) - 1, 1)
        
        return {
            'dominant_emotion': dominant_emotion,
            'average_intensity': avg_intensity,
            'emotional_volatility': volatility,
            'total_interactions': len(self.emotion_history),
            'emotion_distribution': emotion_counts
        } 