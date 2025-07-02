"""Voice Agent for speech recognition and text-to-speech capabilities."""

from typing import Dict, Any, Optional, Union, List
import asyncio
import logging
import json
import base64
import io
from pathlib import Path
import tempfile
import os

# LangChain imports
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseChatModel

# Local imports
from .base import BaseAgent, AgentConfig
from llm.config import settings
from agent.memory.memory_manager import MemoryManager

logger = logging.getLogger(__name__)

class SpeechRecognitionTool(BaseTool):
    """Tool for converting speech to text.\n\n    NOTE: Audio transcription is coming in a future update.\n    Currently, this feature is not available."""
    
    name: str = "speech_recognition"
    description: str = "Convert speech audio to text using advanced speech recognition (Coming Soon)"
    
    def _run(self, audio_data: Union[str, bytes]) -> str:
        """Convert speech to text.\n\n        NOTE: Audio transcription is coming in a future update.\n        Args:\n            audio_data: Base64 encoded audio or audio bytes\n        Returns:\n            str: Transcribed text or feature notice\n        """
        # --- FEATURE COMING SOON ---
        return "Audio transcription is coming in a future update."
    
    def _transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio using Whisper or similar service.\n        NOTE: Audio transcription is coming in a future update."""
        # --- FEATURE COMING SOON ---
        return "Audio transcription is coming in a future update."

class TextToSpeechTool(BaseTool):
    """Tool for converting text to speech."""
    
    name: str = "text_to_speech"
    description: str = "Convert text to speech using advanced TTS"
    
    def _run(self, text: str, voice: str = "alloy", format: str = "mp3") -> str:
        """Convert text to speech. Placeholder only, as OpenAI TTS is not available."""
        return "TTS is not available. Please configure a supported provider."

class VoiceAgent(BaseAgent):
    """Agent specialized in voice interactions and audio processing.
    Modular, stateless, and uses injected memory manager for context/state.
    """
    def __init__(self, config: Union[AgentConfig, Dict[str, Any]], memory_manager: MemoryManager = None):
        """Initialize the voice agent."""
        super().__init__(config)
        self.memory_manager = memory_manager
        logger.info(f"VoiceAgent initialized with memory_manager: {bool(memory_manager)}")
        # Add voice-specific tools
        self.speech_recognition = SpeechRecognitionTool()
        self.text_to_speech = TextToSpeechTool()
        # Voice-specific configuration
        self.default_voice = "alloy"
        self.supported_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        self.supported_formats = ["mp3", "opus", "aac", "flac"]
        logger.info(f"Voice Agent initialized with voice: {self.default_voice}")

    async def process(self, input_text: str, context: Dict[str, Any] = None, conversation_id: str = None, user_id: str = None, conversation_memory=None, **kwargs) -> Dict[str, Any]:
        logger.info(f"VoiceAgent.process called | input_text: {input_text} | conversation_id: {conversation_id} | user_id: {user_id}")
        try:
            chat_history = kwargs.get('chat_history', '')
            topic = None
            intent = None
            last_user_message = None
            if conversation_memory:
                topic = conversation_memory.topics[-1] if conversation_memory.topics else None
                intent = conversation_memory.intents[-1] if conversation_memory.intents else None
                for msg in reversed(conversation_memory.messages):
                    if msg['role'] == 'user':
                        last_user_message = msg['content']
                        break
            # Compose a context-aware prompt
            context_snippets = "\n".join([
                f"{m['role'].capitalize()}: {m['content']}" for m in conversation_memory.messages[-6:]
            ]) if conversation_memory else ""
            system_prompt = (
                "You are a voice assistant. Use the conversation context, topic, and intent to answer the user's question as accurately and helpfully as possible. "
                "If the user is following up, use the previous context to disambiguate."
            )
            from langchain_groq import ChatGroq
            from langchain_core.messages import SystemMessage, HumanMessage
            from llm.config import settings
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Conversation context:\n{context_snippets}\n\nTopic: {topic}\nIntent: {intent}\n\nUser question: {input_text}")
            ]
            llm = ChatGroq(
                temperature=0.2,
                model_name=settings.REASONING_MODEL,
                groq_api_key=os.getenv('GROQ_API_KEY')
            )
            result = await llm.ainvoke(messages)
            return {"output": result.content.strip(), "metadata": {"success": True, "topic": topic, "intent": intent}}
        except Exception as e:
            logger.error(f"Async error in voice process: {str(e)}", exc_info=True)
            return {
                'output': f"I encountered an error while processing your voice request: {str(e)}",
                'metadata': {
                    'error': str(e),
                    'success': False
                }
            }
    
    async def process_voice_input(
        self, 
        audio_data: Union[str, bytes],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Process voice input and return text response with optional audio.\n        NOTE: Audio transcription is coming in a future update."""
        # --- FEATURE COMING SOON ---
        return {
            "success": False,
            "error": "Audio transcription is coming in a future update.",
            "output": "Voice input is not yet supported. Please check back soon."
        }
    
    async def text_to_speech_response(
        self, 
        text: str, 
        voice: str = None,
        format: str = "mp3"
    ) -> Dict[str, Any]:
        """Convert text response to speech."""
        
        try:
            voice = voice or self.default_voice
            audio_data = self.text_to_speech._run(text, voice, format)
            
            return {
                "success": True,
                "audio_data": audio_data,
                "voice": voice,
                "format": format,
                "text_length": len(text)
            }
            
        except Exception as e:
            logger.error(f"Error in TTS: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_supported_voices(self) -> List[str]:
        """Get list of supported voices."""
        return self.supported_voices
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported audio formats."""
        return self.supported_formats

    def _create_agent(self):
        """VoiceAgent does not use a LangChain agent, so return None."""
        return None 

    async def aprocess(self, input_text: str, **kwargs) -> dict:
        """Override aprocess to avoid calling self.agent.invoke, which does not exist for VoiceAgent."""
        return {
            "output": "VoiceAgent does not support generic text processing via aprocess. Use process_voice_input or text_to_speech_response instead.",
            "success": False,
            "error": "NotImplementedError: VoiceAgent does not use a LangChain agent."
        } 