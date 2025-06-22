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

logger = logging.getLogger(__name__)

class SpeechRecognitionTool(BaseTool):
    """Tool for converting speech to text."""
    
    name: str = "speech_recognition"
    description: str = "Convert speech audio to text using advanced speech recognition"
    
    def _run(self, audio_data: Union[str, bytes]) -> str:
        """Convert speech to text.
        
        Args:
            audio_data: Base64 encoded audio or audio bytes
            
        Returns:
            str: Transcribed text
        """
        try:
            # Decode base64 if needed
            if isinstance(audio_data, str):
                if audio_data.startswith('data:audio'):
                    # Remove data URL prefix
                    audio_data = audio_data.split(',')[1]
                audio_bytes = base64.b64decode(audio_data)
            else:
                audio_bytes = audio_data
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(audio_bytes)
                temp_file_path = temp_file.name
            
            try:
                # Use Whisper or similar for transcription
                # This is a placeholder - you'd integrate with OpenAI Whisper API or similar
                transcribed_text = self._transcribe_audio(temp_file_path)
                return transcribed_text
            finally:
                # Clean up temporary file
                os.unlink(temp_file_path)
                
        except Exception as e:
            logger.error(f"Error in speech recognition: {e}")
            return f"Error transcribing audio: {str(e)}"
    
    def _transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio using Whisper or similar service."""
        # Placeholder implementation
        return "Speech recognition is not available. Please configure a supported provider."

class TextToSpeechTool(BaseTool):
    """Tool for converting text to speech."""
    
    name: str = "text_to_speech"
    description: str = "Convert text to speech using advanced TTS"
    
    def _run(self, text: str, voice: str = "alloy", format: str = "mp3") -> str:
        """Convert text to speech. Placeholder only, as OpenAI TTS is not available."""
        return "TTS is not available. Please configure a supported provider."

class VoiceAgent(BaseAgent):
    """Agent specialized in voice interactions and audio processing."""
    
    def __init__(self, config: Union[AgentConfig, Dict[str, Any]]):
        """Initialize the voice agent."""
        super().__init__(config)
        
        # Add voice-specific tools
        self.speech_recognition = SpeechRecognitionTool()
        self.text_to_speech = TextToSpeechTool()
        
        # Voice-specific configuration
        self.default_voice = "alloy"
        self.supported_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        self.supported_formats = ["mp3", "opus", "aac", "flac"]
        
        logger.info(f"Voice Agent initialized with voice: {self.default_voice}")
    
    async def process_voice_input(
        self, 
        audio_data: Union[str, bytes],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Process voice input and return text response with optional audio."""
        
        try:
            # Step 1: Convert speech to text
            transcribed_text = self.speech_recognition._run(audio_data)
            
            if transcribed_text.startswith("Error"):
                return {
                    "success": False,
                    "error": transcribed_text,
                    "output": "I couldn't understand your voice input. Please try again."
                }
            
            # Step 2: Process the transcribed text with the main agent
            response = await self.process(
                input_text=transcribed_text,
                context=context or {}
            )
            
            # Step 3: Convert response to speech (optional)
            audio_response = None
            if response.get("success", False) and context.get("generate_audio", True):
                audio_response = self.text_to_speech._run(
                    text=response.get("output", ""),
                    voice=context.get("voice", self.default_voice)
                )
            
            return {
                "success": True,
                "transcribed_text": transcribed_text,
                "response_text": response.get("output", ""),
                "audio_response": audio_response,
                "metadata": {
                    "voice_used": context.get("voice", self.default_voice),
                    "processing_time": response.get("metadata", {}).get("processing_time", 0)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in voice processing: {e}")
            return {
                "success": False,
                "error": str(e),
                "output": "I encountered an error processing your voice input."
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