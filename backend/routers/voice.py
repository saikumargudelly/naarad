"""Voice router for handling voice-related endpoints and testing."""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import logging
import base64
import json
import asyncio
from datetime import datetime
import tempfile
import os

from dependencies import get_voice_agent
from config.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

class VoiceProcessRequest(BaseModel):
    """Request model for voice processing."""
    audio_data: str = Field(..., description="Base64 encoded audio data")
    user_id: str = Field(default="default", description="User identifier")
    voice_preference: Optional[str] = Field(default="alloy", description="Preferred voice")
    generate_audio: bool = Field(default=True, description="Generate audio response")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")

class VoiceTestRequest(BaseModel):
    """Request model for voice testing."""
    test_type: str = Field(..., description="Type of test (recognition, synthesis, full)")
    audio_data: Optional[str] = Field(default=None, description="Base64 encoded audio for testing")
    text_data: Optional[str] = Field(default=None, description="Text for synthesis testing")

@router.post("/voice/process")
async def process_voice_input(request: VoiceProcessRequest):
    """
    Process voice input and return transcribed text with optional audio response.
    """
    logger.info(f"Processing voice input for user: {request.user_id}")
    
    try:
        start_time = datetime.utcnow()
        
        # Process voice input
        voice_agent = get_voice_agent()
        result = await voice_agent.process_voice_input(
            audio_data=request.audio_data,
            context={
                "user_id": request.user_id,
                "voice": request.voice_preference,
                "generate_audio": request.generate_audio,
                **request.context
            }
        )
        
        process_time = (datetime.utcnow() - start_time).total_seconds()
        
        if result.get("success", False):
            return {
                "success": True,
                "transcribed_text": result.get("transcribed_text", ""),
                "response_text": result.get("response_text", ""),
                "audio_response": result.get("audio_response"),
                "voice_used": result.get("metadata", {}).get("voice_used", request.voice_preference),
                "processing_time": f"{process_time:.2f}s",
                "user_id": request.user_id
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Voice processing failed: {result.get('error', 'Unknown error')}"
            )
            
    except Exception as e:
        logger.error(f"Error processing voice input: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing voice input: {str(e)}"
        )

@router.post("/voice/upload")
async def upload_audio_file(
    file: UploadFile = File(...),
    user_id: str = Form(default="default"),
    voice_preference: str = Form(default="alloy"),
    generate_audio: bool = Form(default=True)
):
    """
    Upload and process audio file.
    """
    logger.info(f"Processing uploaded audio file: {file.filename}")
    
    try:
        # Validate file type
        if not file.content_type.startswith('audio/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an audio file"
            )
        
        # Read file content
        content = await file.read()
        audio_base64 = base64.b64encode(content).decode('utf-8')
        
        # Process with voice agent
        voice_agent = get_voice_agent()
        result = await voice_agent.process_voice_input(
            audio_data=audio_base64,
            context={
                "user_id": user_id,
                "voice": voice_preference,
                "generate_audio": generate_audio,
                "filename": file.filename
            }
        )
        
        if result.get("success", False):
            return {
                "success": True,
                "transcribed_text": result.get("transcribed_text", ""),
                "response_text": result.get("response_text", ""),
                "audio_response": result.get("audio_response"),
                "filename": file.filename,
                "user_id": user_id
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Voice processing failed: {result.get('error', 'Unknown error')}"
            )
            
    except Exception as e:
        logger.error(f"Error processing uploaded audio: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing uploaded audio: {str(e)}"
        )

@router.post("/voice/synthesize")
async def synthesize_speech(
    text: str = Form(...),
    voice: str = Form(default="alloy"),
    format: str = Form(default="mp3")
):
    """
    Convert text to speech.
    """
    logger.info(f"Synthesizing speech for text: {text[:50]}...")
    
    try:
        voice_agent = get_voice_agent()
        result = await voice_agent.text_to_speech_response(
            text=text,
            voice=voice,
            format=format
        )
        
        if result.get("success", False):
            return {
                "success": True,
                "audio_data": result.get("audio_data"),
                "voice": result.get("voice"),
                "format": result.get("format"),
                "text_length": result.get("text_length")
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Speech synthesis failed: {result.get('error', 'Unknown error')}"
            )
            
    except Exception as e:
        logger.error(f"Error synthesizing speech: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error synthesizing speech: {str(e)}"
        )

@router.post("/voice/test")
async def test_voice_features(request: VoiceTestRequest):
    """
    Test voice features with different scenarios.
    """
    logger.info(f"Running voice test: {request.test_type}")
    
    try:
        test_results = {
            "test_type": request.test_type,
            "timestamp": datetime.utcnow().isoformat(),
            "results": {}
        }
        
        if request.test_type == "recognition" and request.audio_data:
            # Test speech recognition
            voice_agent = get_voice_agent()
            transcribed = voice_agent.speech_recognition._run(request.audio_data)
            test_results["results"]["recognition"] = {
                "success": not transcribed.startswith("Error"),
                "transcribed_text": transcribed,
                "audio_length": len(request.audio_data)
            }
        
        elif request.test_type == "synthesis" and request.text_data:
            # Test text-to-speech
            voice_agent = get_voice_agent()
            result = await voice_agent.text_to_speech_response(
                text=request.text_data,
                voice="alloy"
            )
            test_results["results"]["synthesis"] = {
                "success": result.get("success", False),
                "audio_generated": bool(result.get("audio_data")),
                "voice_used": result.get("voice"),
                "text_length": len(request.text_data)
            }
        
        elif request.test_type == "full" and request.audio_data:
            # Test full voice processing pipeline
            voice_agent = get_voice_agent()
            result = await voice_agent.process_voice_input(
                audio_data=request.audio_data,
                context={"generate_audio": True}
            )
            test_results["results"]["full_pipeline"] = {
                "success": result.get("success", False),
                "transcribed_text": result.get("transcribed_text"),
                "response_text": result.get("response_text"),
                "audio_response": bool(result.get("audio_response"))
            }
        
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid test type or missing required data"
            )
        
        return test_results
        
    except Exception as e:
        logger.error(f"Error in voice testing: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error in voice testing: {str(e)}"
        )

@router.get("/voice/voices")
async def get_available_voices():
    """Get list of available voices."""
    voice_agent = get_voice_agent()
    return {
        "voices": voice_agent.get_supported_voices(),
        "formats": voice_agent.get_supported_formats(),
        "default_voice": voice_agent.default_voice
    }

@router.get("/voice/health")
async def voice_health_check():
    """Health check for voice services."""
    try:
        # Test basic functionality
        test_text = "Hello, this is a test."
        voice_agent = get_voice_agent()
        result = await voice_agent.text_to_speech_response(text=test_text)
        
        return {
            "status": "healthy",
            "service": "voice-processing",
            "timestamp": datetime.utcnow().isoformat(),
            "test_result": result.get("success", False),
            "supported_voices": len(voice_agent.get_supported_voices()),
            "supported_formats": len(voice_agent.get_supported_formats())
        }
    except Exception as e:
        logger.error(f"Voice health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "service": "voice-processing",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        } 