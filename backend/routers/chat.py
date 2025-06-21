"""Chat router for handling chat-related endpoints."""

from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Optional, Dict, Any, Annotated
import logging
from datetime import datetime, timedelta

# Import compatibility layer
from agent.compat import get_rate_limiter, create_compatible_model_config

from agent.naarad_agent import naarad_agent
from config.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

# Rate limiting setup
try:
    Limiter, get_remote_address = get_rate_limiter()
    limiter = Limiter(key_func=get_remote_address)
    # Default rate limit: 100 requests per minute
    rate_limit = settings.rate_limit
except ImportError:
    logger.warning("Rate limiting not available")
    limiter = None
    rate_limit = "100/minute"

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    model_config = create_compatible_model_config(
        str_strip_whitespace=True,
        str_min_length=1,
        str_max_length=5000,
        extra='forbid',
        validate_assignment=True,
        json_encoders={object: str},
    )
    
    message: str = Field(..., min_length=1, max_length=5000, description="The user's message")
    images: List[str] = Field(
        default_factory=list,
        max_length=5,
        description="List of image URLs or base64 encoded strings"
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="Optional conversation ID for multi-turn conversations"
    )
    chat_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        max_length=100,
        description="Previous messages in the conversation"
    )

    @field_validator('images')
    @classmethod
    def validate_images(cls, v: List[str]) -> List[str]:
        """Validate that no more than 5 images are provided."""
        if len(v) > 5:
            raise ValueError("Maximum of 5 images allowed per request")
        return v

    @field_validator('chat_history')
    @classmethod
    def validate_chat_history(cls, v: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Limit chat history to the last 100 messages."""
        return v[-100:] if v else []

@router.post("/chat")
async def chat(
    request: Request,
    chat_request: ChatRequest
):
    """
    Process a chat message and return Naarad's response.
    Rate limited to 100 requests per minute per IP.
    """
    logger.info(f"Processing chat request: {chat_request.message[:100]}...")
    
    try:
        start_time = datetime.utcnow()
        
        response = await naarad_agent.process_message(
            message=chat_request.message,
            images=chat_request.images,
            chat_history=chat_request.chat_history
        )
        
        process_time = (datetime.utcnow() - start_time).total_seconds()
        logger.info(f"Chat processed in {process_time:.2f}s")
        
        # Extract the actual message from the response structure
        if isinstance(response, dict):
            # The response from naarad_agent contains the message in 'response.output'
            if 'response' in response and isinstance(response['response'], dict) and 'output' in response['response']:
                message = response['response']['output']
            # Fallback for older or direct message formats
            elif 'message' in response:
                message = response['message']
            elif 'text' in response:
                message = response['text']
            elif 'output' in response:
                message = response['output']
            else:
                # If no clear message field, stringify for debugging
                message = str(response)
        else:
            message = str(response)
        
        return {
            "message": message,
            "conversation_id": chat_request.conversation_id or "",
            "sources": [],
            "processing_time": f"{process_time:.2f}s"
        }
    except Exception as e:
        logger.error(f"Error processing chat: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your request"
        )

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "naarad-chat",
        "timestamp": datetime.utcnow().isoformat()
    }
