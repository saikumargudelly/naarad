from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Optional, Dict, Any, Annotated
import logging
from slowapi import Limiter
from slowapi.util import get_remote_address
from datetime import datetime, timedelta

from agent.naarad_agent import naarad_agent
from config.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

# Rate limiting setup
limiter = Limiter(key_func=get_remote_address)
# Default rate limit: 100 requests per minute
rate_limit = settings.RATE_LIMIT

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    model_config = ConfigDict(
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
@limiter.limit(rate_limit)
async def chat(
    request: Request,  # Required for rate limiting
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
        
        return {
            "message": response,
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
