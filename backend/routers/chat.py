"""Chat router for handling chat-related endpoints."""

from fastapi import APIRouter, HTTPException, Request, Depends, WebSocket
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Optional, Dict, Any, Annotated
import logging
from datetime import datetime, timedelta
import json

# Import compatibility layer
from agent.compat import get_rate_limiter, create_compatible_model_config

from dependencies import get_naarad_agent
from config.config import settings
from agent.factory import create_orchestrator_with_agents

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
    logger.info(f"[CHAT] Incoming REST chat request: {chat_request}")
    try:
        start_time = datetime.utcnow()
        orchestrator = create_orchestrator_with_agents()
        # Optionally, you can add more context here
        context = {}
        user_id = request.headers.get("X-User-Id", "default")
        conversation_id = chat_request.conversation_id or f"conv_{user_id}_{start_time.timestamp()}"
        result = await orchestrator.process_query(
            user_input=chat_request.message,
            context=context,
            conversation_id=conversation_id,
            user_id=user_id
        )
        process_time = (datetime.utcnow() - start_time).total_seconds()
        message = result.get("output", "No response generated.")
        response = {
            "message": message,
            "conversation_id": conversation_id,
            "sources": [],
            "processing_time": f"{process_time:.2f}s"
        }
        logger.info(f"[CHAT] REST chat response: {response}")
        return response
    except Exception as e:
        logger.error(f"[CHAT] REST chat exception: {e}")
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

@router.websocket("/ws/chat/{user_id}/{conversation_id}")
async def websocket_chat(websocket: WebSocket, user_id: str, conversation_id: str):
    logger.info(f"[WEBSOCKET] Connection opened | user_id: {user_id} | conversation_id: {conversation_id}")
    await websocket.accept()
    try:
        orchestrator = create_orchestrator_with_agents()
        while True:
            data = await websocket.receive_text()
            logger.info(f"[WEBSOCKET] Received message: {data} | user_id: {user_id} | conversation_id: {conversation_id}")
            try:
                msg = json.loads(data)
                user_message = msg.get("message", "")
                context = {}
                # Pass timestamp from frontend if present
                if "timestamp" in msg:
                    context["timestamp"] = msg["timestamp"]
                logger.info(f"[WEBSOCKET] Calling orchestrator.process | user_input: {user_message} | context: {context} | conversation_id: {conversation_id} | user_id: {user_id}")
                try:
                    result = await orchestrator.process_query(
                        user_input=user_message,
                        context=context,
                        conversation_id=conversation_id,
                        user_id=user_id
                    )
                    logger.info(f"[WEBSOCKET] Orchestrator result: {result} | user_id: {user_id} | conversation_id: {conversation_id}")
                    response = {
                        "type": "message",
                        "content": result.get("output", "No response generated."),
                        "conversation_id": conversation_id
                    }
                except Exception as orchestrator_exc:
                    logger.error(f"[WEBSOCKET] Orchestrator exception: {orchestrator_exc}", exc_info=True)
                    response = {
                        "type": "error",
                        "error": f"Orchestrator error: {str(orchestrator_exc)}",
                        "conversation_id": conversation_id
                    }
                await websocket.send_text(json.dumps(response))
                logger.info(f"[WEBSOCKET] Sent response: {response} | user_id: {user_id} | conversation_id: {conversation_id}")
            except Exception as e:
                logger.error(f"[WEBSOCKET] Exception during message processing: {e}", exc_info=True)
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "error": str(e),
                    "conversation_id": conversation_id
                }))
    except Exception as e:
        logger.error(f"[WEBSOCKET] Exception: {e} | user_id: {user_id} | conversation_id: {conversation_id}", exc_info=True)
    finally:
        logger.info(f"[WEBSOCKET] Connection closed | user_id: {user_id} | conversation_id: {conversation_id}")
