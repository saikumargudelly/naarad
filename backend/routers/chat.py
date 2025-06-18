from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
from agent.naarad_agent import naarad_agent

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    images: Optional[List[str]] = []
    conversation_id: Optional[str] = None
    chat_history: Optional[List[dict]] = []

@router.post("/chat")
async def chat(chat_request: ChatRequest):
    """
    Process a chat message and return Naarad's response.
    """
    try:
        response = await naarad_agent.process_message(
            message=chat_request.message,
            images=chat_request.images,
            chat_history=chat_request.chat_history
        )
        
        return {
            "message": response,
            "conversation_id": chat_request.conversation_id or "",
            "sources": []  # Will be populated when tools are used
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "naarad-chat"}
