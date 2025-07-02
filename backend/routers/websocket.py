"""WebSocket router for real-time streaming chat functionality."""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict, List, Optional, Any
import json
import logging
import asyncio
import time
from datetime import datetime
import uuid

from dependencies import get_naarad_agent
from agent.monitoring.agent_monitor import agent_monitor
from config.config import settings
from agent.factory import create_orchestrator_with_agents

router = APIRouter()
logger = logging.getLogger(__name__)

# Request debouncing to prevent rate limiting
user_last_request = {}
DEBOUNCE_DELAY = 10.0  # 10 seconds between requests per user to prevent rate limiting

class ConnectionManager:
    """Manages WebSocket connections and broadcasting."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_conversations: Dict[str, str] = {}  # user_id -> conversation_id
    
    async def connect(self, websocket: WebSocket, user_id: str):
        """Connect a new WebSocket client."""
        # If user already has a connection, close the old one
        if user_id in self.active_connections:
            try:
                old_ws = self.active_connections[user_id]
                await old_ws.close(code=1000, reason="New connection from same user")
                logger.info(f"Closed existing connection for user {user_id}")
            except Exception as e:
                logger.warning(f"Error closing old connection for user {user_id}: {e}")
        
        await websocket.accept()
        self.active_connections[user_id] = websocket
        logger.info(f"WebSocket connected for user: {user_id}")
    
    def disconnect(self, user_id: str):
        """Disconnect a WebSocket client."""
        if user_id in self.active_connections:
            del self.active_connections[user_id]
        if user_id in self.user_conversations:
            del self.user_conversations[user_id]
        if user_id in user_last_request:
            del user_last_request[user_id]
        logger.info(f"WebSocket disconnected for user: {user_id}")
    
    async def send_personal_message(self, message: Dict[str, Any], user_id: str):
        """Send a message to a specific user."""
        if user_id in self.active_connections:
            try:
                await self.active_connections[user_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to {user_id}: {e}")
                self.disconnect(user_id)
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast a message to all connected clients."""
        disconnected_users = []
        for user_id, connection in self.active_connections.items():
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error broadcasting to {user_id}: {e}")
                disconnected_users.append(user_id)
        
        # Clean up disconnected users
        for user_id in disconnected_users:
            self.disconnect(user_id)

manager = ConnectionManager()

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time chat without user_id parameter."""
    origin = websocket.headers.get("origin")
    allowed_origins = settings.cors_origins
    
    logger.info(f"Allowed origins: {allowed_origins}")
    logger.info(f"Incoming WebSocket origin: {origin}")
    
    # More permissive CORS for development
    if origin and allowed_origins != ["*"]:
        # Check if origin is in allowed origins or if it's a localhost origin
        is_allowed = origin in allowed_origins
        is_localhost = any(localhost in origin for localhost in ["localhost", "127.0.0.1"])
        
        if not is_allowed and not is_localhost:
            await websocket.close(code=403)
            logger.warning(f"WebSocket connection from disallowed origin: {origin}")
            return
    
    logger.info(f"WebSocket connection accepted from origin: {origin}")
    
    # Generate a user_id for this connection
    user_id = f"user_{uuid.uuid4().hex[:8]}"
    await manager.connect(websocket, user_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Extract message details
            message = message_data.get("message", "")
            conversation_id = message_data.get("conversation_id")
            message_type = message_data.get("type", "text")
            
            # Send typing indicator
            await manager.send_personal_message({
                "type": "typing_start",
                "timestamp": datetime.utcnow().isoformat()
            }, user_id)
            
            try:
                # Process message with streaming response
                await process_streaming_message(
                    user_id=user_id,
                    message=message,
                    conversation_id=conversation_id,
                    message_type=message_type
                )
                
            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)
                await manager.send_personal_message({
                    "type": "error",
                    "error": f"An error occurred while processing your message: {str(e)}",
                    "timestamp": datetime.utcnow().isoformat()
                }, user_id)
                
    except WebSocketDisconnect:
        manager.disconnect(user_id)

@router.websocket("/ws/{user_id}")
async def websocket_endpoint_with_user(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time chat with user_id parameter."""
    origin = websocket.headers.get("origin")
    allowed_origins = settings.cors_origins
    
    logger.info(f"Allowed origins: {allowed_origins}")
    logger.info(f"Incoming WebSocket origin: {origin}")
    
    # More permissive CORS for development
    if origin and allowed_origins != ["*"]:
        # Check if origin is in allowed origins or if it's a localhost origin
        is_allowed = origin in allowed_origins
        is_localhost = any(localhost in origin for localhost in ["localhost", "127.0.0.1"])
        
        if not is_allowed and not is_localhost:
            await websocket.close(code=403)
            logger.warning(f"WebSocket connection from disallowed origin: {origin}")
            return
    
    logger.info(f"WebSocket connection accepted from origin: {origin}")
    await manager.connect(websocket, user_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Extract message details
            message = message_data.get("message", "")
            conversation_id = message_data.get("conversation_id")
            message_type = message_data.get("type", "text")
            
            # Send typing indicator
            await manager.send_personal_message({
                "type": "typing_start",
                "timestamp": datetime.utcnow().isoformat()
            }, user_id)
            
            try:
                # Process message with streaming response
                await process_streaming_message(
                    user_id=user_id,
                    message=message,
                    conversation_id=conversation_id,
                    message_type=message_type
                )
                
            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)
                await manager.send_personal_message({
                    "type": "error",
                    "error": f"An error occurred while processing your message: {str(e)}",
                    "timestamp": datetime.utcnow().isoformat()
                }, user_id)
                
    except WebSocketDisconnect:
        manager.disconnect(user_id)

async def process_streaming_message(
    user_id: str,
    message: str,
    conversation_id: Optional[str] = None,
    message_type: str = "text"
):
    """Process a message with streaming response using the new orchestrator."""
    # Check debouncing - prevent multiple simultaneous requests
    current_time = time.time()
    if user_id in user_last_request:
        time_since_last = current_time - user_last_request[user_id]
        if time_since_last < DEBOUNCE_DELAY:
            logger.info(f"Debouncing request for user {user_id}, last request was {time_since_last:.2f}s ago")
            await manager.send_personal_message({
                "type": "error",
                "error": "Please wait a moment before sending another message.",
                "timestamp": datetime.utcnow().isoformat()
            }, user_id)
            return
    # Update last request time
    user_last_request[user_id] = current_time
    # Generate conversation ID if not provided
    if not conversation_id:
        conversation_id = f"conv_{uuid.uuid4().hex[:8]}"
        manager.user_conversations[user_id] = conversation_id
    # Send conversation ID to client
    await manager.send_personal_message({
        "type": "conversation_id",
        "conversation_id": conversation_id,
        "timestamp": datetime.utcnow().isoformat()
    }, user_id)
    # Use orchestrator to process the message
    orchestrator = create_orchestrator_with_agents()
    try:
        # Optionally, you can add more context here
        context = {}
        result = await orchestrator.process_query(
            user_input=message,
            context=context,
            conversation_id=conversation_id,
            user_id=user_id
        )
        # Send the result back to the user
        await manager.send_personal_message({
            "type": "message",
            "content": result.get("output", "No response generated."),
            "agent": result.get("agent_used", "orchestrator"),
            "metadata": result.get("metadata", {}),
            "timestamp": datetime.utcnow().isoformat()
        }, user_id)
        await manager.send_personal_message({
            "type": "message_complete",
            "conversation_id": conversation_id,
            "timestamp": datetime.utcnow().isoformat()
        }, user_id)
    except Exception as e:
        logger.error(f"Error in orchestrator processing: {e}", exc_info=True)
        await manager.send_personal_message({
            "type": "error",
            "error": f"An error occurred while processing your message: {str(e)}",
            "timestamp": datetime.utcnow().isoformat()
        }, user_id)

@router.get("/ws/connections")
async def get_connection_count():
    """Get the number of active WebSocket connections."""
    return {
        "active_connections": len(manager.active_connections),
        "connected_users": list(manager.active_connections.keys())
    } 