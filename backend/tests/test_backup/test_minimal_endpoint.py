"""Minimal FastAPI endpoint to test message handling."""
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field
from typing import List, Union, Literal, Dict, Any

# Create a minimal FastAPI app
app = FastAPI()

# Minimal message classes
class BaseMessage(BaseModel):
    content: str
    type: str
    additional_kwargs: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        extra = "allow"
        json_encoders = {object: str}

class HumanMessage(BaseMessage):
    type: Literal["human"] = "human"

class AIMessage(BaseMessage):
    type: Literal["ai"] = "ai"

# Union type for messages
AnyMessage = Union[HumanMessage, AIMessage]

# Simple endpoint that echoes back the message
@app.post("/echo")
async def echo_message(message: AnyMessage):
    """Echo back the received message."""
    return {
        "received_type": type(message).__name__,
        "content": message.content,
        "type": message.type,
        "additional_kwargs": message.additional_kwargs
    }

# Test client
client = TestClient(app)

def test_echo_human_message():
    """Test sending a human message to the echo endpoint."""
    response = client.post(
        "/echo",
        json={
            "content": "Hello, world!",
            "type": "human",
            "additional_kwargs": {"key": "value"}
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["content"] == "Hello, world!"
    assert data["type"] == "human"
    assert data["additional_kwargs"]["key"] == "value"

def test_echo_ai_message():
    """Test sending an AI message to the echo endpoint."""
    response = client.post(
        "/echo",
        json={
            "content": "Hi there!",
            "type": "ai",
            "additional_kwargs": {"key": "value"}
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["content"] == "Hi there!"
    assert data["type"] == "ai"
    assert data["additional_kwargs"]["key"] == "value"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
