"""Minimal test file to isolate the Pydantic v2 discriminator issue."""
import pytest
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal, Union

class BaseMessage(BaseModel):
    """Base message class."""
    model_config = ConfigDict(
        extra="allow",
        validate_assignment=True
    )
    
    content: str
    type: str

class HumanMessage(BaseMessage):
    """A message from a human."""
    type: Literal["human"] = "human"

class AIMessage(BaseMessage):
    """A message from an AI."""
    type: Literal["ai"] = "ai"

# Union type for all message types
AnyMessage = Union[HumanMessage, AIMessage]

def test_message_creation():
    """Test creating message objects."""
    human_msg = HumanMessage(content="Hello")
    ai_msg = AIMessage(content="Hi there")
    
    assert human_msg.type == "human"
    assert ai_msg.type == "ai"
    assert human_msg.content == "Hello"
    assert ai_msg.content == "Hi there"

def test_message_union():
    """Test using the message union type."""
    def process_message(msg: AnyMessage) -> str:
        return f"Received {msg.type} message: {msg.content}"
    
    human_msg = HumanMessage(content="Hello")
    ai_msg = AIMessage(content="Hi there")
    
    assert "human" in process_message(human_msg)
    assert "ai" in process_message(ai_msg)
