"""Minimal test case to isolate the Pydantic v2 discriminator issue."""
import pytest
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal, Union, List

# Define a simple message hierarchy
class BaseMessage(BaseModel):
    """Base message class."""
    content: str
    type: str
    
    model_config = ConfigDict(
        extra="allow",
        json_encoders={object: str},
        use_enum_values=True,
    )

class HumanMessage(BaseMessage):
    """A message from a human."""
    type: Literal["human"] = "human"

class AIMessage(BaseMessage):
    """A message from an AI."""
    type: Literal["ai"] = "ai"

# Define a container that uses a union of message types
class MessageContainer(BaseModel):
    """Container that holds a message."""
    message: Union[HumanMessage, AIMessage]

# Test cases
def test_message_creation():
    """Test creating individual message types."""
    human_msg = HumanMessage(content="Hello")
    assert human_msg.content == "Hello"
    assert human_msg.type == "human"
    
    ai_msg = AIMessage(content="Hi there")
    assert ai_msg.content == "Hi there"
    assert ai_msg.type == "ai"

def test_message_container():
    """Test using messages in a container."""
    human_msg = HumanMessage(content="Hello")
    container = MessageContainer(message=human_msg)
    assert container.message.content == "Hello"
    assert container.message.type == "human"
    
    ai_msg = AIMessage(content="Hi there")
    container = MessageContainer(message=ai_msg)
    assert container.message.content == "Hi there"
    assert container.message.type == "ai"

def test_message_container_from_dict():
    """Test creating a container from a dictionary."""
    # This is where we expect issues with discriminators
    human_data = {"content": "Hello", "type": "human"}
    container = MessageContainer(message=human_data)
    assert container.message.content == "Hello"
    assert container.message.type == "human"
    
    ai_data = {"content": "Hi there", "type": "ai"}
    container = MessageContainer(message=ai_data)
    assert container.message.content == "Hi there"
    assert container.message.type == "ai"

if __name__ == "__main__":
    test_message_creation()
    test_message_container()
    test_message_container_from_dict()
    print("All tests passed!")
