"""Test LangChain message types with Pydantic v2."""
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from typing import List, Union, Literal

# Define a simple model that uses LangChain messages
class Conversation(BaseModel):
    messages: List[Union[AIMessage, HumanMessage, SystemMessage]] = Field(default_factory=list)

# Test creating messages
def test_create_messages():
    """Test creating LangChain messages."""
    human_msg = HumanMessage(content="Hello")
    ai_msg = AIMessage(content="Hi there")
    system_msg = SystemMessage(content="System message")
    
    assert human_msg.content == "Hello"
    assert ai_msg.content == "Hi there"
    assert system_msg.content == "System message"

# Test using messages in a Pydantic model
def test_messages_in_model():
    """Test using LangChain messages in a Pydantic model."""
    conversation = Conversation(
        messages=[
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there"),
            SystemMessage(content="System message")
        ]
    )
    
    assert len(conversation.messages) == 3
    assert isinstance(conversation.messages[0], HumanMessage)
    assert isinstance(conversation.messages[1], AIMessage)
    assert isinstance(conversation.messages[2], SystemMessage)

# Test serialization
def test_message_serialization():
    """Test serializing and deserializing messages."""
    human_msg = HumanMessage(content="Hello", additional_kwargs={"key": "value"})
    msg_dict = human_msg.dict()
    
    assert msg_dict["content"] == "Hello"
    assert msg_dict["additional_kwargs"]["key"] == "value"
    
    # Deserialize
    new_msg = HumanMessage(**msg_dict)
    assert new_msg.content == "Hello"
    assert new_msg.additional_kwargs["key"] == "value"
