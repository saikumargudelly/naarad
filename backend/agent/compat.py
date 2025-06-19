"""Compatibility layer for LangChain message types with Pydantic v2."""
from typing import Any, Dict, List, Optional, Union, Type, TypeVar
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict

# Re-export commonly used types from langchain_core.messages
class MessageType(str, Enum):
    HUMAN = "human"
    AI = "ai"
    SYSTEM = "system"
    CHAT = "chat"
    FUNCTION = "function"
    TOOL = "tool"

class BaseMessage(BaseModel):
    """Base message class with discriminator field."""
    model_config = ConfigDict(
        extra='allow',
        use_enum_values=True,
        json_encoders={
            Enum: lambda v: v.value
        },
        validate_assignment=True,
        populate_by_name=True
    )
    
    content: str
    type: MessageType
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments"
    )

class HumanMessage(BaseMessage):
    """A message from a human."""
    type: MessageType = MessageType.HUMAN

class AIMessage(BaseMessage):
    """A message from an AI."""
    type: MessageType = MessageType.AI

class SystemMessage(BaseMessage):
    """A system message."""
    type: MessageType = MessageType.SYSTEM

class ChatMessage(BaseMessage):
    """A chat message with a role."""
    role: str
    type: MessageType = MessageType.CHAT

class FunctionMessage(BaseMessage):
    """A message for passing the result of executing a function back to the model."""
    name: str
    type: MessageType = MessageType.FUNCTION

class ToolMessage(BaseMessage):
    """A message for passing the result of executing a tool back to the model."""
    tool_call_id: str
    type: MessageType = MessageType.TOOL

# Union type for all message types
AnyMessage = Union[HumanMessage, AIMessage, SystemMessage, ChatMessage, FunctionMessage, ToolMessage]

def convert_to_langchain_message(message: Dict[str, Any]) -> AnyMessage:
    """Convert a message dict to the appropriate message type."""
    msg_type = message.get('type')
    content = message.get('content', '')
    
    if msg_type == MessageType.HUMAN or 'role' in message and message['role'] == 'user':
        return HumanMessage(content=content, **{k: v for k, v in message.items() 
                                              if k not in ('type', 'content', 'role')})
    elif msg_type == MessageType.AI or 'role' in message and message['role'] in ('assistant', 'ai'):
        return AIMessage(content=content, **{k: v for k, v in message.items() 
                                           if k not in ('type', 'content', 'role')})
    elif msg_type == MessageType.SYSTEM or 'role' in message and message['role'] == 'system':
        return SystemMessage(content=content, **{k: v for k, v in message.items() 
                                               if k not in ('type', 'content', 'role')})
    elif msg_type == MessageType.CHAT:
        return ChatMessage(role=message.get('role', 'user'), 
                          content=content,
                          **{k: v for k, v in message.items() 
                             if k not in ('type', 'content', 'role')})
    elif msg_type == MessageType.FUNCTION:
        return FunctionMessage(name=message.get('name', ''), 
                             content=content,
                             **{k: v for k, v in message.items() 
                                if k not in ('type', 'content', 'name')})
    elif msg_type == MessageType.TOOL:
        return ToolMessage(tool_call_id=message.get('tool_call_id', ''),
                          content=content,
                          **{k: v for k, v in message.items()
                             if k not in ('type', 'content', 'tool_call_id')})
    else:
        # Default to HumanMessage if type is not recognized
        return HumanMessage(content=content, **{k: v for k, v in message.items() 
                                              if k not in ('type', 'content')})

def convert_from_langchain_message(message: AnyMessage) -> Dict[str, Any]:
    """Convert a message object to a dict."""
    if hasattr(message, 'dict'):
        return message.dict()
    return dict(message)
