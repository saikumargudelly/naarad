from typing import Optional, Dict, Any, List, Union, Literal
from pydantic import Field, ConfigDict
from pydantic.dataclasses import dataclass
from enum import Enum
from pydantic import field_validator

class MessageType(str, Enum):
    HUMAN = "human"
    AI = "ai"
    SYSTEM = "system"
    CHAT = "chat"

@dataclass(config=ConfigDict(extra='allow'))
class BaseMessage:
    """Base message class with discriminator field."""
    content: str
    type: MessageType
    additional_kwargs: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = ConfigDict(
        extra='allow',
        use_enum_values=True,
        validate_default=True
    )

@dataclass
class HumanMessage(BaseMessage):
    """A message from a human."""
    type: MessageType = MessageType.HUMAN

@dataclass
class AIMessage(BaseMessage):
    """A message from an AI."""
    type: MessageType = MessageType.AI

@dataclass
class SystemMessage(BaseMessage):
    """A system message."""
    type: MessageType = MessageType.SYSTEM

class ChatMessage(BaseMessage):
    """A chat message with a role."""
    role: str
    content: str
    type: MessageType = MessageType.CHAT
    additional_kwargs: Dict[str, Any] = Field(default_factory=dict)
    
    def __init__(self, **data):
        # Ensure type is always CHAT
        if 'type' not in data:
            data['type'] = MessageType.CHAT
        super().__init__(**data)

# Union type for all message types
AnyMessage = Union[HumanMessage, AIMessage, SystemMessage, ChatMessage]
