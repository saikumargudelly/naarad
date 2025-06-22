"""Custom message classes that are compatible with Pydantic v2 and LangChain."""
from typing import Any, Dict, List, Optional, Union, Literal, TypeVar, Generic, Type
from pydantic import BaseModel, Field, ConfigDict

# Type variable for message content
T = TypeVar('T')

class BaseMessage(BaseModel):
    """Base message class that's compatible with Pydantic v2 and LangChain."""
    content: str
    type: str
    additional_kwargs: Dict[str, Any] = Field(default_factory=dict)
    
    # Pydantic v2 config
    model_config = ConfigDict(
        extra="allow",
        json_encoders={
            object: lambda v: str(v)  # Convert any non-serializable objects to strings
        },
        use_enum_values=True,
    )
    
    def __init__(self, content: str, **kwargs):
        # Ensure type is set by the subclass, but don't access self.type during init
        if 'type' not in kwargs:
            # Get the type from the class, not the instance
            kwargs['type'] = getattr(self.__class__, 'type', 'unknown')
        super().__init__(content=content, **kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the message to a dictionary."""
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseMessage':
        """Create a message from a dictionary."""
        return cls(**data)

class HumanMessage(BaseMessage):
    """A message from a human."""
    type: Literal["human"] = "human"

class AIMessage(BaseMessage):
    """A message from an AI."""
    type: Literal["ai"] = "ai"

class SystemMessage(BaseMessage):
    """A system message."""
    type: Literal["system"] = "system"

class ChatMessage(BaseMessage):
    """A chat message with a role."""
    role: str
    type: Literal["chat"] = "chat"

# Union type for all message types
AnyMessage = Union[HumanMessage, AIMessage, SystemMessage, ChatMessage]

def convert_to_message(data: Union[str, Dict[str, Any], BaseMessage]) -> AnyMessage:
    """Convert various input types to a message object."""
    if isinstance(data, BaseMessage):
        return data
    
    if isinstance(data, str):
        return HumanMessage(content=data)
    
    if isinstance(data, dict):
        msg_type = data.get('type', 'human')
        content = data.get('content', '')
        
        if msg_type == 'human' or ('role' in data and data['role'] == 'user'):
            return HumanMessage(content=content, **{k: v for k, v in data.items() 
                                                  if k not in ('type', 'content', 'role')})
        elif msg_type == 'ai' or ('role' in data and data['role'] in ('assistant', 'ai')):
            return AIMessage(content=content, **{k: v for k, v in data.items() 
                                               if k not in ('type', 'content', 'role')})
        elif msg_type == 'system' or ('role' in data and data['role'] == 'system'):
            return SystemMessage(content=content, **{k: v for k, v in data.items() 
                                                   if k not in ('type', 'content', 'role')})
        elif msg_type == 'chat':
            return ChatMessage(role=data.get('role', 'user'), 
                             content=content,
                             **{k: v for k, v in data.items() 
                                if k not in ('type', 'content', 'role')})
        else:
            # Default to human message
            return HumanMessage(content=content, **{k: v for k, v in data.items() 
                                                  if k not in ('type', 'content')})
    
    raise ValueError(f"Cannot convert {type(data)} to message")

def convert_to_messages(data: Union[List[Union[str, Dict[str, Any], BaseMessage]], 
                         Union[str, Dict[str, Any], BaseMessage]]) -> List[AnyMessage]:
    """Convert various input types to a list of message objects.
    
    Args:
        data: Can be a single message or a list of messages in various formats
        
    Returns:
        List of message objects
    """
    if data is None:
        return []
        
    if isinstance(data, (str, dict, BaseMessage)):
        return [convert_to_message(data)]
        
    if isinstance(data, list):
        return [convert_to_message(item) for item in data]
        
    raise ValueError(f"Cannot convert {type(data)} to messages")
