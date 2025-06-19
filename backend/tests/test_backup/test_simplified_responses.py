"""Simplified test for agent responses with minimal message implementation."""
import unittest
from fastapi.testclient import TestClient
from fastapi import FastAPI, status
from pydantic import BaseModel, ConfigDict
from typing import List, Dict, Any, Optional, Union, Literal

# Define our minimal message implementation
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

class SystemMessage(BaseMessage):
    """A system message."""
    type: Literal["system"] = "system"

# Union type for all message types
AnyMessage = Union[HumanMessage, AIMessage, SystemMessage]

def convert_to_message(data: Union[str, Dict[str, Any], BaseMessage]) -> BaseMessage:
    """Convert various input types to a message object."""
    if isinstance(data, BaseMessage):
        return data
    
    if isinstance(data, str):
        return HumanMessage(content=data)
    
    if isinstance(data, dict):
        msg_type = data.get('type', 'human')
        content = data.get('content', '')
        
        if msg_type == 'human' or ('role' in data and data.get('role') == 'user'):
            return HumanMessage(content=content, **{k: v for k, v in data.items() 
                                                  if k not in ('type', 'content', 'role')})
        elif msg_type == 'ai' or ('role' in data and data.get('role') in ('assistant', 'ai')):
            return AIMessage(content=content, **{k: v for k, v in data.items() 
                                               if k not in ('type', 'content', 'role')})
        elif msg_type == 'system' or ('role' in data and data.get('role') == 'system'):
            return SystemMessage(content=content, **{k: v for k, v in data.items() 
                                                   if k not in ('type', 'content', 'role')})
        else:
            # Default to human message
            return HumanMessage(content=content, **{k: v for k, v in data.items() 
                                                  if k not in ('type', 'content')})
    
    raise ValueError(f"Cannot convert {type(data)} to message")

# Create a minimal FastAPI app for testing
app = FastAPI()

# Simple agent response model
class AgentResponse(BaseModel):
    """Response model for the agent."""
    content: str
    status: str = "success"

# Test endpoint
@app.post("/api/chat")
async def chat_endpoint(
    message: str,
    conversation_id: Optional[str] = None,
    user_id: Optional[str] = None,
    **kwargs
) -> Dict[str, str]:
    """Simple chat endpoint that echoes back the message."""
    return {
        "content": f"You said: {message}",
        "status": "success"
    }

class TestAgentResponses(unittest.TestCase):
    """Test cases for agent responses and functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test client and test data."""
        cls.client = TestClient(app)
        cls.base_url = "/api/chat"
        
        # Test conversation ID
        cls.test_conversation_id = "test_conv_123"
        cls.test_user_id = "test_user_123"
    
    def send_chat_message(self, message: Union[str, Dict[str, Any], BaseMessage], **kwargs) -> Dict[str, Any]:
        """Helper method to send a chat message and return the response."""
        # Convert to our message format and then to string
        message_obj = convert_to_message(message)
        message_content = message_obj.content if hasattr(message_obj, 'content') else str(message_obj)
        
        # Prepare the query parameters
        params = {
            "message": message_content,
            "kwargs": "{}",  # Required by the endpoint
            **{"conversation_id": kwargs.pop("conversation_id", None)},
            **{"user_id": kwargs.pop("user_id", None)},
            **{k: str(v) for k, v in kwargs.items()}  # Convert all values to strings
        }
        
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        
        print(f"Sending params: {params}")
        response = self.client.post(
            self.base_url,
            params=params,
            headers={"Content-Type": "application/json"}
        )
        print(f"Response status: {response.status_code}")
        print(f"Response content: {response.content}")
        response_data = response.json()
        print(f"Response JSON: {response_data}")
        return response_data
    
    def test_basic_chat(self):
        """Test basic chat functionality."""
        response = self.send_chat_message(
            "Hello, how are you?",
            conversation_id=self.test_conversation_id,
            user_id=self.test_user_id
        )
        
        self.assertEqual(response["status"], "success")
        self.assertIn("You said: Hello, how are you?", response["content"])
    
    def test_message_dict_input(self):
        """Test sending a message as a dictionary."""
        response = self.send_chat_message(
            {"content": "Hello from dict", "type": "human"},
            conversation_id=self.test_conversation_id,
            user_id=self.test_user_id
        )
        
        self.assertEqual(response["status"], "success")
        self.assertIn("You said: Hello from dict", response["content"])

if __name__ == "__main__":
    unittest.main()
