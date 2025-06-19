"""Minimal test for agent functionality without LangChain message classes."""
import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI, status
from pydantic import BaseModel, ConfigDict, Field
from typing import List, Dict, Any, Optional, Union, Literal

# Create a minimal FastAPI app for testing
app = FastAPI()

# Simple message model
class SimpleMessage(BaseModel):
    """Simple message model for testing."""
    content: str
    type: str = "human"
    
    model_config = ConfigDict(
        extra="allow",
        json_encoders={object: str},
        use_enum_values=True,
    )

# Simple agent response model
class AgentResponse(BaseModel):
    """Response model for the agent."""
    content: str
    status: str = "success"

from fastapi import Query

# Test endpoint
@app.get("/api/chat")
async def chat_endpoint(
    message: str = Query(..., description="The message to process"),
    conversation_id: Optional[str] = Query(None, description="The conversation ID"),
    user_id: Optional[str] = Query(None, description="The user ID"),
    **kwargs
) -> Dict[str, str]:
    """Simple chat endpoint that echoes back the message."""
    return {
        "content": f"You said: {message}",
        "status": "success"
    }

class TestMinimalAgent(unittest.TestCase):
    """Minimal test cases for agent functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test client and test data."""
        cls.client = TestClient(app)
        cls.base_url = "/api/chat"
        
        # Test conversation ID
        cls.test_conversation_id = "test_conv_123"
        cls.test_user_id = "test_user_123"
    
    def send_chat_message(self, message: str, **kwargs) -> Dict[str, Any]:
        """Helper method to send a chat message and return the response."""
        # Prepare the query parameters
        params = {
            "message": message,
            "kwargs": "{}",  # Required by the endpoint
            "conversation_id": kwargs.pop("conversation_id", self.test_conversation_id),
            "user_id": kwargs.pop("user_id", self.test_user_id),
            **{k: str(v) for k, v in kwargs.items()}  # Convert all values to strings
        }
        
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        
        print(f"\nSending request to {self.base_url} with params: {params}")
        response = self.client.get(
            self.base_url,
            params=params,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Response status code: {response.status_code}")
        print(f"Response content: {response.content}")
        
        try:
            json_response = response.json()
            print(f"Response JSON: {json_response}")
            return json_response
        except Exception as e:
            print(f"Error parsing JSON response: {e}")
            return {"content": str(response.content), "status": "error"}
    
    def test_basic_chat(self):
        """Test basic chat functionality."""
        response = self.send_chat_message("Hello, how are you?")
        
        self.assertEqual(response.get("status"), "success")
        self.assertIn("You said: Hello, how are you?", response.get("content", ""))
    
    def test_message_processing(self):
        """Test message processing with different message formats."""
        # Test with a simple string
        response = self.send_chat_message("Simple message")
        self.assertEqual(response.get("status"), "success")
        
        # Test with a message that includes special characters
        response = self.send_chat_message("Message with special chars: !@#$%^&*()")
        self.assertEqual(response.get("status"), "success")
        
        # Test with a long message
        long_message = "a" * 1000
        response = self.send_chat_message(long_message)
        self.assertEqual(response.get("status"), "success")
        self.assertIn(long_message, response.get("content", ""))

if __name__ == "__main__":
    unittest.main()
