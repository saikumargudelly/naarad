"""Simplified test for agent responses with custom message types."""
import unittest
from fastapi.testclient import TestClient
from fastapi import FastAPI, status
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# Import our custom message implementation
from agent.custom_messages import (
    HumanMessage, 
    AIMessage, 
    SystemMessage, 
    AnyMessage,
    convert_to_message
)

# Create a minimal FastAPI app for testing
app = FastAPI()

# Simple agent response model
class AgentResponse(BaseModel):
    content: str
    status: str = "success"

# Test endpoint
@app.post("/api/chat")
async def chat_endpoint(
    message: str,
    conversation_id: Optional[str] = None,
    user_id: Optional[str] = None,
    **kwargs
) -> AgentResponse:
    """Simple chat endpoint that echoes back the message."""
    return AgentResponse(
        content=f"You said: {message}",
        status="success"
    )

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
    
    def send_chat_message(self, message: Union[str, Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Helper method to send a chat message and return the response."""
        # Convert to our message format and then to dict
        message_obj = convert_to_message(message)
        payload = {"message": message_obj.content, **kwargs}
        
        response = self.client.post(
            self.base_url,
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        return response.json()
    
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
