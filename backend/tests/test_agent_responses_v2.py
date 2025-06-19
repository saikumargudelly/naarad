"""Tests for agent responses and functionality.

This version uses a simplified message implementation to avoid Pydantic v2 compatibility issues."""
import os
import json
import unittest
from typing import Dict, List, Any, Optional, Union
from unittest.mock import patch, MagicMock, AsyncMock

from fastapi.testclient import TestClient
from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel, ConfigDict, Field

# Import the main FastAPI app
from main import app

# Import agent types
from agent.domain_agents import (
    DomainAgent,
    ContextAwareChatAgent,
    DomainSpecificAgent,
    DomainSpecificAgentWithTools,
    DomainSpecificAgentWithToolsAndMemory,
)

# Import message types
from agent.message_types import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ChatMessage,
    convert_to_message,
    convert_message_to_dict,
    convert_dict_to_message,
)

# Import test utilities
from tests.test_utils import (
    TEST_CONVERSATION_ID,
    TEST_USER_ID,
    TEST_MESSAGES,
    TEST_TOOL_INPUTS,
    TEST_AGENT_RESPONSES,
)

class TestAgentResponses(unittest.TestCase):
    """Test cases for agent responses and functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test client and test data."""
        cls.client = TestClient(app)
        cls.base_url = "/api/chat"
        
        # Test conversation ID
        cls.test_conversation_id = TEST_CONVERSATION_ID
        cls.test_user_id = TEST_USER_ID
        
        # Test messages
        cls.test_messages = TEST_MESSAGES
        
        # Test tool inputs
        cls.test_tool_inputs = TEST_TOOL_INPUTS
        
        # Test agent responses
        cls.test_agent_responses = TEST_AGENT_RESPONSES
    
    def send_chat_message(self, message: Union[str, Dict[str, Any], BaseMessage], **kwargs) -> Dict[str, Any]:
        """Helper method to send a chat message and return the response."""
        # Convert to our message format and then to string
        message_obj = convert_to_message(message)
        message_content = message_obj.content if hasattr(message_obj, 'content') else str(message_obj)
        
        # Prepare the query parameters
        params = {
            "message": message_content,
            "kwargs": json.dumps(kwargs.pop("kwargs", {})),  # Convert dict to JSON string
            "conversation_id": kwargs.pop("conversation_id", self.test_conversation_id),
            "user_id": kwargs.pop("user_id", self.test_user_id),
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
        
        try:
            response_data = response.json()
            print(f"Response JSON: {response_data}")
            return response_data
        except json.JSONDecodeError:
            return {"content": response.text, "status": "error"}
    
    def test_basic_chat(self):
        """Test basic chat functionality."""
        response = self.send_chat_message("Hello, how are you?")
        
        self.assertEqual(response.get("status"), "success")
        self.assertIn("content", response)
    
    def test_message_dict_input(self):
        """Test sending a message as a dictionary."""
        response = self.send_chat_message(
            {"content": "Hello from dict", "type": "human"}
        )
        
        self.assertEqual(response.get("status"), "success")
        self.assertIn("content", response)
    
    @patch('agent.domain_agents.DomainAgent._process_message')
    def test_message_processing(self, mock_process):
        """Test message processing with a mock response."""
        # Set up mock response
        mock_response = {"content": "Mock response", "status": "success"}
        mock_process.return_value = mock_response
        
        # Send a test message
        response = self.send_chat_message("Test message")
        
        # Verify the response
        self.assertEqual(response.get("status"), "success")
        self.assertEqual(response.get("content"), "Mock response")
        
        # Verify the mock was called
        mock_process.assert_called_once()
    
    def test_invalid_message_format(self):
        """Test sending an invalid message format."""
        # Send an invalid message (not a string or dict)
        response = self.send_chat_message(12345)
        
        # The API should still return a success status
        self.assertEqual(response.get("status"), "success")
    
    @patch('agent.domain_agents.DomainSpecificAgentWithTools._process_message_with_tools')
    def test_tool_usage(self, mock_tool_process):
        """Test tool usage in agent responses."""
        # Set up mock tool response
        tool_response = {
            "content": "Tool was used successfully",
            "tool_calls": [{"name": "test_tool", "args": {"param": "value"}}],
            "status": "success"
        }
        mock_tool_process.return_value = tool_response
        
        # Send a test message that should trigger tool usage
        response = self.send_chat_message(
            "Use the test tool",
            agent_type="tool_using_agent"
        )
        
        # Verify the response
        self.assertEqual(response.get("status"), "success")
        self.assertEqual(response.get("content"), "Tool was used successfully")
        self.assertIn("tool_calls", response)
        
        # Verify the mock was called
        mock_tool_process.assert_called_once()

if __name__ == "__main__":
    unittest.main()
