import os
import sys
import json
import unittest
import asyncio
import pytest
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from fastapi.testclient import TestClient
from fastapi import status, FastAPI

# Import our simplified message implementation
from agent.simple_message_v2 import (
    AIMessage, 
    HumanMessage, 
    SystemMessage, 
    BaseMessage,
    convert_to_message
)

# Add parent directory to path to allow imports from the main module
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the main FastAPI app
from main import app
from agent.agents import AgentType

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

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
        
        # Test cases for different agent types
        cls.test_cases = [
            {
                "name": "general_conversation",
                "message": "Hello, how are you?",
                "expected_keywords": ["hello", "hi", "help", "assist"],
                "agent_type": AgentType.RESPONDER,
                "expected_status": 200
            },
            {
                "name": "research_query",
                "message": "What are the latest developments in AI?",
                "expected_keywords": ["research", "AI", "developments", "latest"],
                "agent_type": AgentType.RESEARCHER,
                "expected_status": 200
            },
            {
                "name": "analytical_query",
                "message": "Analyze the impact of remote work on productivity",
                "expected_keywords": ["analysis", "productivity", "remote work", "impact"],
                "agent_type": AgentType.ANALYST,
                "expected_status": 200
            },
            {
                "name": "quality_check_query",
                "message": "Review this response for accuracy and completeness",
                "expected_keywords": ["review", "accuracy", "completeness", "quality"],
                "agent_type": AgentType.QUALITY,
                "expected_status": 200
            },
            {
                "name": "complex_multi_part_query",
                "message": "Research the latest AI developments and analyze their potential business impact",
                "expected_keywords": ["research", "analysis", "AI", "business impact"],
                "agent_type": AgentType.ANALYST,  # Should be handled by analyst
                "expected_status": 200
            }
        ]
    
    def send_chat_message(self, message: Union[str, Dict[str, Any]], 
                         conversation_id: Optional[str] = None, 
                         user_id: Optional[str] = None, 
                         **kwargs) -> Dict[str, Any]:
        """Helper method to send a chat message and return the response.
        
        Args:
            message: The message text to send (can be string or message dict)
            conversation_id: Optional conversation ID
            user_id: Optional user ID
            **kwargs: Additional parameters to include in the request
            
        Returns:
            Dict containing the JSON response
        """
        # Convert to our message format and then to dict
        message_obj = convert_to_message(message)
        payload = {"message": message_obj.model_dump(), **kwargs}
        if conversation_id:
            payload["conversation_id"] = conversation_id
        if user_id:
            payload["user_id"] = user_id
            
        response = self.client.post(
            self.base_url,
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        return response.json()
    
    def test_agent_responses(self):
        """Test that different queries are handled by the appropriate agents."""
        for test_case in self.test_cases:
            with self.subTest(test_case["name"]):
                # Send the test message
                response = self.send_chat_message(
                    message=test_case["message"],
                    conversation_id=self.test_conversation_id,
                    user_id=self.test_user_id
                )
                
                # Verify the response structure
                self.assertIn("message", response, "Response should contain 'message' field")
                self.assertIn("conversation_id", response, "Response should contain 'conversation_id' field")
                self.assertEqual(response["conversation_id"], self.test_conversation_id,
                               f"Expected conversation_id to be {self.test_conversation_id}")
                
                # Check the message response structure
                message = response["message"]
                self.assertIn("success", message, "Message should contain 'success' field")
                self.assertIn("response", message, "Message should contain 'response' field")
                self.assertIn("metadata", message, "Message should contain 'metadata' field")
                self.assertIn("agent_used", message.get("metadata", {}), 
                             "Metadata should contain 'agent_used' field")
                
                # Verify the correct agent was used if specified in test case
                if "agent_type" in test_case:
                    self.assertEqual(
                        message["metadata"]["agent_used"],
                        test_case["agent_type"].value.lower(),
                        f"Expected {test_case['agent_type'].value} to handle the query"
                    )
                
                # Check for expected keywords in the response (case-insensitive)
                response_text = message["response"].lower()
                for keyword in test_case["expected_keywords"]:
                    self.assertIn(
                        keyword.lower(),
                        response_text,
                        f"Expected keyword '{keyword}' not found in response"
                    )
                
                # Verify response time is reasonable
                if "processing_time_seconds" in message.get("metadata", {}):
                    processing_time = message["metadata"]["processing_time_seconds"]
                    self.assertLess(processing_time, 10.0, 
                                  f"Response took too long: {processing_time:.2f} seconds")
                
                # Verify user_id is included in the response
                self.assertEqual(
                    message.get("user_id"),
                    self.test_user_id,
                    "Response should include the user_id"
                )
    
    def test_conversation_continuity(self):
        """Test that conversation context is maintained across messages."""
        # First message
        response1 = self.send_chat_message(
            message="What's the weather like today?",
            conversation_id=self.test_conversation_id,
            user_id=self.test_user_id
        )
        self.assertEqual(response1["conversation_id"], self.test_conversation_id)
        
        # Verify the first response has the expected structure
        self.assertIn("message", response1)
        self.assertTrue(response1["message"].get("success"), "First message should be successful")
        
        # Follow-up message that should maintain context
        response2 = self.send_chat_message(
            message="What about tomorrow?",
            conversation_id=self.test_conversation_id,
            user_id=self.test_user_id
        )
        self.assertEqual(response2["conversation_id"], self.test_conversation_id)
        
        # Verify the second response is contextually relevant
        response_text = response2["message"]["response"].lower()
        self.assertIn(
            "weather",
            response_text,
            "Follow-up response should maintain weather context"
        )
        
        # Verify conversation history is maintained
        self.assertEqual(
            response1["conversation_id"],
            response2["conversation_id"],
            "Conversation ID should remain the same"
        )

    def test_empty_message(self):
        """Test handling of empty messages."""
        response = self.send_chat_message(
            message="",
            conversation_id=self.test_conversation_id,
            user_id=self.test_user_id
        )
        self.assertIn("message", response)
        self.assertFalse(response["message"]["success"])
        self.assertIn("error", response["message"])
        self.assertIn("empty", response["message"]["error"].lower())

    def test_missing_message_field(self):
        """Test handling of requests missing the message field."""
        response = self.client.post(
            self.base_url,
            json={
                "conversation_id": self.test_conversation_id,
                "user_id": self.test_user_id
            },
            headers={"Content-Type": "application/json"}
        )
        self.assertEqual(response.status_code, status.HTTP_422_UNPROCESSABLE_ENTITY)
        
        # Verify error response structure
        error_response = response.json()
        self.assertIn("detail", error_response)
        self.assertIsInstance(error_response["detail"], list)
        self.assertGreater(len(error_response["detail"]), 0)
        self.assertEqual(error_response["detail"][0]["type"], "missing")
        self.assertEqual(error_response["detail"][0]["loc"][-1], "message")

    def test_concurrent_requests(self):
        """Test handling of concurrent requests to the chat endpoint."""
        import threading
        from concurrent.futures import ThreadPoolExecutor
        
        # Test message that will be sent concurrently
        test_message = "Test concurrent request"
        conversation_id = f"concurrent_test_{self.test_conversation_id}"
        
        # Number of concurrent requests
        num_requests = 5
        responses = []
        
        def send_request():
            response = self.send_chat_message(
                message=test_message,
                conversation_id=conversation_id,
                user_id=self.test_user_id
            )
            responses.append(response)
        
        # Create and start threads
        with ThreadPoolExecutor(max_workers=num_requests) as executor:
            futures = [executor.submit(send_request) for _ in range(num_requests)]
            
            # Wait for all threads to complete
            for future in futures:
                future.result()
        
        # Verify all requests were processed successfully
        self.assertEqual(len(responses), num_requests, "All requests should complete")
        
        # Verify each response
        for response in responses:
            self.assertIn("message", response)
            self.assertTrue(response["message"].get("success"), "Response should be successful")
            self.assertEqual(response["conversation_id"], conversation_id)
            self.assertIn("response", response["message"])
            self.assertGreater(len(response["message"]["response"]), 0, "Response should not be empty")

    def test_large_message(self):
        """Test handling of very large messages."""
        # Create a very large message (100KB+)
        large_message = "This is a test message. " * 5000  # ~100KB
        
        response = self.send_chat_message(
            message=large_message,
            conversation_id=self.test_conversation_id,
            user_id=self.test_user_id
        )
        
        # Verify the response
        self.assertIn("message", response)
        self.assertTrue(response["message"].get("success"), "Large message should be processed")
        self.assertIn("response", response["message"])
        self.assertGreater(len(response["message"]["response"]), 0, "Response should not be empty")

    def test_special_characters(self):
        """Test handling of messages with special characters."""
        test_cases = [
            (r"Hello! @#$%^&*()_+{}|:<>?[]\\,./;'\"`~", "Special characters"),
            ("こんにちは世界", "Japanese characters"),
            ("مرحبا بالعالم", "Arabic script"),
            ("😊👍🎉", "Emojis"),
            ("Line 1\nLine 2\nLine 3", "Multiline text")
        ]
        
        for message, description in test_cases:
            with self.subTest(description=description):
                response = self.send_chat_message(
                    message=message,
                    conversation_id=f"special_chars_{self.test_conversation_id}",
                    user_id=self.test_user_id
                )
                
                self.assertIn("message", response)
                self.assertTrue(response["message"].get("success"), 
                              f"Failed to process: {description}")
                self.assertIn("response", response["message"])
                self.assertGreater(len(response["message"]["response"]), 0, 
                                 f"Empty response for: {description}")

    def test_rate_limiting(self):
        """Test that the API enforces rate limiting."""
        # Note: This test assumes the API has rate limiting enabled
        # and may need to be adjusted based on the actual rate limiting configuration
        
        # Send multiple requests in quick succession
        responses = []
        for i in range(15):  # Should be more than the rate limit
            response = self.send_chat_message(
                message=f"Test rate limiting message {i}",
                conversation_id=f"rate_limit_test_{self.test_conversation_id}",
                user_id=self.test_user_id
            )
            responses.append(response)
        
        # Check if any requests were rate limited (HTTP 429)
        rate_limited = any(
            "message" in r and "error" in r["message"] and "rate limit" in r["message"]["error"].lower()
            for r in responses
        )
        
        if rate_limited:
            # If rate limited, at least one response should indicate this
            self.assertTrue(any(
                "message" in r and "error" in r["message"] 
                and "rate limit" in r["message"]["error"].lower()
                for r in responses
            ), "Expected rate limiting error not found in responses")
        else:
            # If not rate limited, all responses should be successful
            for response in responses:
                self.assertIn("message", response)
                self.assertTrue(response["message"].get("success", False), 
                              f"Unexpected error: {response}")

    def test_invalid_json(self):
        """Test handling of invalid JSON in the request body."""
        response = self.client.post(
            self.base_url,
            content="This is not valid JSON",
            headers={"Content-Type": "application/json"}
        )
        
        self.assertEqual(response.status_code, status.HTTP_422_UNPROCESSABLE_ENTITY)
        error_response = response.json()
        self.assertIn("detail", error_response)
        self.assertIn("JSON", str(error_response))

    def test_unsupported_media_type(self):
        """Test handling of requests with unsupported media types."""
        response = self.client.post(
            self.base_url,
            json={"message": "Test message"},
            headers={"Content-Type": "text/plain"}
        )
        
        self.assertEqual(response.status_code, status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)
        error_response = response.json()
        self.assertIn("detail", error_response)
        self.assertIn("Unsupported media type", error_response["detail"])

    def test_agent_fallback_mechanism(self):
        """Test that the system falls back to a default agent when needed."""
        # Test with a message that might not match any specific agent's criteria
        response = self.send_chat_message(
            message="This is a test message that doesn't match any specific agent's criteria.",
            conversation_id=self.test_conversation_id,
            user_id=self.test_user_id
        )
        
        # The system should still respond successfully, likely using the default agent
        self.assertIn("message", response)
        self.assertTrue(response["message"].get("success"), "Fallback agent should handle the message")
        self.assertIn("response", response["message"])
        self.assertGreater(len(response["message"]["response"]), 0, "Response should not be empty")
        
        # Verify the response indicates it was handled by the default agent
        self.assertIn("agent_used", response["message"].get("metadata", {}))
        self.assertEqual(
            response["message"]["metadata"]["agent_used"],
            AgentType.RESPONDER.value.lower(),
            "Expected fallback to default responder agent"
        )

if __name__ == "__main__":
    unittest.main(failfast=True)
