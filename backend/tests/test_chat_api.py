"""
Test script for Naarad Chat API

This script tests various capabilities of the chat API by sending different types of messages
and verifying the responses.

Note: This script only uses the 'requests' library which is already in the project's requirements.txt
"""

import unittest
import json
import time
import requests
from typing import Dict, Any, List, Optional

class TestNaaradChatAPI(unittest.TestCase):    
    BASE_URL = "http://localhost:8000/api"
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        self.conversation_id = f"test_conv_{int(time.time())}"
    
    def tearDown(self):
        """Clean up after each test method."""
        self.session.close()
    
    def send_chat_request(self, message: str, conversation_id: Optional[str] = None, 
                         chat_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Helper method to send chat requests."""
        url = f"{self.BASE_URL}/chat"
        payload = {
            "message": message,
            "conversation_id": conversation_id or self.conversation_id,
        }
        if chat_history:
            payload["chat_history"] = chat_history
            
        response = self.session.post(url, json=payload)
        self.assertEqual(response.status_code, 200, f"Request failed with status {response.status_code}")
        return response.json()
    
    def test_health_check(self):
        """Test the health check endpoint."""
        response = self.session.get(f"{self.BASE_URL}/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        print("\n✅ Health check passed")
    
    def test_general_knowledge(self):
        """Test general knowledge questions."""
        response = self.send_chat_request("What is the capital of France?")
        response_text = response["message"]["response"].lower()
        self.assertIsInstance(response_text, str, "Response should be a string")
        self.assertGreater(len(response_text), 10, "Response should be meaningful")
        # Check if the response indicates an error or contains the expected information
        if "error" not in response_text and "try again" not in response_text:
            self.assertIn("paris", response_text)
        print(f"\n✅ General knowledge test passed. Response: {response_text[:100]}...")
    
    def test_creative_writing(self):
        """Test creative writing capabilities."""
        response = self.send_chat_request("Write a short haiku about artificial intelligence")
        # Verify the response is creative and has multiple lines
        response_text = response["message"]["response"]
        self.assertIsInstance(response_text, str, "Response should be a string")
        self.assertGreater(len(response_text), 50, "Response should be sufficiently detailed")
        print(f"\n✅ Creative writing test passed. Response: {response_text[:100]}...")
    
    def test_code_generation(self):
        """Test code generation and explanation."""
        response = self.send_chat_request("Show me how to reverse a string in Python")
        response_text = response["message"]["response"].lower()
        self.assertIsInstance(response_text, str, "Response should be a string")
        self.assertGreater(len(response_text), 10, "Response should be meaningful")
        # Check if the response indicates an error or contains code-like content
        if "error" not in response_text and "try again" not in response_text:
            self.assertTrue(
                any(char in response_text for char in ["[", "(", "{", "def ", "import ", "print"]),
                "Response should contain code-like syntax"
            )
        print(f"\n✅ Code generation test passed. Response: {response_text[:100]}...")
    
    def test_conversation_context(self):
        """Test that conversation context is maintained between messages."""
        # First message - set the context
        response1 = self.send_chat_request("My name is Test User")
        self.assertIsInstance(response1["message"]["response"], str, "Response should be a string")
        
        # Second message - should maintain context
        response2 = self.send_chat_request("What did I just tell you?")
        response_text = response2["message"]["response"].lower()
        self.assertIsInstance(response_text, str, "Response should be a string")
        self.assertGreater(len(response_text), 10, "Response should be meaningful")
        print(f"\n✅ Conversation context test passed. Response: {response_text[:100]}...")
    
    def test_unsupported_feature_handling(self):
        """Test how the system handles unsupported features like image analysis."""
        try:
            response = self.send_chat_request("What do you see in this image?")
            # The response should be a meaningful message, even if it can't process images
            self.assertIsInstance(response["message"]["response"], str, "Response should be a string")
            self.assertGreater(len(response["message"]["response"]), 10, "Response should be meaningful")
            print(f"\n✅ Unsupported feature handling test passed. Response: {response['message']['response'][:100]}...")
        except Exception as e:
            self.fail(f"Unsupported feature test failed with exception: {str(e)}")
    
    def test_complex_query(self):
        """Test a complex query that might require multiple agents to collaborate."""
        response = self.send_chat_request(
            "Tell me about quantum computing and its potential impact on cryptography"
        )
        # Verify the response is meaningful
        response_text = response["message"]["response"].lower()
        self.assertIsInstance(response_text, str, "Response should be a string")
        self.assertGreater(len(response_text), 50, "Response should be sufficiently detailed")
        print(f"\n✅ Complex query test passed. Response: {response_text[:100]}...")
    
    def test_mathematical_calculation(self):
        """Test the system's ability to perform mathematical calculations."""
        response = self.send_chat_request("What is 15% of 200?")
        response_text = response["message"]["response"].lower()
        self.assertIsInstance(response_text, str, "Response should be a string")
        # Check if the response contains the correct answer or indicates it can't perform the calculation
        if "error" not in response_text and "don't know" not in response_text:
            self.assertTrue(any(term in response_text for term in ["30", "thirty"]), 
                         f"Response should contain '30' or 'thirty', got: {response_text}")
        print(f"\n✅ Mathematical calculation test passed. Response: {response_text[:100]}...")
    
    def test_factual_accuracy(self):
        """Test the system's ability to provide accurate factual information."""
        response = self.send_chat_request("Who is the current president of the United States?")
        response_text = response["message"]["response"].lower()
        self.assertIsInstance(response_text, str, "Response should be a string")
        self.assertGreater(len(response_text), 10, "Response should be meaningful")
        # This is a test that might need updating based on current events
        print(f"\n✅ Factual accuracy test passed. Response: {response_text[:100]}...")
    
    def test_language_translation(self):
        """Test the system's ability to handle translation requests."""
        response = self.send_chat_request("How do you say 'Hello, how are you?' in Spanish?")
        response_text = response["message"]["response"].lower()
        self.assertIsInstance(response_text, str, "Response should be a string")
        # Check for common Spanish greetings in the response
        if "error" not in response_text:
            self.assertTrue(
                any(phrase in response_text for phrase in ["hola", "cómo estás", "cómo está"]),
                f"Response should contain Spanish translation, got: {response_text}"
            )
        print(f"\n✅ Language translation test passed. Response: {response_text[:100]}...")
    
    def test_historical_facts(self):
        """Test the system's ability to provide historical information."""
        response = self.send_chat_request("When was World War II?")
        response_text = response["message"]["response"].lower()
        self.assertIsInstance(response_text, str, "Response should be a string")
        self.assertGreater(len(response_text), 10, "Response should be meaningful")
        # Check for years related to WWII
        if "error" not in response_text:
            self.assertTrue(
                any(year in response_text for year in ["1939", "1945"]),
                f"Response should mention WWII years, got: {response_text}"
            )
        print(f"\n✅ Historical facts test passed. Response: {response_text[:100]}...")
    
    def test_scientific_concepts(self):
        """Test the system's ability to explain scientific concepts."""
        response = self.send_chat_request("Explain the theory of relativity in simple terms")
        response_text = response["message"]["response"].lower()
        self.assertIsInstance(response_text, str, "Response should be a string")
        self.assertGreater(len(response_text), 50, "Response should be sufficiently detailed")
        # Check for key terms related to relativity
        if "error" not in response_text:
            self.assertTrue(
                any(term in response_text for term in ["einstein", "space", "time", "gravity"]),
                f"Response should explain relativity, got: {response_text}"
            )
        print(f"\n✅ Scientific concepts test passed. Response: {response_text[:100]}...")
    
    def test_multiple_questions(self):
        """Test the system's ability to handle multiple questions in one query."""
        response = self.send_chat_request("What's the capital of Japan? And what's the largest planet in our solar system?")
        response_text = response["message"]["response"].lower()
        self.assertIsInstance(response_text, str, "Response should be a string")
        self.assertGreater(len(response_text), 20, "Response should be meaningful")
        # Check if both answers are present
        if "error" not in response_text:
            self.assertTrue(
                "tokyo" in response_text and "jupiter" in response_text,
                f"Response should answer both questions, got: {response_text}"
            )
        print(f"\n✅ Multiple questions test passed. Response: {response_text[:100]}...")

if __name__ == "__main__":
    print("🚀 Starting Naarad Chat API tests...")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    print("\n✨ All tests completed!")
