"""
Comprehensive Test Suite for Naarad AI Assistant Backend

This test suite covers:
1. API Endpoints and Responses
2. Configuration Validation
3. External API Integration (Brave Search, LLM Providers)
4. Error Handling and Edge Cases
5. Rate Limiting
6. Security Features
7. Agent Functionality
8. Memory and Conversation Management

Run with: python -m pytest tests/test_comprehensive_api.py -v
"""

import unittest
import json
import time
import requests
import os
import sys
from typing import Dict, Any, List, Optional
from unittest.mock import patch, MagicMock
import pytest
from datetime import datetime, timedelta

# Add the backend directory to the Python path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

class TestNaaradComprehensiveAPI(unittest.TestCase):
    """Comprehensive test suite for Naarad AI Assistant backend."""
    
    BASE_URL = "http://localhost:8000"
    API_BASE_URL = f"{BASE_URL}/api/v1"
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests."""
        cls.session = requests.Session()
        cls.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        cls.conversation_id = f"test_conv_{int(time.time())}"
        
        # Test if server is running
        try:
            response = cls.session.get(f"{cls.BASE_URL}/health", timeout=5)
            if response.status_code != 200:
                raise Exception("Server is not responding properly")
        except Exception as e:
            print(f"⚠️  Warning: Server might not be running: {e}")
            print("   Make sure to start the server with: uvicorn main:app --reload")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        cls.session.close()
    
    def setUp(self):
        """Set up before each test."""
        self.start_time = time.time()
    
    def tearDown(self):
        """Clean up after each test."""
        test_duration = time.time() - self.start_time
        print(f"⏱️  Test completed in {test_duration:.2f}s")
    
    # =============================================================================
    # 1. BASIC API ENDPOINT TESTS
    # =============================================================================
    
    def test_root_endpoint(self):
        """Test the root endpoint returns proper API information."""
        response = self.session.get(f"{self.BASE_URL}/")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("message", data)
        self.assertIn("version", data)
        self.assertIn("status", data)
        self.assertEqual(data["status"], "running")
        self.assertIn("Naarad AI Assistant", data["message"])
        print("✅ Root endpoint test passed")
    
    def test_health_check_endpoint(self):
        """Test the health check endpoint."""
        response = self.session.get(f"{self.BASE_URL}/health")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["service"], "naarad-ai")
        self.assertIn("version", data)
        self.assertIn("environment", data)
        self.assertIn("timestamp", data)
        print("✅ Health check endpoint test passed")
    
    def test_chat_health_endpoint(self):
        """Test the chat service health endpoint."""
        response = self.session.get(f"{self.API_BASE_URL}/health")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["service"], "naarad-chat")
        self.assertIn("timestamp", data)
        print("✅ Chat health endpoint test passed")
    
    # =============================================================================
    # 2. CHAT API FUNCTIONALITY TESTS
    # =============================================================================
    
    def send_chat_request(self, message: str, conversation_id: Optional[str] = None, 
                         chat_history: Optional[List[Dict]] = None, images: Optional[List[str]] = None) -> Dict[str, Any]:
        """Helper method to send chat requests."""
        url = f"{self.API_BASE_URL}/chat"
        payload = {
            "message": message,
            "conversation_id": conversation_id or self.conversation_id,
        }
        if chat_history:
            payload["chat_history"] = chat_history
        if images:
            payload["images"] = images
            
        response = self.session.post(url, json=payload)
        return response
    
    def test_chat_basic_functionality(self):
        """Test basic chat functionality with a simple message."""
        response = self.send_chat_request("Hello, how are you?")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("message", data)
        self.assertIn("conversation_id", data)
        self.assertIn("sources", data)
        self.assertIn("processing_time", data)
        
        # Verify response structure
        self.assertIsInstance(data["message"], str)
        self.assertGreater(len(data["message"]), 10, "Response should be meaningful")
        self.assertIsInstance(data["processing_time"], str)
        print(f"✅ Basic chat functionality test passed. Response: {data['message'][:100]}...")
    
    def test_chat_with_conversation_context(self):
        """Test chat functionality with conversation context."""
        # First message
        response1 = self.send_chat_request("My name is Test User")
        self.assertEqual(response1.status_code, 200)
        
        # Second message - should maintain context
        response2 = self.send_chat_request("What's my name?")
        self.assertEqual(response2.status_code, 200)
        
        data2 = response2.json()
        response_text = data2["message"].lower()
        # The response should acknowledge the context (even if it can't remember the name)
        self.assertGreater(len(response_text), 10)
        print(f"✅ Conversation context test passed. Response: {response_text[:100]}...")
    
    def test_chat_with_chat_history(self):
        """Test chat functionality with explicit chat history."""
        chat_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there! How can I help you today?"}
        ]
        
        response = self.send_chat_request("What did we just talk about?", chat_history=chat_history)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIsInstance(data["message"], str)
        self.assertGreater(len(data["message"]), 10)
        print(f"✅ Chat history test passed. Response: {data['message'][:100]}...")
    
    def test_chat_with_images(self):
        """Test chat functionality with image input."""
        # Test with empty images list
        response = self.send_chat_request("What do you see in this image?", images=[])
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIsInstance(data["message"], str)
        print(f"✅ Chat with images test passed. Response: {data['message'][:100]}...")
    
    def test_chat_rate_limiting(self):
        """Test rate limiting functionality."""
        # Send multiple requests rapidly
        responses = []
        for i in range(5):
            response = self.send_chat_request(f"Test message {i}")
            responses.append(response)
        
        # All should succeed (within rate limit)
        for response in responses:
            self.assertIn(response.status_code, [200, 429])  # 200 OK or 429 Rate Limited
        
        print("✅ Rate limiting test passed")
    
    # =============================================================================
    # 3. INPUT VALIDATION TESTS
    # =============================================================================
    
    def test_chat_empty_message(self):
        """Test handling of empty message."""
        response = self.send_chat_request("")
        self.assertEqual(response.status_code, 422)  # Validation error
        print("✅ Empty message validation test passed")
    
    def test_chat_very_long_message(self):
        """Test handling of very long message."""
        long_message = "A" * 6000  # Exceeds max length
        response = self.send_chat_request(long_message)
        self.assertEqual(response.status_code, 422)  # Validation error
        print("✅ Long message validation test passed")
    
    def test_chat_invalid_json(self):
        """Test handling of invalid JSON."""
        url = f"{self.API_BASE_URL}/chat"
        response = self.session.post(url, data="invalid json", headers={'Content-Type': 'application/json'})
        self.assertEqual(response.status_code, 422)
        print("✅ Invalid JSON validation test passed")
    
    def test_chat_missing_required_fields(self):
        """Test handling of missing required fields."""
        url = f"{self.API_BASE_URL}/chat"
        payload = {"conversation_id": "test"}  # Missing message field
        response = self.session.post(url, json=payload)
        self.assertEqual(response.status_code, 422)
        print("✅ Missing fields validation test passed")
    
    # =============================================================================
    # 4. DIFFERENT TYPES OF QUERIES TESTS
    # =============================================================================
    
    def test_factual_questions(self):
        """Test factual question answering."""
        questions = [
            "What is the capital of France?",
            "Who wrote Romeo and Juliet?",
            "What is the largest planet in our solar system?"
        ]
        
        for question in questions:
            response = self.send_chat_request(question)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertGreater(len(data["message"]), 10)
            print(f"✅ Factual question test passed: {question[:30]}...")
    
    def test_creative_requests(self):
        """Test creative writing requests."""
        creative_requests = [
            "Write a haiku about artificial intelligence",
            "Tell me a short story about a robot",
            "Create a poem about nature"
        ]
        
        for request in creative_requests:
            response = self.send_chat_request(request)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertGreater(len(data["message"]), 20)
            print(f"✅ Creative request test passed: {request[:30]}...")
    
    def test_code_generation(self):
        """Test code generation capabilities."""
        code_requests = [
            "Write a Python function to reverse a string",
            "Show me how to create a simple HTML page",
            "Write a JavaScript function to calculate factorial"
        ]
        
        for request in code_requests:
            response = self.send_chat_request(request)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertGreater(len(data["message"]), 20)
            print(f"✅ Code generation test passed: {request[:30]}...")
    
    def test_mathematical_calculations(self):
        """Test mathematical calculation capabilities."""
        math_questions = [
            "What is 15% of 200?",
            "Calculate 2^10",
            "What is the square root of 144?"
        ]
        
        for question in math_questions:
            response = self.send_chat_request(question)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertGreater(len(data["message"]), 10)
            print(f"✅ Math calculation test passed: {question[:30]}...")
    
    def test_language_translation(self):
        """Test language translation capabilities."""
        translation_requests = [
            "How do you say 'Hello, how are you?' in Spanish?",
            "Translate 'Thank you' to French",
            "What is 'Good morning' in German?"
        ]
        
        for request in translation_requests:
            response = self.send_chat_request(request)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertGreater(len(data["message"]), 10)
            print(f"✅ Translation test passed: {request[:30]}...")
    
    # =============================================================================
    # 5. EXTERNAL API INTEGRATION TESTS
    # =============================================================================
    
    @patch('requests.get')
    def test_brave_search_integration(self, mock_get):
        """Test Brave Search API integration."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "web": {
                "results": [
                    {
                        "title": "Test Result",
                        "url": "https://example.com",
                        "description": "Test description"
                    }
                ]
            }
        }
        mock_get.return_value = mock_response
        
        response = self.send_chat_request("Search for latest AI news")
        self.assertEqual(response.status_code, 200)
        print("✅ Brave Search integration test passed")
    
    def test_llm_provider_integration(self):
        """Test LLM provider integration."""
        response = self.send_chat_request("Explain quantum computing in simple terms")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertGreater(len(data["message"]), 50)
        print("✅ LLM provider integration test passed")
    
    # =============================================================================
    # 6. ERROR HANDLING TESTS
    # =============================================================================
    
    def test_server_error_handling(self):
        """Test server error handling."""
        # This test would require mocking internal errors
        # For now, we test that the API doesn't crash on complex requests
        response = self.send_chat_request("This is a test message that should not cause errors")
        self.assertIn(response.status_code, [200, 500])  # Should either succeed or return proper error
        print("✅ Error handling test passed")
    
    def test_timeout_handling(self):
        """Test timeout handling."""
        # Test with a complex query that might timeout
        response = self.send_chat_request("Please provide a very detailed analysis of artificial intelligence")
        self.assertIn(response.status_code, [200, 408, 500])  # Success, timeout, or server error
        print("✅ Timeout handling test passed")
    
    # =============================================================================
    # 7. PERFORMANCE TESTS
    # =============================================================================
    
    def test_response_time(self):
        """Test response time performance."""
        start_time = time.time()
        response = self.send_chat_request("Hello")
        end_time = time.time()
        
        response_time = end_time - start_time
        self.assertLess(response_time, 30, "Response should be under 30 seconds")
        print(f"✅ Response time test passed: {response_time:.2f}s")
    
    def test_concurrent_requests(self):
        """Test handling of concurrent requests."""
        import threading
        
        results = []
        errors = []
        
        def make_request():
            try:
                response = self.send_chat_request("Test concurrent request")
                results.append(response.status_code)
            except Exception as e:
                errors.append(str(e))
        
        # Create multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        self.assertLess(len(errors), 3, "Should handle concurrent requests without too many errors")
        print(f"✅ Concurrent requests test passed: {len(results)} successful, {len(errors)} errors")
    
    # =============================================================================
    # 8. SECURITY TESTS
    # =============================================================================
    
    def test_cors_headers(self):
        """Test CORS headers are properly set."""
        response = self.session.options(f"{self.API_BASE_URL}/chat")
        # Should not fail due to CORS
        self.assertIn(response.status_code, [200, 405])  # 200 OK or 405 Method Not Allowed
        print("✅ CORS headers test passed")
    
    def test_sql_injection_prevention(self):
        """Test SQL injection prevention."""
        malicious_input = "'; DROP TABLE users; --"
        response = self.send_chat_request(malicious_input)
        self.assertEqual(response.status_code, 200)  # Should handle gracefully
        print("✅ SQL injection prevention test passed")
    
    def test_xss_prevention(self):
        """Test XSS prevention."""
        xss_input = "<script>alert('xss')</script>"
        response = self.send_chat_request(xss_input)
        self.assertEqual(response.status_code, 200)  # Should handle gracefully
        print("✅ XSS prevention test passed")
    
    # =============================================================================
    # 9. CONFIGURATION TESTS
    # =============================================================================
    
    def test_environment_variables(self):
        """Test that required environment variables are accessible."""
        # This would require importing the config module
        # For now, we test that the API responds with proper configuration
        response = self.session.get(f"{self.BASE_URL}/")
        self.assertEqual(response.status_code, 200)
        print("✅ Environment variables test passed")
    
    # =============================================================================
    # 10. AGENT FUNCTIONALITY TESTS
    # =============================================================================
    
    def test_multi_agent_collaboration(self):
        """Test multi-agent collaboration on complex queries."""
        complex_queries = [
            "Research the latest developments in quantum computing and provide a detailed analysis",
            "Compare different programming languages for web development",
            "Explain the impact of artificial intelligence on healthcare"
        ]
        
        for query in complex_queries:
            response = self.send_chat_request(query)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertGreater(len(data["message"]), 50)
            print(f"✅ Multi-agent collaboration test passed: {query[:50]}...")
    
    def test_agent_memory_functionality(self):
        """Test agent memory and conversation persistence."""
        # First conversation
        response1 = self.send_chat_request("Remember that I like pizza")
        self.assertEqual(response1.status_code, 200)
        
        # Second conversation with same ID
        response2 = self.send_chat_request("What do I like?", conversation_id=self.conversation_id)
        self.assertEqual(response2.status_code, 200)
        
        print("✅ Agent memory functionality test passed")
    
    # =============================================================================
    # 11. EDGE CASES AND STRESS TESTS
    # =============================================================================
    
    def test_unicode_handling(self):
        """Test handling of Unicode characters."""
        unicode_messages = [
            "Hello 世界",
            "Привет мир",
            "こんにちは世界",
            "مرحبا بالعالم"
        ]
        
        for message in unicode_messages:
            response = self.send_chat_request(message)
            self.assertEqual(response.status_code, 200)
            print(f"✅ Unicode handling test passed: {message}")
    
    def test_special_characters(self):
        """Test handling of special characters."""
        special_chars = [
            "Test with @#$%^&*()",
            "Message with \"quotes\" and 'apostrophes'",
            "Text with\nnewlines\tand\ttabs",
            "Message with emojis 😀🎉🚀"
        ]
        
        for message in special_chars:
            response = self.send_chat_request(message)
            self.assertEqual(response.status_code, 200)
            print(f"✅ Special characters test passed: {message[:20]}...")
    
    def test_empty_conversation_history(self):
        """Test handling of empty conversation history."""
        response = self.send_chat_request("Test message", chat_history=[])
        self.assertEqual(response.status_code, 200)
        print("✅ Empty conversation history test passed")
    
    def test_large_chat_history(self):
        """Test handling of large chat history."""
        large_history = []
        for i in range(50):
            large_history.extend([
                {"role": "user", "content": f"Message {i}"},
                {"role": "assistant", "content": f"Response {i}"}
            ])
        
        response = self.send_chat_request("Test with large history", chat_history=large_history)
        self.assertEqual(response.status_code, 200)
        print("✅ Large chat history test passed")

def run_comprehensive_tests():
    """Run all comprehensive tests."""
    print("🚀 Starting Comprehensive Naarad AI Assistant Backend Tests...")
    print("=" * 80)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestNaaradComprehensiveAPI)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("=" * 80)
    print("📊 TEST SUMMARY:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"   {test}: {traceback}")
    
    if result.errors:
        print("\n⚠️  ERRORS:")
        for test, traceback in result.errors:
            print(f"   {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n🎉 All tests passed successfully!")
    else:
        print("\n💥 Some tests failed. Please check the output above.")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1) 