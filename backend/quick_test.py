#!/usr/bin/env python3
"""
Quick Test Script for Naarad AI Assistant

This script provides a simple way to test basic functionality
without running the full test suite.

Usage:
    python quick_test.py
"""

import requests
import json
import time
import os
from typing import Dict, Any

def test_server_health():
    """Test if the server is running and healthy."""
    print("🔍 Testing server health...")
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Server is healthy: {data['status']}")
            return True
        else:
            print(f"❌ Server returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Is it running?")
        print("   Start with: uvicorn main:app --reload")
        return False
    except Exception as e:
        print(f"❌ Error testing server: {e}")
        return False

def test_chat_endpoint():
    """Test the chat endpoint with a simple message."""
    print("\n💬 Testing chat endpoint...")
    
    try:
        start_time = time.time()
        response = requests.post(
            "http://localhost:8000/api/v1/chat",
            json={"message": "Hello, how are you?"},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Handle nested response structure
            if isinstance(data.get("message"), dict):
                # Extract from nested response
                message = data["message"].get("output", str(data["message"]))
            else:
                message = data.get("message", "")
            
            response_time = time.time() - start_time
            print(f"✅ Chat response received in {response_time:.2f}s")
            print(f"   Response: {message[:100]}...")
            return True
        else:
            print(f"❌ Chat endpoint failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error testing chat endpoint: {str(e)}")
        return False

def test_configuration():
    """Test basic configuration."""
    print("\n⚙️  Testing configuration...")
    
    # Check if .env file exists
    if os.path.exists(".env"):
        print("✅ .env file exists")
    else:
        print("⚠️  .env file not found")
        print("   Create it with: cp env.example .env")
    
    # Check required environment variables
    required_vars = ["GROQ_API_KEY", "BRAVE_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if os.getenv(var):
            print(f"✅ {var} is set")
        else:
            print(f"⚠️  {var} is not set")
            missing_vars.append(var)
    
    if missing_vars:
        print(f"\n⚠️  Missing environment variables: {', '.join(missing_vars)}")
        print("   Add them to your .env file")
        return False
    
    return True

def test_api_documentation():
    """Test if API documentation is accessible."""
    print("\n📚 Testing API documentation...")
    
    try:
        response = requests.get("http://localhost:8000/docs", timeout=5)
        if response.status_code == 200:
            print("✅ API documentation is accessible at /docs")
            return True
        else:
            print(f"⚠️  API documentation returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error accessing API documentation: {e}")
        return False

def main():
    """Run all quick tests."""
    print("🚀 Naarad AI Assistant - Quick Test Suite")
    print("=" * 50)
    
    tests = [
        ("Configuration", test_configuration),
        ("Server Health", test_server_health),
        ("API Documentation", test_api_documentation),
        ("Chat Endpoint", test_chat_endpoint),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 QUICK TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All quick tests passed!")
        print("   Your Naarad AI Assistant is working correctly.")
    else:
        print("💥 Some tests failed.")
        print("   Check the output above for details.")
        print("   Run the full test suite with: python run_all_tests.py")

if __name__ == "__main__":
    main() 