"""Test script for updated agents.

This script tests the functionality of the updated agents to ensure they work
with the latest LangChain and Pydantic versions.
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

async def test_responder_agent():
    """Test the responder agent with a simple query."""
    from agent.agents.responder import ResponderAgent
    
    print("\n=== Testing Responder Agent ===")
    
    # Initialize the responder agent
    responder = ResponderAgent({
        'name': 'test_responder',
        'description': 'Test responder agent',
        'model_name': 'gpt-3.5-turbo',
        'temperature': 0.7,
        'system_prompt': 'You are a helpful assistant.',
        'max_iterations': 3,
        'verbose': True
    })
    
    # Test a simple query
    response = await responder.process("Hello, how are you?")
    print("\nResponder Agent Response:")
    print(f"Output: {response['output']}")
    print(f"Metadata: {response['metadata']}")
    
    return response

async def test_analyst_agent():
    """Test the analyst agent with an analysis query."""
    from agent.agents.analyst import AnalystAgent
    
    print("\n=== Testing Analyst Agent ===")
    
    # Initialize the analyst agent
    analyst = AnalystAgent({
        'name': 'test_analyst',
        'description': 'Test analyst agent',
        'model_name': 'gpt-3.5-turbo',
        'temperature': 0.5,
        'system_prompt': 'You are an analytical assistant. Analyze the given input and provide insights.',
        'max_iterations': 5,
        'verbose': True
    })
    
    # Test an analysis query
    response = await analyst.process("Analyze the impact of artificial intelligence on modern businesses.")
    print("\nAnalyst Agent Response:")
    print(f"Output: {response['output']}")
    print(f"Metadata: {response['metadata']}")
    
    return response

async def main():
    """Run all tests."""
    print("Starting agent tests...")
    
    # Test responder agent
    responder_result = await test_responder_agent()
    
    # Test analyst agent
    analyst_result = await test_analyst_agent()
    
    print("\n=== Test Summary ===")
    print(f"Responder agent test: {'PASSED' if responder_result['output'] else 'FAILED'}")
    print(f"Analyst agent test: {'PASSED' if analyst_result['output'] else 'FAILED'}")

if __name__ == "__main__":
    asyncio.run(main())
