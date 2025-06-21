"""Test script to verify agent imports and basic functionality."""

import os
import sys
import logging

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that we can import all the necessary modules."""
    try:
        # Add the current directory to the Python path for local imports
        sys.path.insert(0, os.path.abspath('.'))
        
        # Import the main agent classes from the refactored structure
        from agent.agents import (
            BaseAgent,
            AgentConfig,
            ResponderAgent,
            create_agent,
            AgentManager
        )
        logger.info("✅ Successfully imported core agent classes and functions")
        return True
    except ImportError as e:
        logger.error(f"❌ Import error: {str(e)}")
        return False

def test_responder_creation():
    """Test creating a responder agent."""
    try:
        from agent.agents import ResponderAgent, create_agent
        
        # Test creating a responder agent with proper config
        config = {
            'name': 'test_responder',
            'description': 'Test responder agent',
            'model_name': 'llama3-8b-8192',
            'temperature': 0.7
        }
        responder = ResponderAgent(config=config)
        logger.info("✅ Successfully created ResponderAgent directly")
        
        # Test creating with factory function
        responder2 = create_agent('responder', **config)
        logger.info("✅ Successfully created ResponderAgent via factory")
        
        return True
    except Exception as e:
        logger.error(f"❌ Failed to create responder agent: {str(e)}")
        return False

def test_agent_manager():
    """Test basic AgentManager functionality."""
    try:
        from agent.agents import AgentManager
        
        manager = AgentManager()
        
        # Test registering an agent
        manager.register_agent('test_agent', 'test_agent_type')
        assert 'test_agent' in manager.list_agents(), "Agent not registered"
        
        # Test getting an agent
        agent = manager.get_agent('test_agent')
        assert agent == 'test_agent_type', "Incorrect agent retrieved"
        
        # Test removing an agent
        assert manager.remove_agent('test_agent'), "Failed to remove agent"
        assert 'test_agent' not in manager.list_agents(), "Agent not removed"
        
        logger.info("✅ Successfully tested AgentManager")
        return True
    except Exception as e:
        logger.error(f"❌ AgentManager test failed: {str(e)}")
        return False

def main():
    """Run all tests."""
    tests = [
        ("Import Tests", test_imports),
        ("Responder Agent Creation", test_responder_creation),
        ("Agent Manager Tests", test_agent_manager)
    ]
    
    print("\n=== Running Agent Tests ===\n")
    
    passed = 0
    for name, test_func in tests:
        print(f"Running test: {name}")
        if test_func():
            print(f"✅ {name} passed\n")
            passed += 1
        else:
            print(f"❌ {name} failed\n")
    
    print(f"\n=== Test Results: {passed}/{len(tests)} tests passed ===")
    return passed == len(tests)

if __name__ == "__main__":
    if not main():
        sys.exit(1)
