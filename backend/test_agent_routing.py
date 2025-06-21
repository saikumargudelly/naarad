import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the agent components
from agent.naarad_agent import naarad_agent
from agent.orchestrator import AgentOrchestrator
from agent.agents import (
    ResponderAgent, 
    ResearcherAgent, 
    AnalystAgent, 
    QualityAgent
)

# Test cases with expected agent routing
TEST_CASES = [
    # Simple greetings and conversations (should go to responder)
    {"input": "Hello, how are you?", "expected_agent": "responder"},
    {"input": "Hi there!", "expected_agent": "responder"},
    
    # Simple math questions (should go to responder)
    {"input": "What is 2 + 2?", "expected_agent": "responder"},
    {"input": "Calculate 15 * 3", "expected_agent": "responder"},
    
    # General knowledge (should go to responder if simple, researcher if needs current info)
    {"input": "Who is Albert Einstein?", "expected_agent": "responder"},
    {"input": "What is the capital of France?", "expected_agent": "responder"},
    
    # Research queries (should go to researcher)
    {"input": "Find the latest news about AI", "expected_agent": "researcher"},
    {"input": "Search for recent developments in quantum computing", "expected_agent": "researcher"},
    
    # Analysis queries (should go to analyst)
    {"input": "Compare Python and JavaScript", "expected_agent": "analyst"},
    {"input": "Analyze the pros and cons of electric vehicles", "expected_agent": "analyst"},
    
    # Image-related queries (should go to responder with vision tool)
    {"input": "What's in this image?", "context": {"has_image": True}, "expected_agent": "responder"},
    
    # Complex queries that need reasoning (should go to analyst)
    {"input": "Explain quantum computing in simple terms", "expected_agent": "analyst"},
]

async def run_test_case(test_case: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single test case and verify the agent routing."""
    input_text = test_case["input"]
    expected_agent = test_case["expected_agent"]
    context = test_case.get("context", {})
    
    logger.info(f"\n{'='*80}")
    logger.info(f"TEST CASE: {input_text}")
    logger.info(f"EXPECTED AGENT: {expected_agent}")
    
    try:
        # Process the query through the orchestrator
        response = await naarad_agent.process_message(
            input_text, 
            context=context,
            conversation_id="test_conversation",
            user_id="test_user"
        )
        
        # Get the actual agent used
        actual_agent = response.get('metadata', {}).get('agent_used', 'unknown')
        
        # Check if routing was correct
        is_correct = actual_agent.lower() == expected_agent.lower()
        result = "PASS" if is_correct else "FAIL"
        
        logger.info(f"RESULT: {result}")
        logger.info(f"ACTUAL AGENT: {actual_agent}")
        logger.info(f"RESPONSE: {response.get('output', '')[:200]}...")
        
        return {
            "test_case": input_text,
            "expected_agent": expected_agent,
            "actual_agent": actual_agent,
            "result": result,
            "response": response
        }
        
    except Exception as e:
        logger.error(f"Error running test case: {str(e)}", exc_info=True)
        return {
            "test_case": input_text,
            "expected_agent": expected_agent,
            "actual_agent": "error",
            "result": "ERROR",
            "error": str(e)
        }

async def run_all_tests():
    """Run all test cases and print a summary."""
    logger.info("Starting agent routing tests...")
    
    # Initialize the agent if needed
    await naarad_agent.ensure_initialized()
    
    results = []
    
    # Run each test case
    for test_case in TEST_CASES:
        result = await run_test_case(test_case)
        results.append(result)
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)
    
    passed = sum(1 for r in results if r["result"] == "PASS")
    failed = sum(1 for r in results if r["result"] == "FAIL")
    errors = sum(1 for r in results if r["result"] == "ERROR")
    
    logger.info(f"Total tests: {len(results)}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Errors: {errors}")
    
    # Print failed tests
    if failed > 0:
        logger.info("\nFAILED TESTS:")
        for result in results:
            if result["result"] == "FAIL":
                logger.info(f"- {result['test_case']}")
                logger.info(f"  Expected: {result['expected_agent']}")
                logger.info(f"  Got: {result['actual_agent']}")
    
    # Print errors
    if errors > 0:
        logger.info("\nERRORS:")
        for result in results:
            if result["result"] == "ERROR":
                logger.info(f"- {result['test_case']}")
                logger.info(f"  Error: {result.get('error', 'Unknown error')}")
    
    return results

if __name__ == "__main__":
    asyncio.run(run_all_tests())
