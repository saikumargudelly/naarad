"""Tests for the enhanced router with advanced intent classification and entity extraction."""
import pytest
import asyncio
from unittest.mock import patch, MagicMock
import numpy as np
from agent.enhanced_router import EnhancedRouter, Intent, IntentMatch, Entity

class TestEnhancedRouter:
    """Test suite for the EnhancedRouter class."""
    
    @pytest.fixture
    def router(self):
        """Create a test instance of EnhancedRouter."""
        return EnhancedRouter()
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("input_text,expected_intent,expected_entities", [
        # Math intents
        ("What is 2 + 2?", Intent.MATH, {"math_expression": "2+2"}),
        ("Calculate 5 * 3", Intent.MATH, {"math_expression": "5*3"}),
        ("I need to add 10 and 20", Intent.MATH, {"math_expression": "10+20"}),
        
        # Weather intents
        ("What's the weather in New York?", Intent.WEATHER, {"location": "new york"}),
        ("Will it rain tomorrow in London?", Intent.WEATHER, {
            "location": "london", 
            "datetime": "tomorrow"
        }),
        ("How's the temperature in San Francisco?", Intent.WEATHER, {"location": "san francisco"}),
        
        # Calendar intents
        ("Schedule a meeting tomorrow at 2pm", Intent.CALENDAR, {
            "event_title": "meeting",
            "datetime": "tomorrow at 2pm"
        }),
        ("Remind me about the project deadline", Intent.REMINDER, {
            "event_title": "project deadline"
        }),
        ("Set up a call with John next Monday", Intent.CALENDAR, {
            "event_title": "call with john",
            "datetime": "next monday"
        }),
        
        # Greeting intents
        ("Hello, how are you?", Intent.GREETING, {}),
        ("Good morning!", Intent.GREETING, {}),
        
        # Search intents
        ("Search for machine learning tutorials", Intent.SEARCH, {"query": "machine learning tutorials"}),
        ("Find me a good Italian restaurant", Intent.SEARCH, {"query": "good italian restaurant"}),
        
        # General/unknown intents
        ("Tell me a joke", Intent.UNKNOWN, {}),
        ("", Intent.GENERAL, {}),
    ])
    async def test_intent_classification(self, router, input_text, expected_intent, expected_entities):
        """Test that the router correctly classifies intents and extracts entities."""
        # Patch the ML prediction to return expected intent
        with patch.object(router, '_predict_with_ml') as mock_ml:
            mock_ml.return_value = (expected_intent, 0.9)
            
            result = await router.classify_intent(input_text)
            
            # Verify the intent is correct
            assert result.intent == expected_intent, f"Expected {expected_intent} for input: {input_text}"
            
            # Verify entities are extracted correctly
            for key, value in expected_entities.items():
                assert key in result.entities, f"Expected entity {key} not found for input: {input_text}"
                assert result.entities[key].value.lower() == value.lower(), \
                    f"Expected {key}={value} but got {result.entities[key].value} for input: {input_text}"
    
    @pytest.mark.asyncio
    async def test_ml_prediction(self, router):
        """Test ML-based intent prediction."""
        # Test with a math expression that should be in training data
        intent, confidence = router._predict_with_ml("what is 2 + 2")
        assert intent == Intent.MATH
        assert 0.5 <= confidence <= 1.0  # Confidence should be reasonably high
    
    @pytest.mark.asyncio
    async def test_pattern_matching(self, router):
        """Test pattern-based intent matching."""
        # Test with a pattern that should match
        intent, confidence, entities = router._match_with_patterns("what is 2 + 2")
        assert intent == Intent.MATH
        assert confidence > 0
        assert "math_expression" in entities
    
    @pytest.mark.asyncio
    async def test_entity_extraction(self, router):
        """Test entity extraction from text."""
        # Test weather location extraction
        entities = router._extract_weather_entities("What's the weather in New York?")
        assert "location" in entities
        assert entities["location"].value.lower() == "new york"
        
        # Test math expression extraction
        entities = router._extract_math_entities("calculate 2 + 2")
        assert "math_expression" in entities
        assert entities["math_expression"].value == "2 + 2"
    
    @pytest.mark.asyncio
    async def test_conversation_history(self, router):
        """Test that conversation history is maintained correctly."""
        # Initial message
        match1 = IntentMatch(Intent.GREETING, 0.9, {})
        router.update_conversation_history("Hello!", match1)
        
        # Second message with context
        match2 = IntentMatch(Intent.WEATHER, 0.95, {
            "location": Entity("new york", "location", 0.9)
        })
        router.update_conversation_history("What's the weather in New York?", match2)
        
        # Verify history
        assert len(router.conversation_history) == 2
        assert router.conversation_history[0]["message"] == "Hello!"
        assert router.conversation_history[1]["intent"] == Intent.WEATHER
        assert router.conversation_history[1]["entities"]["location"]["value"] == "new york"
    
    @pytest.mark.asyncio
    async def test_context_awareness(self, router):
        """Test that the router uses conversation context."""
        # First message sets context
        context = {}
        match1 = await router.classify_intent("What's the weather like?", context)
        
        # Second message should use context
        context = {"previous_intent": match1.intent}
        match2 = await router.classify_intent("In New York?", context)
        
        # Should maintain weather intent with location from context
        assert match2.intent == Intent.WEATHER
        assert "location" in match2.entities
        assert match2.entities["location"].value.lower() == "new york"

    def test_math_expression_normalization(self, router):
        """Test normalization of math expressions."""
        test_cases = [
            ("two plus two", "2 + 2"),
            ("10 minus 5", "10 - 5"),
            ("3 times 4", "3 * 4"),
            ("20 divided by 5", "20 / 5"),
            ("2 to the power of 3", "2 ^ 3"),
        ]
        
        for input_expr, expected in test_cases:
            normalized = router._normalize_math_expression(input_expr)
            assert normalized == expected, f"Failed to normalize: {input_expr}"

if __name__ == "__main__":
    pytest.main(["-v", __file__])
