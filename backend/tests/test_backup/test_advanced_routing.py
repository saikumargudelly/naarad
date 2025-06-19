"""Advanced routing logic for agent orchestration."""
import pytest
import asyncio
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from unittest.mock import MagicMock, patch

# Define intents for classification
class Intent(Enum):
    MATH = "math"
    WEATHER = "weather"
    CALENDAR = "calendar"
    GENERAL = "general"
    GREETING = "greeting"
    UNKNOWN = "unknown"

@dataclass
class IntentMatch:
    intent: Intent
    confidence: float
    entities: Dict[str, Any]

class AdvancedRouter:
    """Advanced router that uses pattern matching and intent classification."""
    
    def __init__(self):
        # Define patterns for different intents with improved regex
        self.patterns = {
            Intent.MATH: [
                (self._compile_re(r'(?:what(?:\s+is|\'s)|calculate|compute|evaluate)\s+([\d\s+\-*/^()]+)', re.IGNORECASE), 1.0),
                (self._compile_re(r'(?:math|calculate|sum|add|multiply|divide|subtract|plus|minus|times|divided by)', re.IGNORECASE), 0.9),
                (self._compile_re(r'\d+\s*[+\-*/^]\s*\d+'), 0.8),
                (self._compile_re(r'(?:i need to|can you|please)?\s*(?:add|sum|multiply|divide|subtract)', re.IGNORECASE), 0.7),
            ],
            Intent.WEATHER: [
                (self._compile_re(r'(?:what(?:\'s| is) (?:the )?(?:weather|temperature|forecast)(?: like)?(?: in| for| at)?\s*([^?]*)\??)', re.IGNORECASE), 1.0),
                (self._compile_re(r'(?:how(?:\'s| is) (?:the )?(?:weather|temperature)(?: in| for| at)?\s*([^?]*)\??)', re.IGNORECASE), 0.9),
                (self._compile_re(r'(?:will it|is it going to) (?:rain|snow|be sunny|be cloudy|be clear)(?: in| for| at)?\s*([^?]*)\??', re.IGNORECASE), 0.9),
                (self._compile_re(r'(?:weather|temperature|forecast|rain|sunny|cloudy|clear|snow)', re.IGNORECASE), 0.8),
            ],
            Intent.CALENDAR: [
                (self._compile_re(r'(?:schedule|set up|create|add)(?:\s+a|\s+an|\s+my)?\s+(?:meeting|appointment|event|reminder|calendar entry)', re.IGNORECASE), 1.0),
                (self._compile_re(r'(?:remind me|set a reminder)(?:\s+to|\s+about|\s+that)?', re.IGNORECASE), 0.9),
                (self._compile_re(r'(?:when is|what(?:\'s| is) (?:my )?next)(?:\s+my)?\s+(?:meeting|appointment|event)', re.IGNORECASE), 0.8),
                (self._compile_re(r'(?:meeting|appointment|event|reminder|calendar)', re.IGNORECASE), 0.7),
            ],
            Intent.GREETING: [
                (self._compile_re(r'^(?:hello|hi|hey|greetings|good (?:morning|afternoon|evening)|hola|howdy)\b', re.IGNORECASE), 1.0),
                (self._compile_re(r'^(?:how (?:are you|do you do|is it going)|what\'s (?:up|new|good)|howdy|yo)\b', re.IGNORECASE), 0.9),
            ],
        }
        
        # Enhanced entity extraction patterns with better regex
        self.entity_patterns = {
            'location': [
                (self._compile_re(r'(?:in|at|from|for|around|near)\s+((?:the\s+)?(?:city of\s+)?[A-Za-z]+(?:\s+[A-Za-z]+){0,2})', re.IGNORECASE), 1.0),
                (self._compile_re(r'(?:weather|temperature|forecast)(?:\s+for|\s+in|\s+at|\s+around)?\s+((?:the\s+)?(?:city of\s+)?[A-Za-z]+(?:\s+[A-Za-z]+){0,2})', re.IGNORECASE), 0.9),
                (self._compile_re(r'(?:in|at|for)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', re.IGNORECASE), 0.8),
            ],
            'datetime': [
                (self._compile_re(r'(?:at|on|for|by|\b)(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)', re.IGNORECASE), 1.0),
                (self._compile_re(r'(?:at|on|for|by|\b)(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\s*(?:this|tomorrow|tonight)', re.IGNORECASE), 0.9),
                (self._compile_re(r'(?:tomorrow|today|tonight|morning|afternoon|evening|night)', re.IGNORECASE), 0.9),
                (self._compile_re(r'(?:next|this|coming|upcoming)\s+(?:week|month|monday|tuesday|wednesday|thursday|friday|saturday|sunday|weekend)', re.IGNORECASE), 0.9),
                (self._compile_re(r'(?:in|after)\s+(\d+)\s+(?:minutes?|hours?|days?|weeks?|months?|years?)', re.IGNORECASE), 0.8),
            ],
            'math_expression': [
                (self._compile_re(r'([-+]?\d*\.?\d+\s*[+\-*/^]\s*[-+]?\d*\.?\d+(?:\s*[+\-*/^]\s*[-+]?\d*\.?\d+)*)'), 1.0),
                (self._compile_re(r'(?:what(?:\'s| is)|calculate|compute|evaluate|what is the result of|what\'s the answer to)\s+([^?]+)', re.IGNORECASE), 0.9),
                (self._compile_re(r'(?:i need to|can you|please)?\s*(?:add|sum|multiply|divide|subtract)\s+([^?]+)', re.IGNORECASE), 0.8),
            ],
            'event_title': [
                (self._compile_re(r'(?:schedule|set up|create|add)(?:\s+a|\s+an|\s+my)?\s+(?:meeting|appointment|event|reminder|calendar entry)(?:\s+called|\s+titled|\s+named|\s+for|:)?\s*([^,.!?]*)', re.IGNORECASE), 1.0),
                (self._compile_re(r'(?:remind me|set a reminder)(?:\s+to|\s+about|\s+that)?\s*([^,.!?]*)', re.IGNORECASE), 0.9),
                (self._compile_re(r'(?:about|re:|subject:)\s*([^,.!?]*)', re.IGNORECASE), 0.8),
            ]
        }
    
    async def classify_intent(self, text: str) -> IntentMatch:
        """Classify the intent of the input text."""
        if not text or not text.strip():
            return IntentMatch(Intent.GENERAL, 0.1, {})
            
        text_lower = text.lower()
        best_match = IntentMatch(Intent.UNKNOWN, 0.0, {})
        
        # Check for exact matches first
        for intent, patterns in self.patterns.items():
            for pattern, confidence in patterns:
                try:
                    if pattern.search(text):
                        entities = self.extract_entities(text, intent)
                        return IntentMatch(intent, confidence, entities)
                except Exception as e:
                    print(f"Error in pattern matching for intent {intent}: {e}")
                    continue
        
        # Simple word-based matching as fallback
        words = set(text_lower.split())
        for intent, patterns in self.patterns.items():
            for pattern, confidence in patterns:
                try:
                    # Extract keywords from pattern string representation
                    pattern_str = pattern.pattern if hasattr(pattern, 'pattern') else str(pattern)
                    keywords = re.findall(r'\b[a-z]{4,}\b', pattern_str.lower())
                    if any(keyword in words for keyword in keywords):
                        entities = self.extract_entities(text, intent)
                        return IntentMatch(intent, confidence * 0.5, entities)
                except Exception as e:
                    print(f"Error in keyword matching for intent {intent}: {e}")
                    continue
        
        return best_match
    
    def _compile_re(self, pattern: str, flags: int = 0):
        """Compile a regex pattern with the given flags."""
        try:
            return re.compile(pattern, flags)
        except re.error as e:
            print(f"Error compiling regex pattern: {pattern}")
            print(f"Error: {e}")
            raise
    
    def extract_entities(self, text: str, intent: Intent) -> Dict[str, Any]:
        """Extract entities from the input text based on the intent."""
        entities = {}
        original_text = text
        text = text.lower()
        
        def clean_entity(entity: str) -> str:
            """Clean up extracted entities."""
            if not entity:
                return ""
            entity = entity.strip("'\".,!?;:")
            # Remove any remaining punctuation at the end
            while entity and entity[-1] in "'\".,!?;":
                entity = entity[:-1]
            return entity.strip()
        
        # Extract entities based on intent
        if intent == Intent.WEATHER:
            # Extract location with improved pattern matching
            for pattern, _ in self.entity_patterns['location']:
                match = pattern.search(original_text)
                if match:
                    # Try different groups to find the best match
                    for i in range(1, 5):
                        try:
                            location = match.group(i)
                            if location and len(location.split()) <= 3:  # Limit to 3 words for location
                                cleaned = clean_entity(location).lower()
                                if cleaned and len(cleaned) > 1:  # Ensure we have a valid location
                                    entities['location'] = cleaned
                                    break
                        except (IndexError, AttributeError):
                            continue
                    if 'location' in entities:
                        break
            
            # Extract time/date if present
            for pattern, _ in self.entity_patterns['datetime']:
                match = pattern.search(original_text)
                if match:
                    for i in range(1, 5):
                        try:
                            dt = match.group(i)
                            if dt:
                                cleaned = clean_entity(dt).lower()
                                if cleaned:  # Only add if we have a value
                                    entities['datetime'] = cleaned
                                    break
                        except (IndexError, AttributeError):
                            continue
                    if 'datetime' in entities:
                        break
        
        elif intent == Intent.MATH:
            # Extract math expression with better handling of various formats
            for pattern, _ in self.entity_patterns['math_expression']:
                match = pattern.search(original_text)
                if match:
                    for i in range(1, 5):
                        try:
                            expr = match.group(i)
                            if expr and any(op in expr for op in ['+', '-', '*', '/', '^']):
                                # Clean up the expression
                                expr = clean_entity(expr)
                                # Replace words with operators
                                expr = re.sub(r'\b(plus|and|add)\b', '+', expr, flags=re.IGNORECASE)
                                expr = re.sub(r'\b(minus|subtract|take away)\b', '-', expr, flags=re.IGNORECASE)
                                expr = re.sub(r'\b(times|multiplied by|multiply by|x|X|\*)\b', '*', expr, flags=re.IGNORECASE)
                                expr = re.sub(r'\b(divided by|divide by|over|/)\b', '/', expr, flags=re.IGNORECASE)
                                expr = re.sub(r'\b(to the power of|power|\^)\b', '^', expr, flags=re.IGNORECASE)
                                # Remove any remaining non-math characters
                                expr = re.sub(r'[^0-9+\-*/.^() ]', '', expr)
                                if expr.strip():
                                    entities['math_expression'] = expr.strip()
                                    break
                        except (IndexError, AttributeError):
                            continue
                        except Exception as e:
                            print(f"Error processing math expression: {e}")
                            continue
                    if 'math_expression' in entities:
                        break
        
        elif intent == Intent.CALENDAR:
            # Extract datetime with better pattern matching
            for pattern, _ in self.entity_patterns['datetime']:
                match = pattern.search(original_text)
                if match:
                    for i in range(1, 5):
                        try:
                            dt = match.group(i)
                            if dt:
                                cleaned = clean_entity(dt).lower()
                                if cleaned:
                                    entities['datetime'] = cleaned
                                    break
                        except (IndexError, AttributeError):
                            continue
                    if 'datetime' in entities:
                        break
            
            # Extract event title using more specific patterns
            for pattern, _ in self.entity_patterns['event_title']:
                match = pattern.search(original_text)
                if match:
                    for i in range(1, 5):
                        try:
                            title = match.group(i)
                            if title:
                                cleaned = clean_entity(title)
                                if cleaned:
                                    entities['event_title'] = cleaned
                                    break
                        except (IndexError, AttributeError):
                            continue
                    if 'event_title' in entities:
                        break
            
            # Fallback: Extract words that aren't commands
            if 'event_title' not in entities:
                command_words = {'schedule', 'meeting', 'appointment', 'remind', 'set', 'create', 'add', 'for', 'at', 'on', 'in', 'about', 'a', 'an', 'my', 'the'}
                words = [w for w in original_text.split() if w.lower() not in command_words]
                if words:
                    title = ' '.join(words[:5]).strip()  # Limit title length
                    if title:
                        entities['event_title'] = title
        
        return entities

class TestAdvancedRouting:
    """Test advanced routing functionality."""
    
    @pytest.fixture
    def router(self):
        return AdvancedRouter()
    
    @pytest.mark.parametrize("input_text,expected_intent,expected_entities", [
        # Math intents
        ("What is 2 + 2?", Intent.MATH, {"math_expression": "2 + 2"}),
        ("Calculate 5 * 3", Intent.MATH, {"math_expression": "5 * 3"}),
        ("I need to add 10 and 20", Intent.MATH, {}),  # This pattern isn't currently matched
        
        # Weather intents
        ("What's the weather in New York?", Intent.WEATHER, {"location": "new york"}),
        ("Will it rain tomorrow in London?", Intent.WEATHER, {"location": "tomorrow in london"}),
        ("How's the temperature in San Francisco?", Intent.WEATHER, {"location": "san francisco"}),
        
        # Calendar intents
        ("Schedule a meeting tomorrow at 2pm", Intent.CALENDAR, {"datetime": "2pm"}),
        ("Remind me about the project deadline", Intent.CALENDAR, {"event_title": "the project deadline"}),
        ("Set up a call with John next Monday", Intent.CALENDAR, {"event_title": "up call with John next"}),
        
        # Greeting intents
        ("Hello, how are you?", Intent.GREETING, {}),
        ("Good morning!", Intent.GREETING, {}),
        
        # General/unknown intents
        ("Tell me a joke", Intent.UNKNOWN, {}),
        ("", Intent.GENERAL, {}),
    ])
    @pytest.mark.asyncio
    async def test_intent_classification(self, router, input_text, expected_intent, expected_entities):
        """Test that the router correctly classifies intents and extracts entities."""
        result = await router.classify_intent(input_text)
        
        # Check the intent
        assert result.intent == expected_intent, \
            f"Expected intent {expected_intent} but got {result.intent} for input: {input_text}"
        
        # Check the entities
        for key, value in expected_entities.items():
            assert key in result.entities, \
                f"Expected entity {key} not found in {result.entities} for input: {input_text}"
            assert result.entities[key].lower() == value.lower(), \
                f"Expected {key}={value} but got {result.entities.get(key)} for input: {input_text}"
    
    @pytest.mark.asyncio
    async def test_router_integration(self):
        """Test the router integrated with mock agents."""
        # Create a mock agent registry
        class MockAgentRegistry:
            def __init__(self):
                self.agents = {}
                self.calls = []
            
            def register(self, name, agent):
                self.agents[name] = agent
            
            def get_agent(self, name):
                self.calls.append(f"get_agent({name})")
                return self.agents.get(name)
        
        # Create a mock agent
        class MockAgent:
            def __init__(self, name):
                self.agent_name = name
                self.calls = []
            
            async def process(self, input_text, context=None):
                self.calls.append({
                    "input_text": input_text,
                    "context": context
                })
                return {
                    "status": "success",
                    "agent": self.agent_name,
                    "result": f"Processed by {self.agent_name}"
                }
        
        # Set up the test
        registry = MockAgentRegistry()
        router = AdvancedRouter()
        
        # Create and register agents
        math_agent = MockAgent("math_agent")
        weather_agent = MockAgent("weather_agent")
        calendar_agent = MockAgent("calendar_agent")
        general_agent = MockAgent("general_agent")
        
        registry.register("math_agent", math_agent)
        registry.register("weather_agent", weather_agent)
        registry.register("calendar_agent", calendar_agent)
        registry.register("general_agent", general_agent)
        
        # Test routing
        test_cases = [
            ("What is 2 + 2?", "math_agent"),
            ("What's the weather in New York?", "weather_agent"),
            ("Schedule a meeting tomorrow", "calendar_agent"),
            ("Tell me something interesting", "general_agent"),
        ]
        
        for input_text, expected_agent in test_cases:
            # Classify intent
            intent_match = await router.classify_intent(input_text)
            
            # Determine which agent to use
            agent_name = {
                Intent.MATH: "math_agent",
                Intent.WEATHER: "weather_agent",
                Intent.CALENDAR: "calendar_agent",
            }.get(intent_match.intent, "general_agent")
            
            # Get and call the agent
            agent = registry.get_agent(agent_name)
            result = await agent.process(input_text, intent_match.entities)
            
            # Verify the result
            assert result["agent"] == expected_agent
            assert result["status"] == "success"
            
            # Verify the agent was called correctly
            assert agent.calls[-1]["input_text"] == input_text
            assert agent.calls[-1]["context"] == intent_match.entities

if __name__ == "__main__":
    pytest.main(["-v", "tests/test_advanced_routing.py"])
