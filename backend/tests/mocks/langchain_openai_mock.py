"""Mock implementation of langchain_openai for testing purposes."""

class MockChatOpenAI:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.temperature = kwargs.get('temperature', 0.7)
        self.model_name = kwargs.get('model_name', 'gpt-3.5-turbo')
    
    async def ainvoke(self, *args, **kwargs):
        return "Mocked response from ChatOpenAI"

# Create a mock module
class MockLangchainOpenAIModule:
    ChatOpenAI = MockChatOpenAI

# Create the mock module in sys.modules
import sys
sys.modules['langchain_openai'] = MockLangchainOpenAIModule()
