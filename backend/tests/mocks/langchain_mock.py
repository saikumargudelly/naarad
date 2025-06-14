"""Mock implementation of langchain for testing purposes."""
import sys

class MockLLM:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
    
    def __call__(self, *args, **kwargs):
        return "Mock LLM response"

class MockChatOpenAI:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.temperature = kwargs.get('temperature', 0.7)

class MockAgentExecutor:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
    
    async def ainvoke(self, *args, **kwargs):
        return {"output": "Mock agent response"}

class MockPromptTemplate:
    @classmethod
    def from_messages(cls, messages):
        return cls(messages)
    
    def __init__(self, messages):
        self.messages = messages

# Create mock modules
class MockModules:
    class agents:
        AgentExecutor = MockAgentExecutor
        
        @staticmethod
        def create_openai_tools_agent(*args, **kwargs):
            return "mock_agent"
    
    class llms:
        OpenAI = MockLLM
    
    class chat_models:
        ChatOpenAI = MockChatOpenAI
    
    class prompts:
        ChatPromptTemplate = MockPromptTemplate
        MessagesPlaceholder = object()

# Create the mock module
sys.modules['langchain'] = MockModules()
sys.modules['langchain.agents'] = MockModules.agents
sys.modules['langchain.llms'] = MockModules.llms
sys.modules['langchain.chat_models'] = MockModules.chat_models
sys.modules['langchain.prompts'] = MockModules.prompts
