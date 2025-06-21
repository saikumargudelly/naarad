"""A dummy tool that does nothing but return a message."""
from typing import Optional, Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

class DummyToolInput(BaseModel):
    """Input for the DummyTool."""
    query: str = Field(..., description="The query to process")

class DummyTool(BaseTool):
    """A dummy tool that does nothing but return a message.
    
    This tool is used as a placeholder when no other tools are available.
    """
    name: str = "dummy_tool"
    description: str = "A dummy tool that does nothing but return a message."
    args_schema: Type[BaseModel] = DummyToolInput
    
    def _run(self, query: str, **kwargs) -> str:
        """Run the dummy tool.
        
        Args:
            query: The query to process
            
        Returns:
            A message indicating this is a dummy tool
        """
        return "This is a dummy tool. No actual processing was done."
    
    async def _arun(self, query: str, **kwargs) -> str:
        """Run the dummy tool asynchronously.
        
        Args:
            query: The query to process
            
        Returns:
            A message indicating this is a dummy tool
        """
        return self._run(query, **kwargs)
