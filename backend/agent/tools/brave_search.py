"""Brave Search tool for performing web searches."""
from typing import Dict, Any, Optional, List
import requests
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from llm.config import settings
import httpx

class BraveSearchTool(BaseTool):
    """Tool for performing web searches using the Brave Search API."""
    
    name: str = "brave_search"
    description: str = """
    Useful for when you need to answer questions about current events, find recent information, or search for products.
    Input should be a clear and specific search query. For product searches, include brand names, models, and specific features.
    """
    
    def _run(self, query: str) -> dict:
        """Perform a web search using the Brave Search API.
        
        Args:
            query: The search query string.
            
        Returns:
            dict: Search results or an error message.
        """
        api_key = settings.BRAVE_API_KEY
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {"Accept": "application/json", "X-Subscription-Token": api_key}
        params = {"q": query}
        try:
            response = httpx.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    async def _arun(self, query: str) -> str:
        """Async version of _run."""
        return self._run(query)
