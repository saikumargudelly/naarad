"""Brave Search tool for performing web searches."""
from typing import Dict, Any, Optional, List
import requests
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from llm.config import settings

class BraveSearchTool(BaseTool):
    """Tool for performing web searches using the Brave Search API."""
    
    name: str = "brave_search"
    description: str = """
    Useful for when you need to answer questions about current events or find recent information.
    Input should be a search query.
    """
    
    def _run(self, query: str) -> str:
        """Perform a web search using the Brave Search API.
        
        Args:
            query: The search query string.
            
        Returns:
            str: Search results or an error message.
        """
        if not hasattr(settings, 'BRAVE_API_KEY') or not settings.BRAVE_API_KEY:
            return "Error: BRAVE_API_KEY is not configured in the settings."
            
        headers = {
            "X-Subscription-Token": settings.BRAVE_API_KEY,
            "Accept": "application/json",
        }
        
        try:
            response = requests.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers=headers,
                params={"q": query, "count": 5}  # Get top 5 results
            )
            response.raise_for_status()
            
            results = response.json()
            if 'web' in results and 'results' in results['web']:
                search_results = []
                for i, result in enumerate(results['web']['results'][:5], 1):
                    search_results.append(
                        f"{i}. {result.get('title', 'No title')}\n"
                        f"   {result.get('url', 'No URL')}\n"
                        f"   {result.get('description', 'No description')}\n"
                    )
                return "\n".join(search_results)
            return "No search results found."
            
        except requests.exceptions.RequestException as e:
            return f"Error performing search: {str(e)}"
    
    async def _arun(self, query: str) -> str:
        """Async version of _run."""
        return self._run(query)
