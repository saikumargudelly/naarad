# (This file is now obsolete; all Brave Search functionality is handled in agent/tools/brave_search.py)
# You may safely delete this file if not used elsewhere.

from typing import Optional, Type
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
import requests
import json
from ..llm.config import settings

class BraveSearchTool(BaseTool):
    name = "brave_search"
    description = "Useful for when you need to answer questions about current events or get real-time information from the web. Input should be a search query."
    
    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the Brave Search API to get search results."""
        headers = {
            "X-Subscription-Token": settings.brave_api_key,
            "Accept": "application/json",
        }
        
        params = {
            "q": query,
            "count": 5,  # Number of results to return
        }
        
        try:
            response = requests.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers=headers,
                params=params,
                timeout=10
            )
            response.raise_for_status()
            
            results = response.json()
            web_results = results.get("web", {}).get("results", [])
            
            if not web_results:
                return "No results found for the query."
                
            # Format the results
            formatted_results = []
            for i, result in enumerate(web_results, 1):
                formatted_results.append(
                    f"{i}. {result.get('title', 'No title')}\n"
                    f"   URL: {result.get('url', 'No URL')}\n"
                    f"   Description: {result.get('description', 'No description')}"
                )
            
            return "\n\n".join(formatted_results)
            
        except Exception as e:
            return f"Error performing search: {str(e)}"
    
    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Async version of the tool."""
        # For simplicity, we'll just call the sync version
        return self._run(query, run_manager)
