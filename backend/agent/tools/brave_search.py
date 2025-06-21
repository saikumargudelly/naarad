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
    Useful for when you need to answer questions about current events, find recent information, or search for products.
    Input should be a clear and specific search query. For product searches, include brand names, models, and specific features.
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
            "Accept-Encoding": "gzip",
            "User-Agent": "NaaradAI/1.0"
        }
        
        try:
            params = {
                "q": query,
                "count": 5,  # Get top 5 results
                "result_filter": "web",
                "safesearch": "moderate"
            }
            
            response = requests.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers=headers,
                params=params,
                timeout=10  # 10 second timeout
            )
            response.raise_for_status()
            
            results = response.json()
            
            if not results.get('web', {}).get('results'):
                return "No search results found. Try adjusting your search terms."
                
            search_results = []
            for i, result in enumerate(results['web']['results'][:5], 1):
                title = result.get('title', 'No title')
                url = result.get('url', 'No URL')
                desc = result.get('description', 'No description available')
                
                search_results.append(
                    f"{i}. {title}\n"
                    f"   URL: {url}\n"
                    f"   {desc}\n"
                )
                
            return "\n\n".join(search_results)
            
        except requests.exceptions.Timeout:
            return "Error: Search request timed out. Please try again later."
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                return "Error: Invalid API key. Please check your Brave Search API configuration."
            elif e.response.status_code == 429:
                return "Error: Rate limit exceeded. Please wait before making more requests."
            else:
                return f"Error performing search: HTTP {e.response.status_code}"
        except requests.exceptions.RequestException as e:
            return f"Error performing search: {str(e)}"
        except Exception as e:
            return f"An unexpected error occurred: {str(e)}"
    
    async def _arun(self, query: str) -> str:
        """Async version of _run."""
        return self._run(query)
