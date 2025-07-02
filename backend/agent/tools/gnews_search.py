"""GNews Search tool for performing real-time news searches."""
from typing import Dict, Any
import httpx
from langchain_core.tools import BaseTool
from llm.config import settings

class GNewsSearchTool(BaseTool):
    """Tool for performing news searches using the GNews API."""
    name: str = "gnews_search"
    description: str = "Useful for answering questions about current news. Input should be a clear news query."

    def _run(self, query: str) -> dict:
        api_key = settings.GNEWS_API_KEY
        url = "https://gnews.io/api/v4/search"
        params = {
            "q": query,
            "token": api_key,
            "lang": "en",
            "max": 5
        }
        try:
            response = httpx.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    async def _arun(self, query: str) -> dict:
        return self._run(query) 