from typing import Optional, Dict, Any
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
import requests
import base64
import os
from ..llm.config import settings

class LLaVAVisionTool(BaseTool):
    name = "image_understanding"
    description = "Useful when you need to understand or analyze images. Input should be a JSON with 'image_url' (URL of the image) and 'prompt' (what you want to know about the image)."
    
    def _run(
        self, input_json: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the LLaVA model to understand images."""
        try:
            # Parse the input JSON
            try:
                input_data = json.loads(input_json)
                image_url = input_data.get('image_url')
                prompt = input_data.get('prompt', 'What is in this image?')
            except json.JSONDecodeError:
                return "Invalid input format. Please provide a JSON with 'image_url' and 'prompt'."
            
            if not image_url:
                return "No image URL provided. Please provide a valid image URL."
            
            # Prepare the API request
            headers = {
                "Authorization": f"Bearer {settings.together_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": settings.vision_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_url}}
                        ]
                    }
                ],
                "max_tokens": 1024
            }
            
            # Make the API call
            response = requests.post(
                f"{settings.together_base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            # Extract and return the response
            result = response.json()
            return result['choices'][0]['message']['content']
            
        except Exception as e:
            return f"Error processing image: {str(e)}"
    
    async def _arun(
        self, input_json: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Async version of the tool."""
        return self._run(input_json, run_manager)
