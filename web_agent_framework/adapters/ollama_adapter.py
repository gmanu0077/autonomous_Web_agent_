import ollama
import re
import json
from typing import List, Dict, Any, Optional

class OllamaAdapter:
    def __init__(self, model_name: str):
        self.model_name = model_name

    async def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # ollama.chat is synchronous in the library usually, 
        # but there is an async client or we can wrap it.
        # Based on pydoll_agent code, it seems they use it synchronously or expect it to be.
        # I'll use a thread pool or just call it if it's fast enough for now, 
        # but let's try to find if there's an async way.
        # Actually, ollama library does have an AsyncClient.
        
        response = ollama.chat(model=self.model_name, messages=messages, **kwargs)
        return response["message"]["content"]

    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        response = ollama.chat(model=self.model_name, messages=messages, **kwargs)
        return response["message"]["content"]

    async def generate_json(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        # Force JSON mode if supported or just parse
        response_text = await self.generate(prompt, system_prompt, **kwargs)
        try:
            # Clean up potential markdown code blocks
            clean_text = response_text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:]
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3]
            return json.loads(clean_text.strip())
        except json.JSONDecodeError:
            # Fallback: try to find anything that looks like JSON
            match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise
