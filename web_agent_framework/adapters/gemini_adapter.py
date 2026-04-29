import google.generativeai as genai
from typing import List, Dict, Any, Optional
import json
import os
import re

class GeminiAdapter:
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    async def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        # Gemini 1.5 supports system_instruction in constructor or as part of content
        if system_prompt:
            # For simplicity, we can prepend it or use a new model instance with system_instruction
            model = genai.GenerativeModel(
                model_name=self.model.model_name,
                system_instruction=system_prompt
            )
        else:
            model = self.model
            
        response = model.generate_content(prompt)
        return response.text

    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        # Convert messages to Gemini format
        history = []
        for msg in messages[:-1]:
            role = "user" if msg["role"] == "user" else "model"
            history.append({"role": role, "parts": [msg["content"]]})
        
        chat_session = self.model.start_chat(history=history)
        response = chat_session.send_message(messages[-1]["content"])
        return response.text

    async def generate_json(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        # Use response_mime_type="application/json" if supported by the model
        if system_prompt:
            model = genai.GenerativeModel(
                model_name=self.model.model_name,
                system_instruction=system_prompt
            )
        else:
            model = self.model
            
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(response_mime_type="application/json")
        )
        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            # Fallback to regex if JSON mode fails
            match = re.search(r"\{.*\}", response.text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise

    async def embed(self, text: str, **kwargs) -> List[float]:
        from ..config import EMBEDDING_MODEL
        # Use config or provide a stable default
        model = kwargs.get('model') or EMBEDDING_MODEL or 'models/gemini-embedding-001'
        task_type = kwargs.get('task_type', 'retrieval_document')
        res = genai.embed_content(model=model, content=text, task_type=task_type)
        return res['embedding']
