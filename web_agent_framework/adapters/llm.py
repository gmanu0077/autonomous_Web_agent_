from typing import Protocol, List, Dict, Any, Optional

class LLMAdapter(Protocol):
    async def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        ...

    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        ...

    async def generate_json(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        ...
