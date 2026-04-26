"""
LLM adapter factory. Returns the appropriate adapter based on LLM_PROVIDER config.
Only the selected provider's dependencies are loaded (e.g. ollama not imported if using gemini).
"""
from typing import Any

from ..config import LLM_PROVIDER, LLM_MODEL, GEMINI_API_KEY, GOOGLE_API_KEY


def get_llm_adapter(**kwargs: Any) -> Any:
    """
    Return an LLM adapter for the configured provider.

    Config:
      LLM_PROVIDER: ollama (default), gemini
      LLM_MODEL: model name (e.g. qwen3-coder:30b for Ollama, gemini-1.5-flash for Gemini)

    Returns an adapter implementing: generate(), chat(), generate_json()
    """
    provider = (LLM_PROVIDER or "ollama").lower()
    model = kwargs.get("model") or LLM_MODEL

    if provider == "gemini":
        api_key = GEMINI_API_KEY or GOOGLE_API_KEY
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY or GOOGLE_API_KEY required for LLM_PROVIDER=gemini. "
                "Set it in .env or environment."
            )
        from .gemini_adapter import GeminiAdapter
        return GeminiAdapter(api_key=api_key, model_name=model)

    if provider == "ollama":
        try:
            from .ollama_adapter import OllamaAdapter
        except ImportError as e:
            raise ImportError(
                "ollama package required for LLM_PROVIDER=ollama. Run: pip install ollama. "
                "Or use LLM_PROVIDER=gemini with GEMINI_API_KEY."
            ) from e
        return OllamaAdapter(model_name=model)

    raise ValueError(f"Unknown LLM_PROVIDER: {provider}. Use ollama or gemini.")
