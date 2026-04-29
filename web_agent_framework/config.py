"""
Configuration for the 3-Phase Autonomous Web Agent.

Chroma DB: Used for (1) Pydoll/Selenium code RAG (DB_DIR / COLLECTION_NAME) and
(2) per-job DOM node embeddings under each job's node_rag/ folder (NODE_RAG_CHROMA_COLLECTION).

MCP Server: Optional. When MCP_SERVER_URL is set, browser control uses MCP (Selenium/Pydoll via MCP).
"""
import os
from pathlib import Path

# Paths
WORKSPACE_ROOT = Path(__file__).parent.parent.parent.absolute()
DEFAULT_DB_DIR = r"D:\autonomous web agent\chroma_db_pydoll"
DEFAULT_TEMP_ROOT = str(WORKSPACE_ROOT / "temp_pydoll")

# Chroma DB — used by RAG for Pydoll/Selenium code embeddings (planner, step generator)
DB_DIR = os.getenv("DB_DIR", DEFAULT_DB_DIR)
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pydoll_code")

# MCP Browser — which MCP server to use. Default: pydoll. Options: pydoll, selenium
# Our scripts run the selected MCP server; agent connects to it via stdio.
MCP_BROWSER = os.getenv("MCP_BROWSER", "pydoll").lower()
if MCP_BROWSER not in ("pydoll", "selenium"):
    MCP_BROWSER = "pydoll"

# MCP_SERVER_URL — optional override. When set, connect to this external MCP server
# instead of spawning our own. Example: "stdio:npx -y @modelcontextprotocol/server-browser"
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "")

# ---------------------------------------------------------------------------
# LLM — choose provider and model. Edit here or set env vars (env overrides).
# ---------------------------------------------------------------------------
# Provider: "ollama" (local) or "gemini" (API)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
if LLM_PROVIDER not in ("ollama", "gemini"):
    LLM_PROVIDER = "ollama"

# Model name per provider (env LLM_MODEL overrides):
#   ollama:  qwen3-coder:30b, llama3.2, mistral, etc. (run: ollama pull <model>)
#   gemini:  gemini-1.5-flash, gemini-1.5-pro, gemini-2.0-flash, etc.
_DEFAULT_MODEL = {"ollama": "qwen3-coder:30b", "gemini": "gemini-flash-latest"}
LLM_MODEL = os.getenv("LLM_MODEL", _DEFAULT_MODEL.get(LLM_PROVIDER, "qwen3-coder:30b"))

# Gemini API key — set via environment or .env only (never commit real keys; leaked keys are revoked by Google).
# https://aistudio.google.com/apikey
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", GEMINI_API_KEY)

# Embedding model name:
#   ollama:  nomic-embed-text (default)
#   gemini:  gemini-embedding-001
_DEFAULT_EMBED = {"ollama": "nomic-embed-text", "gemini": "models/gemini-embedding-001"}
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", _DEFAULT_EMBED.get(LLM_PROVIDER, "nomic-embed-text"))
CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME", LLM_MODEL)
PLANNER_MODEL = os.getenv("PLANNER_MODEL", LLM_MODEL)

MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "425"))
MAX_HTML_CONTEXT_CHARS = int(os.getenv("MAX_HTML_CONTEXT_CHARS", "108000"))
TOP_K = int(os.getenv("TOP_K", "5"))

TEMP_ROOT = os.getenv("TEMP_ROOT", DEFAULT_TEMP_ROOT)
MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", "3"))

# Node RAG (DOM node embeddings per job): chroma (default) or faiss
NODE_RAG_BACKEND = os.getenv("NODE_RAG_BACKEND", "chroma").lower()
if NODE_RAG_BACKEND not in ("chroma", "faiss"):
    NODE_RAG_BACKEND = "chroma"
# Collection name for Chroma node index (stored under each job's node_rag/ folder)
NODE_RAG_CHROMA_COLLECTION = os.getenv("NODE_RAG_CHROMA_COLLECTION", "dom_nodes")

HTML_CHUNK_SIZE = int(os.getenv("HTML_CHUNK_SIZE", "1024"))
HTML_CHUNK_OVERLAP = int(os.getenv("HTML_CHUNK_OVERLAP", "128"))

