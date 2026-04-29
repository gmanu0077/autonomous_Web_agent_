import os
from typing import List, Dict, Any, Protocol
import chromadb
from chromadb.config import Settings
from ..config import (
    DB_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    TOP_K,
    MAX_CONTEXT_CHARS,
)

class RAGAdapter(Protocol):
    def get_top_chunks(self, query: str, n_results: int = TOP_K) -> List[Dict[str, Any]]:
        ...

    def build_context_from_chunks(self, chunks: List[Dict[str, Any]], max_chars: int = MAX_CONTEXT_CHARS) -> str:
        ...

class ChromaOllamaRAG:
    def __init__(self):
        self.db_dir = DB_DIR
        self.collection_name = COLLECTION_NAME
        self.embedding_model = EMBEDDING_MODEL

    def embed_text(self, text: str) -> List[float]:
        from .llm_factory import get_llm_adapter
        import asyncio
        llm = get_llm_adapter()
        # ChromaOllamaRAG is synchronous, but llm.embed is async
        return asyncio.run(llm.embed(text))

    def get_top_chunks(self, query: str, n_results: int = TOP_K) -> List[Dict[str, Any]]:
        if not os.path.isdir(self.db_dir):
            return []

        try:
            try:
                client = chromadb.PersistentClient(path=self.db_dir)
            except (TypeError, AttributeError):
                client = chromadb.Client(
                    Settings(persist_directory=self.db_dir, is_persistent=True)
                )
            collection = client.get_collection(self.collection_name)
            q_emb = self.embed_text(query)
            results = collection.query(
                query_embeddings=[q_emb],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            print(f"[RAG] Chroma/Ollama query failed: {e}")
            return []

        if not results.get("ids") or not results["ids"][0]:
            return []

        chunks: List[Dict[str, Any]] = []
        for i in range(len(results["ids"][0])):
            doc_text = results["documents"][0][i]
            meta = results["metadatas"][0][i] or {}
            dist = results["distances"][0][i]

            chunks.append({
                "path": meta.get("path", "UNKNOWN"),
                "start_offset": meta.get("start_offset", 0),
                "text": doc_text,
                "distance": dist,
            })
        return chunks

    def build_context_from_chunks(self, chunks: List[Dict[str, Any]], max_chars: int = MAX_CONTEXT_CHARS) -> str:
        parts: List[str] = []
        total = 0

        for ch in chunks:
            path = ch.get("path", "UNKNOWN")
            start_offset = ch.get("start_offset", 0)
            distance = ch.get("distance", 0.0)

            header = f"\n---\nFILE: {path} (offset {start_offset}, distance {distance:.4f})\n"
            body = ch.get("text", "")
            piece = header + body

            if total + len(piece) > max_chars:
                remaining = max_chars - total
                if remaining > 0:
                    parts.append(piece[:remaining])
                    total += remaining
                break

            parts.append(piece)
            total += len(piece)

        return "".join(parts)
