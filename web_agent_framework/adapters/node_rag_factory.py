"""
Factory for DOM node RAG backends (Chroma primary, FAISS optional).
"""
from typing import Any

from ..config import NODE_RAG_BACKEND, NODE_RAG_CHROMA_COLLECTION
from .node_rag import ChromaNodeRAGAdapter, FaissNodeRAGAdapter


def get_node_rag_adapter(storage_dir: str, **kwargs: Any) -> Any:
    """
    Return a Node RAG adapter for indexing/querying DOM nodes per job.

    Config:
      NODE_RAG_BACKEND: chroma (default) | faiss
      NODE_RAG_CHROMA_COLLECTION: Chroma collection name (default dom_nodes)
    """
    backend = (kwargs.get("backend") or NODE_RAG_BACKEND or "chroma").lower()
    if backend == "faiss":
        return FaissNodeRAGAdapter(storage_dir=storage_dir)
    collection = kwargs.get("collection_name") or NODE_RAG_CHROMA_COLLECTION
    return ChromaNodeRAGAdapter(storage_dir=storage_dir, collection_name=collection)
