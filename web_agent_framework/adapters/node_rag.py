import os
import json
from typing import List, Dict, Any, Protocol, Optional

import numpy as np

from .llm import LLMAdapter


def _embedding_error_is_fatal(err: BaseException) -> bool:
    """Stop bulk indexing on auth / revoked key errors (avoid thousands of identical log lines)."""
    s = str(err).lower()
    return any(
        x in s
        for x in (
            "403",
            "401",
            "leaked",
            "invalid api key",
            "api key not valid",
            "permission denied",
            "requested entity was not found",
        )
    )


def _node_to_embed_text(node: Dict[str, Any]) -> str:
    """Build text used for embedding a DOM node."""
    tag = node.get("tag", "")
    text = node.get("text", "")
    attrs = str(node.get("attrs", {}))
    hint = node.get("semantic_hint", "")
    return f"Tag: {tag}\nText: {text}\nAttrs: {attrs}\nHint: {hint}"


def _chroma_client(persist_path: str):
    import chromadb
    from chromadb.config import Settings

    try:
        return chromadb.PersistentClient(path=persist_path)
    except (TypeError, AttributeError):
        return chromadb.Client(Settings(persist_directory=persist_path, is_persistent=True))


class NodeRAGAdapter(Protocol):
    async def index_nodes(self, nodes: List[Dict[str, Any]], llm_adapter: LLMAdapter):
        ...

    async def query(self, query: str, llm_adapter: LLMAdapter, k: int = 5) -> List[Dict[str, Any]]:
        ...


class ChromaNodeRAGAdapter:
    """
    Persist node embeddings in ChromaDB (local per-job directory).
    Full node payloads are stored in nodes_by_id.json for retrieval after similarity search.
    """

    _BATCH = 128

    def __init__(self, storage_dir: str, collection_name: Optional[str] = None):
        from ..config import NODE_RAG_CHROMA_COLLECTION

        self.storage_dir = storage_dir
        self.collection_name = collection_name or NODE_RAG_CHROMA_COLLECTION
        os.makedirs(self.storage_dir, exist_ok=True)
        self.nodes_path = os.path.join(self.storage_dir, "nodes_by_id.json")

    async def index_nodes(self, nodes: List[Dict[str, Any]], llm_adapter: LLMAdapter):
        coll_name = self.collection_name
        print(f"[NodeRAG/Chroma] Indexing {len(nodes)} nodes into {self.storage_dir!r}...")

        client = _chroma_client(self.storage_dir)
        try:
            client.delete_collection(coll_name)
        except Exception:
            pass
        collection = client.create_collection(
            name=coll_name,
            metadata={"hnsw:space": "l2"},
        )

        nodes_by_id: Dict[str, Any] = {}
        batch_ids: List[str] = []
        batch_docs: List[str] = []
        batch_embs: List[List[float]] = []

        async def flush():
            nonlocal batch_ids, batch_docs, batch_embs
            if not batch_ids:
                return
            collection.add(
                ids=batch_ids,
                documents=batch_docs,
                embeddings=batch_embs,
            )
            batch_ids, batch_docs, batch_embs = [], [], []

        for node in nodes:
            nid = str(node.get("id", ""))
            if not nid:
                continue
            node_str = _node_to_embed_text(node)
            try:
                emb = await llm_adapter.embed(node_str)
            except Exception as e:
                if _embedding_error_is_fatal(e):
                    print(f"[NodeRAG/Chroma] Embedding stopped (API/auth): {e}")
                    print(
                        "  → If the key was leaked or revoked: create a new key at "
                        "https://aistudio.google.com/apikey and set GEMINI_API_KEY in .env (do not commit it)."
                    )
                    break
                print(f"[NodeRAG/Chroma] Embedding failed for node {nid}: {e}")
                continue
            nodes_by_id[nid] = node
            batch_ids.append(nid)
            batch_docs.append(node_str)
            batch_embs.append(emb)
            if len(batch_ids) >= self._BATCH:
                await flush()

        await flush()

        with open(self.nodes_path, "w", encoding="utf-8") as f:
            json.dump(nodes_by_id, f, indent=2, default=str)

        print(f"[NodeRAG/Chroma] Indexed {len(nodes_by_id)} nodes.")

    async def query(self, query: str, llm_adapter: LLMAdapter, k: int = 5) -> List[Dict[str, Any]]:
        coll_name = self.collection_name
        if not os.path.isfile(self.nodes_path):
            print("[NodeRAG/Chroma] nodes_by_id.json not found.")
            return []

        with open(self.nodes_path, "r", encoding="utf-8") as f:
            nodes_by_id = json.load(f)

        if not nodes_by_id:
            return []

        client = _chroma_client(self.storage_dir)
        try:
            collection = client.get_collection(coll_name)
        except Exception as e:
            print(f"[NodeRAG/Chroma] Collection missing: {e}")
            return []

        try:
            q_emb = await llm_adapter.embed(query, task_type="retrieval_query")
        except Exception:
            q_emb = await llm_adapter.embed(query)

        results = collection.query(
            query_embeddings=[q_emb],
            n_results=min(k, max(1, len(nodes_by_id))),
            include=["distances"],
        )
        ids = (results.get("ids") or [[]])[0]
        out: List[Dict[str, Any]] = []
        for nid in ids:
            if nid in nodes_by_id:
                out.append(nodes_by_id[nid])
        return out


class FaissNodeRAGAdapter:
    def __init__(self, storage_dir: str):
        self.storage_dir = storage_dir
        self.index = None
        self.node_data = [] # List of node dicts corresponding to index
        
        os.makedirs(self.storage_dir, exist_ok=True)
        self.index_path = os.path.join(self.storage_dir, "faiss_index.bin")
        self.data_path = os.path.join(self.storage_dir, "node_data.json")

    async def index_nodes(self, nodes: List[Dict[str, Any]], llm_adapter: LLMAdapter):
        """Index nodes into FAISS."""
        import faiss
        
        print(f"[NodeRAG] Indexing {len(nodes)} nodes into FAISS...")
        embeddings = []
        valid_nodes = []
        
        for node in nodes:
            node_str = _node_to_embed_text(node)
            
            try:
                emb = await llm_adapter.embed(node_str)
                embeddings.append(emb)
                valid_nodes.append(node)
            except Exception as e:
                if _embedding_error_is_fatal(e):
                    print(f"[NodeRAG/FAISS] Embedding stopped (API/auth): {e}")
                    print(
                        "  → Create a new Gemini key at https://aistudio.google.com/apikey "
                        "or use LLM_PROVIDER=ollama with a local embedding model."
                    )
                    break
                print(f"[NodeRAG/FAISS] Embedding failed for node {node.get('id')}: {e}")
                continue
                
        if not embeddings:
            print("[NodeRAG] No embeddings generated.")
            return

        emb_array = np.array(embeddings).astype('float32')
        dim = emb_array.shape[1]
        
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(emb_array)
        self.node_data = valid_nodes
        
        # Save to disk
        faiss.write_index(self.index, self.index_path)
        with open(self.data_path, "w", encoding="utf-8") as f:
            json.dump(self.node_data, f, indent=2)
        print(f"[NodeRAG] Successfully indexed and saved {len(self.node_data)} nodes.")

    async def query(self, query: str, llm_adapter: LLMAdapter, k: int = 5) -> List[Dict[str, Any]]:
        """Query FAISS for top-k nodes."""
        import faiss
        
        if self.index is None:
            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
                with open(self.data_path, "r", encoding="utf-8") as f:
                    self.node_data = json.load(f)
            else:
                print("[NodeRAG] Index not found.")
                return []
        
        q_emb = await llm_adapter.embed(query, task_type='retrieval_query')
        q_emb_array = np.array([q_emb]).astype('float32')
        
        distances, indices = self.index.search(q_emb_array, k)
        
        results = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.node_data):
                results.append(self.node_data[idx])
                
        return results
