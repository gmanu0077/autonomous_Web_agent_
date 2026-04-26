"""
Create embeddings from Pydoll/Selenium source files and store them in Chroma DB.

Run this script before using the agent to populate the RAG context for the planner
and step generator. Chroma DB location is configured in config.py (DB_DIR).

Usage:
  python -m web_agent_framework.build_embeddings [--source-dir PATH] [--chunk-size N]

Environment:
  DB_DIR          — Chroma DB persist directory (default: <workspace>/chroma_db_pydoll)
  COLLECTION_NAME — Collection name (default: pydoll_code)
  EMBEDDING_MODEL — Ollama embedding model (default: nomic-embed-text)
"""

import os
import sys
import argparse
from pathlib import Path

# Ensure workspace and src are on path when run as script
_SCRIPT_DIR = Path(__file__).parent.resolve()
_WORKSPACE = _SCRIPT_DIR.parent.parent
for p in (str(_WORKSPACE), str(_WORKSPACE / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from web_agent_framework.config import DB_DIR, COLLECTION_NAME, EMBEDDING_MODEL
from web_agent_framework.adapters.rag import ChromaOllamaRAG


# Default source dirs: Pydoll agent, framework
DEFAULT_SOURCE_DIRS = [
    _WORKSPACE / "pydoll_agent",
    _WORKSPACE / "src" / "web_agent_framework",
]
# Include root-level scripts (graph_scraper, universal_scraper)
for f in ["graph_scraper.py", "universal_scraper (1).py"]:
    p = _WORKSPACE / f
    if p.exists():
        DEFAULT_SOURCE_DIRS.append(p)


def chunk_text(text: str, chunk_size: int = 1024, overlap: int = 128) -> list[tuple[int, str]]:
    """Split text into overlapping chunks. Returns [(start_offset, chunk), ...]."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        if chunk.strip():
            chunks.append((start, chunk))
        start = end - overlap
        if start >= len(text):
            break
    return chunks


def collect_files(source_dirs: list[Path], extensions: tuple = (".py", ".md", ".txt")) -> list[tuple[Path, str]]:
    """Collect (path, content) for all matching files."""
    results = []
    for base in source_dirs:
        if not base.exists():
            print(f"[WARN] Source dir not found: {base}")
            continue
        if base.is_file():
            try:
                content = base.read_text(encoding="utf-8", errors="ignore")
                results.append((base, content))
            except Exception as e:
                print(f"[WARN] Could not read {base}: {e}")
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix.lower() in extensions:
                try:
                    content = path.read_text(encoding="utf-8", errors="ignore")
                    if content.strip():
                        results.append((path, content))
                except Exception as e:
                    print(f"[WARN] Could not read {path}: {e}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Build Chroma DB embeddings for Pydoll/Selenium RAG")
    parser.add_argument(
        "--source-dir",
        action="append",
        dest="source_dirs",
        help="Source directory to index (can repeat). Default: pydoll_agent, src/web_agent_framework",
    )
    parser.add_argument("--chunk-size", type=int, default=1024, help="Chunk size in chars")
    parser.add_argument("--overlap", type=int, default=128, help="Overlap between chunks")
    parser.add_argument("--db-dir", default=DB_DIR, help=f"Chroma DB directory (default: {DB_DIR})")
    parser.add_argument("--collection", default=COLLECTION_NAME, help=f"Collection name (default: {COLLECTION_NAME})")
    args = parser.parse_args()

    source_dirs = [Path(d).resolve() for d in (args.source_dirs or [])]
    if not source_dirs:
        source_dirs = [p.resolve() for p in DEFAULT_SOURCE_DIRS if p.exists()]

    if not source_dirs:
        print("No source directories found. Use --source-dir to specify paths.")
        sys.exit(1)

    print(f"Chroma DB: {args.db_dir}")
    print(f"Collection: {args.collection}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Source dirs: {source_dirs}")
    print()

    # Collect files
    files = collect_files(source_dirs)
    print(f"Collected {len(files)} files")

    # Build chunks
    all_docs = []
    all_metadatas = []
    all_ids = []
    idx = 0

    for path, content in files:
        rel_path = path.relative_to(_WORKSPACE) if _WORKSPACE in path.parents else path.name
        for start_offset, chunk in chunk_text(content, args.chunk_size, args.overlap):
            all_docs.append(chunk)
            all_metadatas.append({"path": str(rel_path), "start_offset": start_offset})
            all_ids.append(f"chunk_{idx}")
            idx += 1

    print(f"Created {len(all_docs)} chunks")

    if not all_docs:
        print("No chunks to embed. Exiting.")
        sys.exit(0)

    # Embed and store
    rag = ChromaOllamaRAG()
    rag.db_dir = args.db_dir
    rag.collection_name = args.collection

    os.makedirs(args.db_dir, exist_ok=True)

    try:
        import chromadb

        try:
            client = chromadb.PersistentClient(path=args.db_dir)
        except TypeError:
            from chromadb.config import Settings
            client = chromadb.Client(Settings(persist_directory=args.db_dir, is_persistent=True))

        # Replace existing collection
        try:
            client.delete_collection(args.collection)
        except Exception:
            pass

        collection = client.create_collection(args.collection, metadata={"description": "Pydoll/Selenium code for RAG"})

        # Embed in batches
        batch_size = 20
        for i in range(0, len(all_docs), batch_size):
            batch_docs = all_docs[i : i + batch_size]
            batch_metas = all_metadatas[i : i + batch_size]
            batch_ids = all_ids[i : i + batch_size]
            embeddings = [rag.embed_text(d) for d in batch_docs]
            collection.add(ids=batch_ids, documents=batch_docs, metadatas=batch_metas, embeddings=embeddings)
            print(f"  Added batch {i // batch_size + 1}/{(len(all_docs) + batch_size - 1) // batch_size}")

        print(f"\nDone. Stored {len(all_docs)} chunks in {args.db_dir}")
    except ImportError as e:
        print(f"Error: chromadb not installed. Run: pip install chromadb")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
