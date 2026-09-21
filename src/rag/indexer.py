"""Build the RAG index: markdown -> chunks -> vectors -> store.

Indexes are namespaced by backend (`data/vectorstore/hash/`,
`data/vectorstore/st/`) so more than one can exist side by side. That is what
lets the hybrid retriever query both, and what lets the evaluation harness
compare them without rebuilding between runs.
"""
from __future__ import annotations

from pathlib import Path

from config.settings import settings
from src.core.logging import get_logger
from src.rag.chunker import chunk_directory
from src.rag.embeddings import get_embedder
from src.rag.vector_store import get_store

log = get_logger("rag.indexer")

# Backends the hybrid retriever fuses.
HYBRID_PARTS = ("hash", "sentence-transformers")


def index_dir(backend: str) -> Path:
    """Directory holding the index for one backend."""
    key = "st" if backend.lower().startswith("sentence") else "hash"
    return settings.vectorstore_dir / key


def build_one(backend: str, kb_dir: Path = None) -> int:
    """(Re)build the index for a single backend. Returns the chunk count."""
    kb_dir = kb_dir or settings.kb_dir
    records = chunk_directory(kb_dir)
    if not records:
        log.warning("no .md files found in %s - nothing to index", kb_dir)
        return 0

    texts = [r["text"] for r in records]
    target = index_dir(backend)
    target.mkdir(parents=True, exist_ok=True)

    embedder = get_embedder(backend)
    # Fit IDF on the corpus first - queries must be encoded with the same
    # weights, so the fitted state is saved alongside the vectors.
    if hasattr(embedder, "fit"):
        embedder.fit(texts)
    vectors = embedder.encode(texts)
    if hasattr(embedder, "save"):
        embedder.save(target)

    store = get_store(directory=target)
    store.add(records, vectors)
    store.save()

    file_count = len(list(kb_dir.glob("*.md")))
    log.info("[%s] indexed %d chunks from %d files", backend, len(records), file_count)
    return len(records)


def build_index(kb_dir: Path = None, backend: str = None) -> int:
    """Build whatever the configured backend needs.

    For `hybrid`, that means both underlying indexes.
    """
    backend = backend or settings.embedding_backend

    if backend.lower() == "hybrid":
        counts = []
        for part in HYBRID_PARTS:
            try:
                counts.append(build_one(part, kb_dir))
            except ImportError as exc:
                log.warning("skipping %s: %s", part, exc)
        return max(counts) if counts else 0

    return build_one(backend, kb_dir)


if __name__ == "__main__":
    print(f"Indexed {build_index()} chunks.")
