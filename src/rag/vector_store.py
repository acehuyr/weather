"""Vector stores.

`SimpleVectorStore` is an exact cosine-similarity search over a numpy matrix.
For a knowledge base of a few hundred chunks that is not a compromise - it is
faster than an approximate index and has no moving parts. Swap in Chroma via
VECTOR_BACKEND=chroma when the corpus grows.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from config.settings import settings
from src.core.errors import RetrievalError
from src.core.logging import get_logger

log = get_logger("rag.store")


class VectorStore(Protocol):
    def add(self, records: list, vectors: np.ndarray) -> None: ...
    def search(self, vector: np.ndarray, top_k: int) -> list: ...
    def save(self) -> None: ...
    def load(self) -> bool: ...


class SimpleVectorStore:
    """Exact cosine search, persisted as one .npz plus one .json."""

    name = "simple"

    def __init__(self, directory: Path = None) -> None:
        self.dir = directory or settings.vectorstore_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.vectors: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self.records: list = []

    def add(self, records: list, vectors: np.ndarray) -> None:
        if len(records) != vectors.shape[0]:
            raise RetrievalError("records and vectors length mismatch")
        self.records = list(records)
        self.vectors = vectors.astype(np.float32)

    def save(self) -> None:
        np.savez_compressed(self.dir / "vectors.npz", vectors=self.vectors)
        (self.dir / "records.json").write_text(
            json.dumps(self.records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log.info("index saved: %d chunks -> %s", len(self.records), self.dir)

    def load(self) -> bool:
        vec_path = self.dir / "vectors.npz"
        rec_path = self.dir / "records.json"
        if not (vec_path.exists() and rec_path.exists()):
            return False
        self.vectors = np.load(vec_path)["vectors"]
        self.records = json.loads(rec_path.read_text(encoding="utf-8"))
        return True

    def search(self, vector: np.ndarray, top_k: int) -> list:
        if not self.records or self.vectors.size == 0:
            return []
        query = vector.reshape(-1).astype(np.float32)
        norm = float(np.linalg.norm(query))
        if norm > 0:
            query = query / norm
        # Stored vectors are already L2-normalised, so the dot product
        # IS the cosine similarity - no extra division needed.
        scores = self.vectors @ query
        top = np.argsort(scores)[::-1][:top_k]
        return [dict(self.records[i], score=float(scores[i])) for i in top]

    def __len__(self) -> int:
        return len(self.records)


class ChromaVectorStore:
    """Persistent Chroma collection. Install chromadb to use it."""

    name = "chroma"

    def __init__(self, directory: Path = None) -> None:
        import chromadb

        self.dir = directory or settings.vectorstore_dir
        self.client = chromadb.PersistentClient(path=str(self.dir / "chroma"))
        self.collection = self.client.get_or_create_collection(
            name="weather_kb", metadata={"hnsw:space": "cosine"}
        )

    def add(self, records: list, vectors: np.ndarray) -> None:
        self.collection.upsert(
            ids=[r["id"] for r in records],
            embeddings=[v.tolist() for v in vectors],
            documents=[r["text"] for r in records],
            metadatas=[{"source": r["source"], "title": r["title"]} for r in records],
        )

    def save(self) -> None:
        return None

    def load(self) -> bool:
        return self.collection.count() > 0

    def search(self, vector: np.ndarray, top_k: int) -> list:
        res = self.collection.query(
            query_embeddings=[vector.reshape(-1).tolist()], n_results=top_k
        )
        out = []
        for i, doc_id in enumerate(res["ids"][0]):
            meta = res["metadatas"][0][i]
            out.append({
                "id": doc_id,
                "text": res["documents"][0][i],
                "source": meta.get("source", ""),
                "title": meta.get("title", ""),
                # Chroma returns cosine DISTANCE; convert to similarity.
                "score": 1.0 - float(res["distances"][0][i]),
            })
        return out

    def __len__(self) -> int:
        return self.collection.count()


def get_store(directory: Path = None):
    if settings.vector_backend.lower() == "chroma":
        try:
            store = ChromaVectorStore(directory=directory)
            log.info("vector store: chroma")
            return store
        except ImportError:
            log.warning("chromadb not installed - falling back to SimpleVectorStore")
    return SimpleVectorStore(directory=directory)
