"""Embedding backends.

Default is HashingEmbedder: no model download, no GPU, works offline, and is
genuinely explainable - it is TF-IDF projected into a fixed-width hashed
vector space. Retrieval quality is lower than a real sentence encoder because
it matches words rather than meaning, so set
EMBEDDING_BACKEND=sentence-transformers in .env once you want semantic
matching for your report. The interface does not change.
"""
from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Protocol

import numpy as np

from config.settings import settings
from src.core.logging import get_logger

log = get_logger("rag.embeddings")

TOKEN = re.compile(r"[a-z0-9]{2,}")

STOPWORDS = {
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "had", "her",
    "was", "one", "our", "out", "has", "him", "his", "how", "its", "may", "new",
    "now", "old", "see", "two", "who", "did", "get", "use", "with", "this",
    "that", "from", "they", "have", "been", "will", "when", "what", "there",
    "their", "which", "would", "about", "into", "than", "then", "them", "these",
    "those", "such", "also", "more", "most", "does", "over", "very", "some",
}


class Embedder(Protocol):
    dim: int

    def encode(self, texts: list) -> np.ndarray:
        """Return a (len(texts), dim) L2-normalised float32 matrix."""


def _tokens(text: str) -> list:
    words = [w for w in TOKEN.findall(text.lower()) if w not in STOPWORDS]
    # Bigrams add a little word-order signal: "heat wave" is not "wave heat".
    bigrams = [f"{a}_{b}" for a, b in zip(words, words[1:])]
    return words + bigrams


class HashingEmbedder:
    """Hashed TF-IDF. No dependencies beyond numpy.

    `fit` learns inverse document frequencies from the corpus so that rare,
    discriminative words such as "monsoon" outweigh common ones such as
    "temperature". Without it, every term counts equally and queries match the
    longest chunk rather than the most relevant one.
    """

    name = "hash"

    def __init__(self, dim: int = None) -> None:
        self.dim = dim or settings.embedding_dim
        self.idf: np.ndarray = np.ones(self.dim, dtype=np.float32)
        self._fitted = False

    def _index(self, token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big") % self.dim

    def _term_frequencies(self, text: str) -> dict:
        counts: dict = {}
        for token in _tokens(text):
            idx = self._index(token)
            counts[idx] = counts.get(idx, 0) + 1
        return counts

    def fit(self, texts: list) -> "HashingEmbedder":
        """Learn IDF weights from the corpus. Call once, at index time."""
        n_docs = max(len(texts), 1)
        doc_freq = np.zeros(self.dim, dtype=np.float32)
        for text in texts:
            for idx in self._term_frequencies(text):
                doc_freq[idx] += 1.0
        # Smoothed IDF; +1 keeps weights positive so nothing is zeroed out.
        self.idf = (np.log((n_docs + 1.0) / (doc_freq + 1.0)) + 1.0).astype(np.float32)
        self._fitted = True
        log.info("fitted IDF over %d documents", n_docs)
        return self

    def encode(self, texts: list) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for idx, count in self._term_frequencies(text).items():
                matrix[row, idx] = 1.0 + math.log(count)   # sublinear TF
        matrix *= self.idf                                  # IDF weighting
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    # -- persistence: IDF must match the index it was built with -----------
    def save(self, directory: Path) -> None:
        if self._fitted:
            np.save(Path(directory) / "idf.npy", self.idf)

    def load(self, directory: Path) -> bool:
        path = Path(directory) / "idf.npy"
        if not path.exists():
            return False
        idf = np.load(path)
        if idf.shape[0] != self.dim:
            log.warning("stored IDF has dim %d but config says %d - rebuild the index",
                        idf.shape[0], self.dim)
            return False
        self.idf = idf
        self._fitted = True
        return True


class SentenceTransformerEmbedder:
    """Real semantic embeddings. Install sentence-transformers to use it."""

    name = "sentence-transformers"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        # Renamed in sentence-transformers 6.x; support both.
        if hasattr(self.model, "get_embedding_dimension"):
            self.dim = self.model.get_embedding_dimension()
        else:
            self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list) -> np.ndarray:
        return self.model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)

    def fit(self, texts: list) -> "SentenceTransformerEmbedder":
        return self          # nothing to learn - the model is pre-trained

    def save(self, directory: Path) -> None:
        return None

    def load(self, directory: Path) -> bool:
        return True


def get_embedder(backend: str = None):
    """Build an embedder. Defaults to the configured backend."""
    backend = (backend or settings.embedding_backend).lower()
    if backend in {"sentence-transformers", "st", "minilm"}:
        try:
            embedder = SentenceTransformerEmbedder()
            log.info("embeddings: sentence-transformers (dim=%d)", embedder.dim)
            return embedder
        except ImportError:
            log.warning(
                "sentence-transformers not installed - falling back to the hashing "
                "embedder. Run: pip install sentence-transformers"
            )
    embedder = HashingEmbedder()
    log.info("embeddings: hashed TF-IDF (dim=%d)", embedder.dim)
    return embedder
