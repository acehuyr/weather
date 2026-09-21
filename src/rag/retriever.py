"""Query-time retrieval.

Three configurations, selected by `EMBEDDING_BACKEND` in .env:

    hash                    lexical, zero-install, the default
    sentence-transformers   semantic, needs the extra dependency
    hybrid                  both, fused with Reciprocal Rank Fusion

`hybrid` measured best on Recall@4 - the metric that matters most for RAG,
because the generator reads every retrieved chunk, not just the first. See
`eval/results/report.md`.
"""
from __future__ import annotations

from config.settings import settings
from src.core.logging import get_logger
from src.core.models import Document
from src.rag.embeddings import get_embedder
from src.rag.fusion import reciprocal_rank_fusion
from src.rag.indexer import HYBRID_PARTS, build_one, index_dir
from src.rag.vector_store import get_store

log = get_logger("rag.retriever")


class Retriever:
    """Single-backend retrieval.

    If no index exists yet it builds one automatically, so a fresh clone of
    the repo works without anyone remembering to run a script first.
    """

    def __init__(self, backend: str = None, auto_build: bool = True) -> None:
        self.backend = backend or settings.embedding_backend
        self.embedder = get_embedder(self.backend)
        directory = index_dir(self.backend)
        self.store = get_store(directory=directory)

        if not self.store.load() and auto_build:
            log.info("no %s index found - building one now", self.backend)
            build_one(self.backend)
            self.store = get_store(directory=directory)
            self.store.load()

        # Queries must be encoded with the same IDF weights as the documents.
        if hasattr(self.embedder, "load"):
            self.embedder.load(directory)

    def search(self, query: str, top_k: int) -> list:
        """Raw ranked hits - the unit the fusion layer works with."""
        if not query.strip() or len(self.store) == 0:
            return []
        return self.store.search(self.embedder.encode([query])[0], top_k)

    def retrieve(self, query: str, top_k: int = None,
                 min_score: float = 0.05) -> list:
        top_k = top_k or settings.top_k
        hits = self.search(query, top_k)
        # Drop weak matches rather than padding the prompt with noise -
        # irrelevant context measurably degrades the generated answer.
        return [_to_document(h) for h in hits if h["score"] >= min_score]

    def __len__(self) -> int:
        return len(self.store)


class HybridRetriever:
    """Lexical and semantic retrieval fused with RRF.

    The two backends fail on different questions: lexical wins when the
    question reuses the corpus's vocabulary, semantic wins when it
    paraphrases. Fusing by rank rather than by score avoids comparing two
    similarity scales that were never comparable.
    """

    def __init__(self, auto_build: bool = True) -> None:
        self.parts = []
        for backend in HYBRID_PARTS:
            try:
                self.parts.append(Retriever(backend, auto_build=auto_build))
            except ImportError as exc:
                log.warning("hybrid: skipping %s (%s)", backend, exc)

        if not self.parts:
            raise RuntimeError("no retrieval backend is available")
        log.info("hybrid retrieval over %d backends", len(self.parts))

    def retrieve(self, query: str, top_k: int = None,
                 min_score: float = 0.0) -> list:
        top_k = top_k or settings.top_k
        if len(self.parts) == 1:
            return self.parts[0].retrieve(query, top_k)

        # Each backend contributes a deeper list than we finally need, so
        # fusion can promote a chunk ranked 5th by one and 2nd by the other.
        lists = [part.search(query, top_k * 2) for part in self.parts]
        fused = reciprocal_rank_fusion(lists)[:top_k]

        # A fused score is a rank statistic, not a similarity, so the
        # per-backend min_score threshold does not apply. Agreement between
        # backends is the quality signal instead.
        return [_to_document(h, score_key="fusion_score") for h in fused]

    def __len__(self) -> int:
        return max((len(p) for p in self.parts), default=0)


def _to_document(hit: dict, score_key: str = "score") -> Document:
    return Document(
        id=hit["id"],
        text=hit["text"],
        source=hit["source"],
        title=hit.get("title", ""),
        score=hit.get(score_key, hit.get("score", 0.0)),
    )


def build_retriever():
    """Construct whatever `EMBEDDING_BACKEND` asks for."""
    if settings.embedding_backend.lower() == "hybrid":
        return HybridRetriever()
    return Retriever()
