"""Reciprocal Rank Fusion.

The evaluation showed the two retrievers failing on different questions:
lexical TF-IDF wins when the question reuses the corpus's own words, semantic
embeddings win when it paraphrases. Fusing them should beat either alone.

RRF combines ranked lists using only positions, never the raw scores:

    score(d) = sum over lists of  1 / (k + rank(d))

That matters here because a cosine similarity from a hashed TF-IDF space and
one from MiniLM are not on a comparable scale - averaging them directly would
let whichever backend happens to produce larger numbers dominate. Ranks have
no such problem, and RRF has no weights to tune.

k = 60 is the value from the original paper (Cormack et al., 2009); it damps
the influence of the very top rank so one confident-but-wrong list cannot
control the outcome.
"""
from __future__ import annotations

RRF_K = 60


def reciprocal_rank_fusion(ranked_lists: list, k: int = RRF_K,
                           id_key: str = "id") -> list:
    """Fuse ranked result lists into one.

    Each input is a list of dicts ordered best-first. Returns a single list
    ordered by fused score, with `fusion_score` and `sources_agreeing` added.
    """
    scores: dict = {}
    records: dict = {}
    agreement: dict = {}

    for results in ranked_lists:
        for rank, item in enumerate(results, start=1):
            key = item[id_key]
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            agreement[key] = agreement.get(key, 0) + 1
            # Keep the first record seen; they describe the same chunk.
            records.setdefault(key, item)

    fused = []
    for key, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        item = dict(records[key])
        item["fusion_score"] = score
        item["sources_agreeing"] = agreement[key]
        fused.append(item)
    return fused
