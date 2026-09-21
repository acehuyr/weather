"""Retrieval agent: the R in RAG.

Retrieval is driven by the question *plus* what the analysis actually found,
not by the question alone. If the anomaly agent flagged extreme rainfall, the
retrieved context should include the rainfall category definitions even
though the user never used the word "category".
"""
from __future__ import annotations

from src.agents.base import Agent
from src.core.models import QueryPlan
from src.rag.retriever import build_retriever

_retriever = None


def get_retriever():
    """Built once per process - loading the index on every query is wasteful."""
    global _retriever
    if _retriever is None:
        _retriever = build_retriever()
    return _retriever


class RetrievalAgent(Agent):
    name = "retrieval"
    description = "Retrieves supporting meteorological knowledge"

    def execute(self, context: dict) -> list:
        plan: QueryPlan = context["plan"]
        question: str = context["question"]

        query = self._expand(question, plan, context)
        self.log.info("retrieval query: %s", query[:110])

        docs = get_retriever().retrieve(query)
        self.log.info("retrieved %d chunks", len(docs))
        return docs

    @staticmethod
    def _expand(question: str, plan: QueryPlan, context: dict) -> str:
        """Enrich the query with terms drawn from what the pipeline found."""
        parts = [question, plan.intent, " ".join(plan.variables)]

        anomaly_result = context.get("anomalies") or {}
        if anomaly_result.get("total"):
            parts.append("anomaly departure from normal z-score")
        for flag in anomaly_result.get("threshold_flags", [])[:3]:
            parts.append(str(flag.get("category", "")))

        if plan.intent == "trend":
            parts.append("trend interpretation record length seasonality")
        if plan.intent == "comparison":
            parts.append("climate normal baseline comparison")

        return " ".join(p for p in parts if p).strip()
