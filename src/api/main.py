"""FastAPI application.

    uvicorn src.api.main:app --reload

Interactive documentation is generated automatically at /docs - useful as a
demonstration artefact in its own right.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from src.agents.orchestrator import get_orchestrator
from src.agents.retrieval_agent import get_retriever
from src.api.schemas import (
    AskRequest,
    AskResponse,
    DocumentOut,
    HealthResponse,
    PlanOut,
    TraceStep,
    series_from_bundle,
)
from src.core.logging import get_logger
from src.rag.indexer import build_index

log = get_logger("api")

app = FastAPI(
    title="Intelligent Weather Analysis API",
    description=(
        "Agentic AI and Retrieval-Augmented Generation over live and "
        "historical weather data."
    ),
    version="0.1.0",
)

# The Streamlit UI runs on a different port, so it is a cross-origin caller.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.on_event("startup")
def warm_up() -> None:
    """Build the index and load the agents before the first request."""
    get_orchestrator()
    log.info("API ready (llm_enabled=%s)", settings.llm_enabled)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        llm_enabled=settings.llm_enabled,
        # `settings.model` is the Anthropic model specifically; on a Groq
        # deployment it reported a Claude model that was never called.
        # `llm_label` resolves whichever backend is actually configured.
        model=settings.llm_label,
        indexed_chunks=len(get_retriever()),
        embedding_backend=settings.embedding_backend,
        vector_backend=settings.vector_backend,
    )


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    result = get_orchestrator().answer(request.question)

    if result.insight is None:
        raise HTTPException(status_code=500, detail="No answer could be produced.")

    plan = None
    if result.plan:
        plan = PlanOut(
            intent=result.plan.intent,
            location=result.plan.location,
            variables=result.plan.variables,
            start_date=result.plan.start_date,
            end_date=result.plan.end_date,
            rationale=result.plan.rationale,
        )

    return AskResponse(
        question=result.question,
        answer=result.insight.answer,
        key_findings=result.insight.key_findings,
        recommendations=result.insight.recommendations,
        confidence=result.insight.confidence,
        location=result.bundle.location.label if result.bundle else "",
        plan=plan,
        sources=[
            DocumentOut(
                source=d.source,
                title=d.title,
                score=round(d.score, 4),
                excerpt=d.text[:280],
            )
            for d in result.documents
        ],
        trace=[TraceStep(**step.to_dict()) for step in result.trace],
        series=series_from_bundle(result.bundle),
    )


@app.post("/reindex")
def reindex() -> dict:
    """Rebuild the RAG index after editing data/knowledge_base."""
    count = build_index()
    return {"status": "ok", "chunks": count}
