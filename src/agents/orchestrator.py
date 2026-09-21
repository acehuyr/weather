"""Orchestrator: runs the agents and carries state between them.

The execution order is fixed but the *path* is conditional - each agent is
skipped when the plan says it is not needed. That is the practical meaning of
"agentic" here: the planner's output decides which agents run, and every
decision is recorded in the trace.

    question
       |
    Planner ................ decides intent, location, date range
       |
    Data ................... fetches only what the plan requires
       |
    Analysis ............... statistics, trends, comparison
       |
    Anomaly ................ departures from the climatological normal
       |
    Retrieval .............. RAG over the knowledge base (query is
       |                     expanded using what the analysis found)
    Insight ................ grounded natural-language answer
"""
from __future__ import annotations

import time

from src.agents.analysis_agent import AnalysisAgent
from src.agents.anomaly_agent import AnomalyAgent
from src.agents.chat_agent import ChatAgent
from src.agents.data_agent import DataAgent
from src.agents.insight_agent import InsightAgent
from src.agents.planner import PlannerAgent
from src.agents.retrieval_agent import RetrievalAgent
from src.core.logging import get_logger
from src.core.models import Insight, PipelineResult

log = get_logger("orchestrator")


class Orchestrator:
    def __init__(self) -> None:
        self.planner = PlannerAgent()
        self.data = DataAgent()
        self.analysis = AnalysisAgent()
        self.anomaly = AnomalyAgent()
        self.retrieval = RetrievalAgent()
        self.insight = InsightAgent()
        self.chat_agent = ChatAgent()

    def chat(self, question: str, history: list = None,
             default_location: str = None) -> tuple:
        """Conversational entry point.

        Runs the same agent pipeline, but the final step writes prose and
        keeps the conversation in view. Returns (reply, PipelineResult) so
        the UI can show the charts and trace alongside the reply.
        """
        result = self.answer(
            question, history=history or [], conversational=True,
            default_location=default_location,
        )
        return result.reply, result

    def answer(self, question: str, history: list = None,
               conversational: bool = False,
               default_location: str = None) -> PipelineResult:
        started = time.perf_counter()
        result = PipelineResult(question=question)
        context: dict = {
            "question": question,
            "history": history or [],
            # The place the user picked in the UI - used only when the
            # question names none, or names one that does not exist.
            "default_location": default_location,
        }

        log.info("=" * 60)
        log.info("Q: %s", question)

        # 1. Plan -------------------------------------------------------
        step = self.planner.run(context)
        result.trace.append(step)
        if not step.ok:
            # Without a plan nothing downstream can run.
            result.insight = self._failure(
                f"The question could not be interpreted: {step.error}"
            )
            return result
        result.plan = context["plan"] = step.data
        step.note = f"intent={step.data.intent}, location={step.data.location}"

        # 2. Data -------------------------------------------------------
        step = self.data.run(context)
        result.trace.append(step)
        context["bundle"] = result.bundle = step.data if step.ok else None
        if step.ok and step.data is not None:
            step.note = step.data.describe()

        # 3. Analysis ---------------------------------------------------
        if result.bundle is not None:
            step = self.analysis.run(context)
            result.trace.append(step)
            context["analysis"] = result.analysis = step.data if step.ok else {}
            if step.ok:
                step.note = ", ".join(step.data.keys())

            # 4. Anomaly ------------------------------------------------
            if result.plan.intent in {"anomaly", "comparison", "trend", "forecast"}:
                step = self.anomaly.run(context)
                result.trace.append(step)
                context["anomalies"] = result.anomalies = step.data if step.ok else {}
                if step.ok:
                    step.note = f"{step.data.get('total', 0)} anomalies"

        # 5. Retrieval --------------------------------------------------
        if result.plan.needs_knowledge:
            step = self.retrieval.run(context)
            result.trace.append(step)
            context["documents"] = result.documents = step.data if step.ok else []
            if step.ok:
                step.note = f"{len(step.data)} chunks"

        # 6. Answer -----------------------------------------------------
        # Same grounded inputs, two presentations: prose for the chat UI,
        # a structured report for the API.
        if conversational:
            step = self.chat_agent.run(context)
            result.trace.append(step)
            result.reply = step.data if step.ok else (
                f"Sorry - I could not put an answer together: {step.error}"
            )
            # The confidence rating and bullet findings still come from the
            # Insight agent, but via its template path - no second model
            # call. Its prose would be discarded anyway, and generating it
            # was dominating total latency.
            try:
                result.insight = self.insight.summarise(context)
            except Exception as exc:  # noqa: BLE001 - never lose the reply
                log.warning("could not summarise: %s", exc)
                result.insight = self._failure(result.reply)
        else:
            step = self.insight.run(context)
            result.trace.append(step)
            if step.ok:
                result.insight = step.data
            else:
                result.insight = self._failure(
                    f"An answer could not be generated: {step.error}"
                )
            result.reply = result.insight.answer

        elapsed = (time.perf_counter() - started) * 1000
        log.info("pipeline finished in %.0f ms (%d agents)", elapsed, len(result.trace))
        return result

    @staticmethod
    def _failure(message: str) -> Insight:
        return Insight(
            answer=message, key_findings=[], recommendations=[], confidence="low"
        )


_singleton: Orchestrator = None


def get_orchestrator() -> Orchestrator:
    """Shared instance - the RAG index is loaded once, not per request."""
    global _singleton
    if _singleton is None:
        _singleton = Orchestrator()
    return _singleton
