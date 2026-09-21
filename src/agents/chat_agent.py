"""Chat agent: the conversational face of the pipeline.

The Insight agent produces a structured report - answer, findings,
recommendations, confidence. That is the right shape for an API response and
the wrong shape for a conversation, where it reads like a form letter.

This agent takes the same grounded inputs and writes prose instead, keeping
the conversation history so follow-ups like "what about Pune?" work without
the user repeating themselves.

It is still bound by the same rule: every number must come from the data.
Conversational tone is a presentation choice, not permission to invent.
"""
from __future__ import annotations

from datetime import date

from src.agents.base import Agent
from src.core.errors import LLMError
from src.core.models import QueryPlan, WeatherBundle
from src.llm.client import get_llm
from src.llm.prompts import CHAT_SYSTEM, build_chat_prompt


class ChatAgent(Agent):
    name = "chat"
    description = "Writes the conversational reply"

    def execute(self, context: dict) -> str:
        plan: QueryPlan = context["plan"]
        bundle: WeatherBundle = context.get("bundle")
        history: list = context.get("history") or []

        llm = get_llm()
        if not llm.available:
            # No LLM: fall back to the structured template, flattened into
            # something that at least reads as a paragraph.
            return self._fallback(context)

        prompt = build_chat_prompt(
            question=context["question"],
            today=date.today().isoformat(),
            location=bundle.location.label if bundle else plan.location,
            current=(bundle.current if bundle else {}) or {},
            analysis=context.get("analysis") or {},
            anomalies=context.get("anomalies") or {},
            documents=context.get("documents") or [],
        )

        try:
            return llm.converse(CHAT_SYSTEM, history, prompt)
        except LLMError as exc:
            self.log.warning("chat generation failed (%s) - using template", exc)
            return self._fallback(context)

    # ------------------------------------------------------------------
    @staticmethod
    def _fallback(context: dict) -> str:
        """Offline path: reuse the Insight agent's template output."""
        from src.agents.insight_agent import InsightAgent

        insight = InsightAgent().execute(context)
        parts = [insight.answer]
        if insight.key_findings:
            parts.append(" ".join(insight.key_findings))
        if insight.recommendations:
            parts.append(" ".join(insight.recommendations))
        return "\n\n".join(parts)
