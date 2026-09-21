"""Agent base class.

Every agent has the same shape: it takes a context dict, does one job, and
returns an AgentResult. The uniform envelope is what makes the orchestrator
simple and what makes the execution trace displayable in the UI - a real
requirement for an agentic system, where "why did it answer that?" must be
answerable.

Agents never raise into the orchestrator. A failure is data: ok=False plus a
message, so the pipeline can degrade instead of collapsing.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from src.core.errors import WeatherIQError
from src.core.logging import get_logger
from src.core.models import AgentResult


class Agent(ABC):
    name: str = "agent"
    description: str = ""

    def __init__(self) -> None:
        self.log = get_logger(f"agent.{self.name}")

    @abstractmethod
    def execute(self, context: dict) -> Any:
        """Do the work. Raise WeatherIQError on an expected failure."""

    def run(self, context: dict) -> AgentResult:
        """Execute with timing, logging and error capture."""
        started = time.perf_counter()
        try:
            data = self.execute(context)
            elapsed = (time.perf_counter() - started) * 1000
            self.log.info("ok in %.0f ms", elapsed)
            return AgentResult(agent=self.name, ok=True, data=data, duration_ms=elapsed)

        except WeatherIQError as exc:
            # Expected failure: no location, no data, no index.
            elapsed = (time.perf_counter() - started) * 1000
            self.log.warning("failed: %s", exc)
            return AgentResult(
                agent=self.name, ok=False, error=str(exc), duration_ms=elapsed
            )

        except Exception as exc:  # noqa: BLE001 - the pipeline must not die
            elapsed = (time.perf_counter() - started) * 1000
            self.log.exception("unexpected error")
            return AgentResult(
                agent=self.name,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=elapsed,
            )
