"""Command-line interface.

    python -m src.cli ask "What is the weather trend in Mumbai?"
    python -m src.cli index
    python -m src.cli demo
"""
from __future__ import annotations

import argparse
import sys
import textwrap

from src.agents.orchestrator import get_orchestrator
from src.core.models import PipelineResult
from src.rag.indexer import build_index

DEMO_QUESTIONS = [
    "What is the weather trend in Mumbai?",
    "Will rainfall increase this week in Pune?",
    "How has the temperature in Delhi changed compared with previous years?",
    "Why is the current weather in Chennai unusual?",
    "What is relative humidity?",
]

WIDTH = 78


def _rule(char: str = "-") -> str:
    return char * WIDTH


def render(result: PipelineResult) -> str:
    out: list = [_rule("="), f"Q: {result.question}", _rule("=")]

    if result.plan:
        p = result.plan
        out.append(
            f"Plan    : intent={p.intent} | location={p.location} | "
            f"variables={', '.join(p.variables)}"
        )
        if p.start_date:
            out.append(f"Window  : {p.start_date} to {p.end_date}")

    insight = result.insight
    if insight:
        out.append("")
        out.append(textwrap.fill(insight.answer, WIDTH))

        if insight.key_findings:
            out.append("")
            out.append("Key findings:")
            for item in insight.key_findings:
                out.append(textwrap.fill(item, WIDTH,
                                         initial_indent="  - ",
                                         subsequent_indent="    "))

        if insight.recommendations:
            out.append("")
            out.append("Recommendations:")
            for item in insight.recommendations:
                out.append(textwrap.fill(item, WIDTH,
                                         initial_indent="  * ",
                                         subsequent_indent="    "))

        out.append("")
        out.append(f"Confidence: {insight.confidence}")

    if result.documents:
        out.append("")
        out.append("Sources retrieved:")
        for doc in result.documents:
            out.append(f"  [{doc.score:.3f}] {doc.source} :: {doc.title}")

    out.append("")
    out.append(_rule())
    out.append("Agent trace:")
    for step in result.trace:
        status = "ok  " if step.ok else "FAIL"
        detail = step.note or step.error
        out.append(f"  {status} {step.agent:<10} {step.duration_ms:7.0f} ms  {detail}")

    return "\n".join(out)


def main(argv: list = None) -> int:
    parser = argparse.ArgumentParser(
        prog="weatheriq",
        description="Agentic weather analysis with retrieval-augmented generation",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="Ask a question")
    ask.add_argument("question", nargs="+", help="The question, in plain English")

    sub.add_parser("index", help="Rebuild the RAG index from data/knowledge_base")
    sub.add_parser("demo", help="Run a set of sample questions")

    args = parser.parse_args(argv)

    if args.command == "index":
        print(f"Indexed {build_index()} chunks.")
        return 0

    orchestrator = get_orchestrator()

    if args.command == "demo":
        for question in DEMO_QUESTIONS:
            print(render(orchestrator.answer(question)))
            print()
        return 0

    print(render(orchestrator.answer(" ".join(args.question))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
