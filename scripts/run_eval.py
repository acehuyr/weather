"""Evaluation harness.

    python scripts/run_eval.py                 # planner + retrieval, offline, fast
    python scripts/run_eval.py --full          # adds groundedness (needs network)
    python scripts/run_eval.py --llm           # also scores the LLM planner

Measures three things:

1. **Planning**    - does the question get routed to the right intent,
                     location and variables?
2. **Retrieval**   - does the right knowledge-base document come back, and at
                     what rank? Every available embedding backend is scored
                     side by side in one process.
3. **Groundedness**- does every number in the generated answer actually trace
                     back to computed data?

Results are written to `eval/results/` as markdown and JSON.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import settings  # noqa: E402
from eval import metrics  # noqa: E402
from eval.dataset import CASES, cases_in, retrieval_cases, stats  # noqa: E402
from src.agents.planner import PlannerAgent  # noqa: E402
from src.core.errors import LLMError  # noqa: E402
from src.rag.chunker import chunk_directory  # noqa: E402
from src.rag.embeddings import HashingEmbedder, SentenceTransformerEmbedder  # noqa: E402
from src.rag.fusion import reciprocal_rank_fusion  # noqa: E402
from src.rag.vector_store import SimpleVectorStore  # noqa: E402

RESULTS_DIR = ROOT / "eval" / "results"
TOP_K = settings.top_k


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------

def evaluate_planner(use_llm: bool = False, split: str = None,
                     sleep: float = 0.0) -> dict:
    """Score intent routing, location extraction and variable detection.

    Scored on the planner's *raw* output, before defaults are applied -
    defaulting an empty location to Mumbai would otherwise be counted as a
    correct extraction.
    """
    agent = PlannerAgent()
    rows, intents_pred, intents_true = [], [], []
    location_hits, variable_f1s = [], []
    fell_back = 0

    for case in cases_in(split):
        if use_llm:
            # `plan_raw` skips `_apply_defaults` so both modes are scored on
            # what the planner itself produced. Going through `execute` here
            # scored the LLM after a fallback location had been substituted,
            # and penalised it on every definitional question.
            try:
                plan = agent.plan_raw(case.question, use_llm=True)
            except LLMError as exc:
                # A rate-limited run must not quietly report the rule-based
                # score under the LLM's name.
                print(f"    LLM failed ({str(exc)[:60]}) - counted as fallback")
                fell_back += 1
                plan = agent._rule_based(case.question)
            if sleep:
                # Free tiers meter tokens per minute, so the fix is pacing,
                # not retrying harder.
                time.sleep(sleep)
        else:
            plan = agent.plan_raw(case.question, use_llm=False)

        intents_pred.append(plan.intent)
        intents_true.append(case.intent)

        location_ok = plan.location.strip().lower() == case.location.strip().lower()
        location_hits.append(1.0 if location_ok else 0.0)

        f1 = metrics.set_f1(set(plan.variables), set(case.variables))
        variable_f1s.append(f1)

        rows.append({
            "question": case.question,
            "paraphrase": case.paraphrase,
            "intent_expected": case.intent,
            "intent_predicted": plan.intent,
            "intent_ok": plan.intent == case.intent,
            "location_expected": case.location,
            "location_predicted": plan.location,
            "location_ok": location_ok,
            "variable_f1": round(f1, 3),
        })

    paraphrases = [r for r in rows if r["paraphrase"]]
    plain = [r for r in rows if not r["paraphrase"]]

    return {
        "mode": "llm" if use_llm else "rule-based",
        "split": split or "all",
        "cases": len(rows),
        "fell_back_to_rules": fell_back,
        "intent_accuracy": metrics.accuracy(intents_pred, intents_true),
        "location_accuracy": metrics.mean(location_hits),
        "variable_f1": metrics.mean(variable_f1s),
        "intent_accuracy_plain": metrics.mean(
            [1.0 if r["intent_ok"] else 0.0 for r in plain]
        ),
        "intent_accuracy_paraphrase": metrics.mean(
            [1.0 if r["intent_ok"] else 0.0 for r in paraphrases]
        ),
        "confusion": metrics.confusion(intents_pred, intents_true),
        "failures": [r for r in rows if not r["intent_ok"] or not r["location_ok"]],
        "rows": rows,
    }


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------

def build_probe_index(embedder) -> tuple:
    """Build an in-memory index with a given embedder.

    Deliberately does not touch the persisted index or global settings, so
    two backends can be compared inside one process with no side effects.
    """
    records = chunk_directory(settings.kb_dir)
    texts = [r["text"] for r in records]
    if hasattr(embedder, "fit"):
        embedder.fit(texts)
    store = SimpleVectorStore(directory=RESULTS_DIR / "_scratch")
    store.add(records, embedder.encode(texts))
    return embedder, store


class HybridProbe:
    """Two backends fused with RRF, presented as one searchable index."""

    def __init__(self, parts: list) -> None:
        self.parts = parts                     # [(embedder, store), ...]
        self.dim = sum(e.dim for e, _ in parts)

    def search(self, question: str, top_k: int) -> list:
        # Each backend contributes a deeper list than we finally need, so
        # fusion has room to promote a document ranked 5th by one retriever
        # and 2nd by the other.
        lists = [
            store.search(embedder.encode([question])[0], top_k * 2)
            for embedder, store in self.parts
        ]
        return reciprocal_rank_fusion(lists)[:top_k]


def evaluate_retrieval(built, label: str, split: str = None) -> dict:
    """`built` is either a HybridProbe or an (embedder, store) pair."""
    hybrid = isinstance(built, HybridProbe)
    if hybrid:
        embedder = built
    else:
        embedder, store = built
    cases = retrieval_cases(split)

    r_at_1, r_at_k, p_at_1, rr, rows = [], [], [], [], []

    for case in cases:
        if hybrid:
            hits = embedder.search(case.question, TOP_K)
        else:
            hits = store.search(embedder.encode([case.question])[0], TOP_K)
        retrieved = [h["source"] for h in hits]
        gold = set(case.gold_sources)

        r1 = metrics.recall_at_k(retrieved, gold, 1)
        rk = metrics.recall_at_k(retrieved, gold, TOP_K)
        p1 = metrics.precision_at_1(retrieved, gold)
        mrr = metrics.reciprocal_rank(retrieved, gold)

        r_at_1.append(r1)
        r_at_k.append(rk)
        p_at_1.append(p1)
        rr.append(mrr)

        rows.append({
            "question": case.question,
            "gold": sorted(gold),
            "retrieved": retrieved,
            "top_score": round(hits[0].get("fusion_score", hits[0]["score"]), 4)
            if hits else 0.0,
            "hit": bool(rk),
            "reciprocal_rank": round(mrr, 3),
        })

    return {
        "backend": label,
        "split": split or "all",
        "dim": embedder.dim,
        "cases": len(cases),
        f"recall@{TOP_K}": metrics.mean(r_at_k),
        "recall@1": metrics.mean(r_at_1),
        "precision@1": metrics.mean(p_at_1),
        "mrr": metrics.mean(rr),
        "misses": [r for r in rows if not r["hit"]],
        "rows": rows,
    }


def available_backends(only: str = None) -> list:
    """Every embedding backend installed right now.

    `only` filters by substring, so a single backend can be re-measured
    without waiting on the other.
    """
    backends = [("hashed TF-IDF", lambda: HashingEmbedder())]
    try:
        import sentence_transformers  # noqa: F401
        backends.append(
            ("sentence-transformers", lambda: SentenceTransformerEmbedder())
        )
    except ImportError:
        pass

    if only:
        backends = [b for b in backends if only.lower() in b[0].lower()]
    return backends


def build_all(backends: list, report: dict) -> list:
    """Construct and index every backend exactly once.

    Loading a transformer takes tens of seconds, so the hybrid reuses the
    indexes already built rather than constructing its own copies.
    """
    built = []
    for label, factory in backends:
        print(f"  {label}")
        try:
            built.append((label, build_probe_index(factory())))
        except Exception as exc:  # noqa: BLE001
            # A backend that cannot load (missing weights, offline hub) must
            # not take the rest of the report down with it.
            print(f"    unavailable: {type(exc).__name__}: {exc}")
            report.setdefault("retrieval_errors", []).append(
                {"backend": label, "error": f"{type(exc).__name__}: {exc}"}
            )

    # Fusing is only meaningful with two independent retrievers.
    if len(built) > 1:
        print("  hybrid (RRF)")
        built.append(("hybrid (RRF)", HybridProbe([p for _, p in built])))
    return built


# --------------------------------------------------------------------------
# Groundedness
# --------------------------------------------------------------------------

class _FallbackWatcher(logging.Handler):
    """Detects the Insight agent quietly answering from its template.

    `settings.llm_enabled` only reports that a key is configured. On a free
    tier a burst of 429s makes individual generations fail, and the agent
    falls back to the template - which can only emit numbers the pipeline
    just computed, and therefore scores a trivial 100% here.

    Reporting that as a model-generated result is precisely the mislabelling
    this section exists to prevent, so each case is watched and the ones the
    model did not actually write are counted separately.
    """

    MARKER = "using template"

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.tripped = False

    def emit(self, record: logging.LogRecord) -> None:
        if self.MARKER in record.getMessage():
            self.tripped = True



def evaluate_groundedness(limit: int = None, sleep: float = 0.0) -> dict:
    """Run the real pipeline and check every number in the answer.

    Requires network access: each case fetches live and archived weather.

    `sleep` paces the loop for the same reason the planner scorer takes it -
    this is the heaviest LLM consumer in the harness, one generation per
    case, and an unpaced run on a free tier spends its budget partway
    through and finishes on degraded output.
    """
    from src.agents.orchestrator import get_orchestrator

    orchestrator = get_orchestrator()
    cases = CASES[:limit] if limit else CASES
    scores, rows = [], []

    # Agent loggers hang off "weatheriq", which does not propagate further.
    watcher = _FallbackWatcher()
    logging.getLogger("weatheriq").addHandler(watcher)
    fell_back = 0

    for index, case in enumerate(cases, start=1):
        print(f"  [{index}/{len(cases)}] {case.question}", flush=True)
        watcher.tripped = False
        result = orchestrator.answer(case.question)
        templated = watcher.tripped
        fell_back += 1 if templated else 0
        if sleep:
            time.sleep(sleep)

        known: set = set()
        metrics.collect_known_numbers(result.analysis, known)
        metrics.collect_known_numbers(result.anomalies, known)
        if result.bundle is not None:
            metrics.collect_known_numbers(result.bundle.current, known)

        insight = result.insight
        text = " ".join(
            [insight.answer] + insight.key_findings + insight.recommendations
        )
        score = metrics.groundedness(text, known)
        scores.append(score["score"])

        rows.append({
            "question": case.question,
            "intent": result.plan.intent if result.plan else "",
            "templated": templated,
            "confidence": insight.confidence,
            "numbers_claimed": score["claimed"],
            "grounded": score["grounded"],
            "ungrounded": score["ungrounded"],
            "score": round(score["score"], 3),
        })

    logging.getLogger("weatheriq").removeHandler(watcher)

    return {
        "cases": len(rows),
        "mean_groundedness": metrics.mean(scores),
        "fully_grounded": sum(1 for r in rows if r["score"] == 1.0),
        "violations": [r for r in rows if r["score"] < 1.0],
        # Which generator actually wrote these answers. Without this the two
        # very different numbers this metric can produce are indistinguishable
        # in the report: offline the template can only emit figures it just
        # computed, so 100% is arithmetic rather than evidence, while with a
        # model connected the same 100% is a real (if single-sample) result.
        "llm_enabled": settings.llm_enabled,
        "llm_label": settings.llm_label,
        # How many answers the model did NOT write, despite a key being set.
        "fell_back_to_template": fell_back,
        "model_written": len(rows) - fell_back,
        "rows": rows,
    }


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_report(report: dict) -> str:
    out: list = [
        "# Evaluation Report",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Dataset",
        "",
        f"- Questions: **{report['dataset']['total']}** "
        f"({report['dataset']['dev']} dev / {report['dataset']['test']} test)",
        f"- With a labelled gold document: **{report['dataset']['with_gold_sources']}**",
        f"- Paraphrase cases: **{report['dataset']['paraphrases']}**",
        f"- By intent: {report['dataset']['by_intent']}",
        "",
        "> **dev** cases are fair game for tuning, so those scores are",
        "> fitted and read high. They include the 12 questions that were the",
        "> v1 held-out split until their failures were used to broaden the",
        "> intent patterns - spending a split is allowed, pretending you",
        "> did not is not. **test** is a fresh 24-question split written",
        "> before that change and not consulted during it.",
        ">",
        "> **Quote the test column.**",
        "",
        "## 1. Planning",
        "",
        "| Mode | Split | n | Intent accuracy | Location accuracy "
        "| Variable F1 | Fell back |",
        "|---|---|---|---|---|---|---|",
    ]
    degraded = False
    for block in report["planning"]:
        fallbacks = block.get("fell_back_to_rules", 0)
        if fallbacks:
            degraded = True
        out.append(
            f"| {block['mode']} | **{block['split']}** | {block['cases']} "
            f"| {pct(block['intent_accuracy'])} "
            f"| {pct(block['location_accuracy'])} "
            f"| {block['variable_f1']:.3f} "
            f"| {fallbacks or '—'} |"
        )
    if degraded:
        out += [
            "",
            "> **The LLM row is not a clean measurement.** *Fell back* counts "
            "questions where the API call failed - almost always a free-tier "
            "rate limit - and the rule-based planner answered instead. Those "
            "rows drag the LLM score toward the rule-based one. Re-run when "
            "the limit resets, or pace the run with `--sleep`.",
        ]

    for block in report["planning"]:
        if block["failures"]:
            out += [
                "",
                f"### Planning failures ({block['mode']}, {block['split']})",
                "",
            ]
            for row in block["failures"]:
                bits = []
                if not row["intent_ok"]:
                    bits.append(
                        f"intent {row['intent_predicted']} != {row['intent_expected']}"
                    )
                if not row["location_ok"]:
                    bits.append(
                        f"location {row['location_predicted']!r} != "
                        f"{row['location_expected']!r}"
                    )
                out.append(f"- `{row['question']}` - {'; '.join(bits)}")

    out += [
        "",
        "## 2. Retrieval",
        "",
        f"Top-k = {TOP_K}. Scored on the "
        f"{report['dataset']['with_gold_sources']} questions with a labelled "
        "gold document. Retrieval was never tuned against either split, so "
        "dev and test are both honest here; they are shown separately only "
        "for consistency.",
        "",
        "| Backend | Split | n | Dim | Recall@1 | "
        f"Recall@{TOP_K} | Precision@1 | MRR |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for block in report["retrieval"]:
        out.append(
            f"| {block['backend']} | **{block['split']}** | {block['cases']} "
            f"| {block['dim']} "
            f"| {pct(block['recall@1'])} "
            f"| {pct(block[f'recall@{TOP_K}'])} "
            f"| {pct(block['precision@1'])} "
            f"| {block['mrr']:.3f} |"
        )

    for block in report.get("retrieval_errors", []):
        out += ["", f"> Backend **{block['backend']}** could not be scored: "
                    f"`{block['error']}`"]

    for block in report["retrieval"]:
        if block["misses"]:
            out += [
                "",
                f"### Retrieval misses ({block['backend']}, {block['split']})",
                "",
            ]
            for row in block["misses"]:
                out.append(
                    f"- `{row['question']}` - wanted {row['gold']}, "
                    f"got {row['retrieved'][:2]}"
                )

    if report.get("groundedness"):
        block = report["groundedness"]
        cases = block["cases"]
        templated = block.get("fell_back_to_template", 0)
        written = block.get("model_written", 0)
        configured = block.get("llm_enabled")

        if not configured:
            generator = "offline template (no model connected)"
        elif templated == 0:
            generator = f"language model (`{block.get('llm_label')}`)"
        elif written == 0:
            generator = (
                f"template - every call to `{block.get('llm_label')}` failed"
            )
        else:
            generator = (
                f"mixed: {written}/{cases} written by "
                f"`{block.get('llm_label')}`, {templated} fell back to the "
                "template"
            )

        out += [
            "",
            "## 3. Groundedness",
            "",
            f"- Generated by: **{generator}**",
            f"- Mean groundedness: **{pct(block['mean_groundedness'])}**",
            f"- Fully grounded answers: **{block['fully_grounded']}/{cases}**",
            "",
            "Every number in the answer is checked against the numbers the "
            "pipeline actually computed. IMD thresholds and years are excluded "
            "as system vocabulary rather than data claims.",
        ]

        if configured and templated == 0:
            out += [
                "",
                "> Measured with the model writing every answer, so it was "
                "free to state a figure nothing computed. This is the number "
                "that carries information - but generation is "
                "non-deterministic, so one run is one sample. Run `--full` "
                "several times and quote the worst, not the best.",
            ]
        elif templated:
            out += [
                "",
                f"> **Only {written} of {cases} answers are evidence.** The "
                "other "
                f"{templated} fell back to the template after a failed "
                "generation - almost always a free-tier rate limit - and the "
                "template can only emit numbers the pipeline just computed, "
                "so those score 100% by construction. Re-run when the limit "
                "resets, or pace it harder with `--sleep`.",
            ]
        else:
            out += [
                "",
                "> **This is the control, not the result.** With no model "
                "connected the answers come from a template that can only "
                "emit numbers the pipeline just computed, so a perfect score "
                "is arithmetic - it shows the checker works end to end and "
                "nothing more. Set a key and re-run `--full` for the number "
                "worth quoting.",
            ]

        if block["violations"]:
            out += ["", "### Ungrounded numbers", ""]
            for row in block["violations"]:
                out.append(
                    f"- `{row['question']}` - {row['ungrounded']}"
                )

    out += ["", "---", "", "Reproduce with `python scripts/run_eval.py --full`."]
    return "\n".join(out)


# --------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Run the evaluation suite")
    parser.add_argument("--full", action="store_true",
                        help="include groundedness (makes live API calls)")
    parser.add_argument("--llm", action="store_true",
                        help="also score the LLM planner (needs ANTHROPIC_API_KEY)")
    parser.add_argument("--limit", type=int, default=None,
                        help="limit groundedness cases, for a quick check")
    parser.add_argument("--backend", default=None,
                        help="score only backends matching this substring, "
                             "e.g. --backend hashed")
    parser.add_argument("--sleep", type=float, default=0.0,
                        help="seconds to pause between LLM calls, in both "
                             "the planner and groundedness passes, to stay "
                             "under a free-tier rate limit (try 2.0)")
    args = parser.parse_args()

    logging.disable(logging.INFO)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "dataset": stats(),
        "planning": [],
        "retrieval": [],
    }

    # Dev and test are reported separately throughout: the dev number is
    # fitted (the patterns were debugged against it), the test number is not.
    print("Planning...")
    for split in ("dev", "test"):
        report["planning"].append(evaluate_planner(use_llm=False, split=split))
    if args.llm:
        if settings.llm_enabled:
            print("Planning (LLM)...")
            for split in ("dev", "test"):
                report["planning"].append(
                    evaluate_planner(use_llm=True, split=split,
                                     sleep=args.sleep)
                )
        else:
            print("  skipped: ANTHROPIC_API_KEY is not set")

    print("Retrieval...")
    for label, built in build_all(available_backends(args.backend), report):
        for split in ("dev", "test"):
            report["retrieval"].append(
                evaluate_retrieval(built, label, split=split)
            )

    if args.full:
        print("Groundedness (live data)...")
        report["groundedness"] = evaluate_groundedness(
            limit=args.limit, sleep=args.sleep
        )

    markdown = render_report(report)
    (RESULTS_DIR / "report.md").write_text(markdown, encoding="utf-8")
    (RESULTS_DIR / "report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )

    print()
    print(markdown)
    print()
    print(f"Saved to {RESULTS_DIR / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
