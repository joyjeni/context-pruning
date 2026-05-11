"""Offline pruning benchmarks for ACPA and baseline algorithms.

The benchmark intentionally does not call Gemma. It isolates the context
pruning layer so it can run cheaply in Kaggle or CI while comparing ACPA with
common existing strategies: no pruning, random eviction, LRU, LFU, importance
ranking, and sliding-window truncation.
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import re
from statistics import mean
from typing import Any, Callable, Dict, Iterable, List, Sequence, Set, Tuple

from acpa_gemma.acpa import AdaptiveContextPruner, ContextElement
from acpa_gemma.config import load_config
from acpa_gemma.data import (
    SAFETY_KEYWORDS,
    AgenticEvalRecord,
    build_context_elements,
    dataset_diagnostics,
    extract_citations,
    format_dataset_diagnostics,
    load_agentic_eval_dataset,
)


PolicyFn = Callable[
    [AgenticEvalRecord, List[ContextElement], int, Set[str], random.Random],
    Tuple[List[ContextElement], Dict[str, float]],
]


@dataclass
class BenchmarkRow:
    policy: str
    record_id: str
    n_original: int
    n_retained: int
    n_evicted: int
    retention_ratio: float
    eviction_ratio: float
    original_token_count: int
    retained_token_count: int
    token_reduction_ratio: float
    citation_elements: int
    citations_preserved: int
    citation_preservation_rate: float
    safety_keyword_elements: int
    safety_keyword_elements_preserved: int
    safety_keyword_preservation_rate: float
    avg_importance_original: float
    avg_importance_retained: float


@dataclass
class BenchmarkSummary:
    policy: str
    records: int
    avg_retention_ratio: float
    avg_token_reduction_ratio: float
    avg_citation_preservation_rate: float
    avg_safety_keyword_preservation_rate: float
    avg_importance_lift: float


def run_benchmark(
    records: Sequence[AgenticEvalRecord],
    prune_ratio: float = 0.45,
    seed: int = 7,
) -> Tuple[List[BenchmarkRow], List[BenchmarkSummary]]:
    """Run ACPA and baseline pruning policies over records."""

    policies = build_policies(prune_ratio=prune_ratio)
    rows: List[BenchmarkRow] = []
    rng = random.Random(seed)

    for record in records:
        context = build_context_elements(record, timestamp=0)
        if not context:
            continue
        citations = extract_citations(record, context)
        for policy_name, policy_fn in policies.items():
            policy_rng = random.Random(rng.randint(0, 2**31 - 1))
            retained, stats = policy_fn(record, context, 1, citations, policy_rng)
            rows.append(
                score_policy_result(
                    policy=policy_name,
                    record=record,
                    context=context,
                    retained=retained,
                    citations=citations,
                    stats=stats,
                )
            )

    return rows, summarize_rows(rows)


def build_policies(prune_ratio: float) -> Dict[str, PolicyFn]:
    return {
        "no_pruning": no_pruning_policy,
        "random": lambda record, context, timestamp, citations, rng: rank_policy(
            context,
            prune_ratio=prune_ratio,
            key_fn=lambda element: rng.random(),
            reverse=True,
        ),
        "lru": lambda record, context, timestamp, citations, rng: rank_policy(
            context,
            prune_ratio=prune_ratio,
            key_fn=lambda element: element.timestamp + context_order(context, element.id) / 1000,
            reverse=True,
        ),
        "lfu": lambda record, context, timestamp, citations, rng: rank_policy(
            assign_simulated_access(copy_context(context), record),
            prune_ratio=prune_ratio,
            key_fn=lambda element: element.access_count,
            reverse=True,
        ),
        "importance": lambda record, context, timestamp, citations, rng: rank_policy(
            context,
            prune_ratio=prune_ratio,
            key_fn=lambda element: element.importance_score,
            reverse=True,
        ),
        "sliding_window": lambda record, context, timestamp, citations, rng: sliding_window_policy(
            context,
            prune_ratio=prune_ratio,
        ),
        "acpa_lfu_lru_dependency": lambda record, context, timestamp, citations, rng: acpa_policy(
            record=record,
            context=context,
            timestamp=timestamp,
            citations=citations,
            prune_ratio=prune_ratio,
        ),
    }


def no_pruning_policy(
    record: AgenticEvalRecord,
    context: List[ContextElement],
    timestamp: int,
    citations: Set[str],
    rng: random.Random,
) -> Tuple[List[ContextElement], Dict[str, float]]:
    copied = copy_context(context)
    return copied, {
        "n_original": len(copied),
        "n_retained": len(copied),
        "n_evicted": 0,
        "eviction_ratio": 0.0,
    }


def acpa_policy(
    record: AgenticEvalRecord,
    context: List[ContextElement],
    timestamp: int,
    citations: Set[str],
    prune_ratio: float,
) -> Tuple[List[ContextElement], Dict[str, float]]:
    copied = assign_simulated_access(copy_context(context), record)
    pruner = AdaptiveContextPruner(prune_ratio=prune_ratio)

    # Seed ACPA's LFU tracker from the simulated agent signal so hot context
    # receives the same priority boost path as the production pipeline.
    signal = simulated_agent_signal(record)
    pruner.track_context_access(copied, "benchmark_agent", signal)
    return pruner.competitive_eviction(
        context=copied,
        current_timestamp=timestamp,
        citations=citations,
    )


def rank_policy(
    context: List[ContextElement],
    prune_ratio: float,
    key_fn: Callable[[ContextElement], float],
    reverse: bool,
) -> Tuple[List[ContextElement], Dict[str, float]]:
    copied = copy_context(context)
    n_retain = max(1, round(len(copied) * (1 - prune_ratio)))
    ranked = sorted(copied, key=key_fn, reverse=reverse)
    retained = ranked[:n_retain]
    return retained, {
        "n_original": len(copied),
        "n_retained": len(retained),
        "n_evicted": len(copied) - len(retained),
        "eviction_ratio": 1 - (len(retained) / len(copied)),
    }


def assign_simulated_access(
    context: List[ContextElement],
    record: AgenticEvalRecord,
) -> List[ContextElement]:
    signal_terms = token_set(simulated_agent_signal(record))
    for element in context:
        element_terms = token_set(element.text)
        element.access_count = len(element_terms & signal_terms)
    return context


def simulated_agent_signal(record: AgenticEvalRecord) -> str:
    """Approximate the content an initial safety analyst would cite."""

    return " ".join(
        [
            record.label,
            record.response,
            " ".join(keyword for keyword in SAFETY_KEYWORDS if keyword in record.combined_text().lower()),
        ]
    )


def score_policy_result(
    policy: str,
    record: AgenticEvalRecord,
    context: List[ContextElement],
    retained: List[ContextElement],
    citations: Set[str],
    stats: Dict[str, float],
) -> BenchmarkRow:
    retained_ids = {element.id for element in retained}
    citation_ids = {
        element.id for element in context if element_contains_any(element, citations)
    }
    safety_ids = {
        element.id for element in context if contains_safety_keyword(element.text)
    }
    original_tokens = sum(token_count(element.text) for element in context)
    retained_tokens = sum(token_count(element.text) for element in retained)
    avg_original_importance = mean([element.importance_score for element in context])
    avg_retained_importance = mean([element.importance_score for element in retained])

    return BenchmarkRow(
        policy=policy,
        record_id=record.record_id,
        n_original=len(context),
        n_retained=len(retained),
        n_evicted=len(context) - len(retained),
        retention_ratio=len(retained) / len(context),
        eviction_ratio=1 - (len(retained) / len(context)),
        original_token_count=original_tokens,
        retained_token_count=retained_tokens,
        token_reduction_ratio=1 - (retained_tokens / original_tokens)
        if original_tokens
        else 0.0,
        citation_elements=len(citation_ids),
        citations_preserved=len(citation_ids & retained_ids),
        citation_preservation_rate=safe_ratio(len(citation_ids & retained_ids), len(citation_ids)),
        safety_keyword_elements=len(safety_ids),
        safety_keyword_elements_preserved=len(safety_ids & retained_ids),
        safety_keyword_preservation_rate=safe_ratio(len(safety_ids & retained_ids), len(safety_ids)),
        avg_importance_original=avg_original_importance,
        avg_importance_retained=avg_retained_importance,
    )


def summarize_rows(rows: Sequence[BenchmarkRow]) -> List[BenchmarkSummary]:
    grouped: Dict[str, List[BenchmarkRow]] = {}
    for row in rows:
        grouped.setdefault(row.policy, []).append(row)

    summaries: List[BenchmarkSummary] = []
    for policy, policy_rows in grouped.items():
        summaries.append(
            BenchmarkSummary(
                policy=policy,
                records=len(policy_rows),
                avg_retention_ratio=mean([row.retention_ratio for row in policy_rows]),
                avg_token_reduction_ratio=mean(
                    [row.token_reduction_ratio for row in policy_rows]
                ),
                avg_citation_preservation_rate=mean(
                    [row.citation_preservation_rate for row in policy_rows]
                ),
                avg_safety_keyword_preservation_rate=mean(
                    [row.safety_keyword_preservation_rate for row in policy_rows]
                ),
                avg_importance_lift=mean(
                    [
                        row.avg_importance_retained - row.avg_importance_original
                        for row in policy_rows
                    ]
                ),
            )
        )

    return sorted(
        summaries,
        key=lambda item: (
            item.avg_citation_preservation_rate,
            item.avg_safety_keyword_preservation_rate,
            item.avg_token_reduction_ratio,
        ),
        reverse=True,
    )


def write_csv(path: str | Path, rows: Sequence[Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_as_dicts = [asdict(row) for row in rows]
    if not rows_as_dicts:
        output_path.write_text("", encoding="utf-8")
        return

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_as_dicts[0].keys()))
        writer.writeheader()
        writer.writerows(rows_as_dicts)


def write_markdown_report(
    path: str | Path,
    summaries: Sequence[BenchmarkSummary],
    detail_rows_path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Pruning Benchmark Report",
        "",
        "This offline benchmark compares ACPA against common baseline pruning algorithms.",
        f"Detailed per-record rows: `{detail_rows_path}`",
        "",
        "| Policy | Records | Retention | Token reduction | Citation preservation | Safety keyword preservation | Importance lift |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            "| {policy} | {records} | {retention:.3f} | {tokens:.3f} | "
            "{citations:.3f} | {safety:.3f} | {lift:.3f} |".format(
                policy=summary.policy,
                records=summary.records,
                retention=summary.avg_retention_ratio,
                tokens=summary.avg_token_reduction_ratio,
                citations=summary.avg_citation_preservation_rate,
                safety=summary.avg_safety_keyword_preservation_rate,
                lift=summary.avg_importance_lift,
            )
        )

    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "- **Retention**: fraction of original context chunks retained.",
            "- **Token reduction**: whitespace-token proxy reduction after pruning.",
            "- **Citation preservation**: fraction of citation-bearing chunks retained.",
            "- **Safety keyword preservation**: fraction of safety-relevant chunks retained.",
            "- **Importance lift**: retained average importance minus original average importance.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def element_contains_any(element: ContextElement, needles: Iterable[str]) -> bool:
    haystack = element.text.lower()
    return any(needle.lower() in haystack for needle in needles if needle)


def contains_safety_keyword(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in SAFETY_KEYWORDS)


def token_set(text: str) -> Set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9_:-]+", text)
        if len(token) > 2
    }


def token_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 1.0
    return numerator / denominator


def copy_context(context: List[ContextElement]) -> List[ContextElement]:
    return copy.deepcopy(context)


def context_order(context: Sequence[ContextElement], element_id: str) -> int:
    for index, element in enumerate(context):
        if element.id == element_id:
            return index
    return 0


def sliding_window_policy(
    context: List[ContextElement],
    prune_ratio: float,
) -> Tuple[List[ContextElement], Dict[str, float]]:
    copied = copy_context(context)
    n_retain = max(1, round(len(copied) * (1 - prune_ratio)))
    retained = copied[-n_retain:]
    return retained, {
        "n_original": len(copied),
        "n_retained": len(retained),
        "n_evicted": len(copied) - len(retained),
        "eviction_ratio": 1 - (len(retained) / len(copied)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark ACPA against baseline context pruning algorithms."
    )
    parser.add_argument("--input", help="Agentic Eval dataset path.")
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--prune-ratio", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--details-output",
        default="outputs/benchmark_details.csv",
        help="Per-record benchmark CSV path.",
    )
    parser.add_argument(
        "--summary-output",
        default="outputs/benchmark_summary.csv",
        help="Aggregate summary CSV path.",
    )
    parser.add_argument(
        "--report-output",
        default="outputs/benchmark_report.md",
        help="Markdown report path.",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config()
    input_dir = args.input or config.data.input_dir
    sample_size = args.sample_size or config.data.sample_size
    records = load_agentic_eval_dataset(input_dir, sample_size=sample_size)
    if not records:
        diagnostics = dataset_diagnostics(input_dir)
        raise RuntimeError(
            "No AgentEval records were loaded for benchmark. Attach "
            "mukundakatta/agent-eval-scenarios in Kaggle Add data/Input, "
            "or point --input to CSV, JSON, JSONL, NDJSON, or Parquet records.\n\n"
            + format_dataset_diagnostics(diagnostics)
        )

    rows, summaries = run_benchmark(
        records,
        prune_ratio=args.prune_ratio,
        seed=args.seed,
    )
    write_csv(args.details_output, rows)
    write_csv(args.summary_output, summaries)
    write_markdown_report(args.report_output, summaries, args.details_output)

    print(
        json.dumps(
            {
                "records": len(records),
                "policies": [summary.policy for summary in summaries],
                "details_output": args.details_output,
                "summary_output": args.summary_output,
                "report_output": args.report_output,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
