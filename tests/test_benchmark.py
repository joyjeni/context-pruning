from pathlib import Path

from acpa_gemma.benchmark import run_benchmark, write_csv, write_markdown_report
from acpa_gemma.pipeline import demo_records


def test_benchmark_runs_all_policies_on_demo_record():
    rows, summaries = run_benchmark(demo_records(), prune_ratio=0.45, seed=1)

    policies = {summary.policy for summary in summaries}

    assert "acpa_lfu_lru_dependency" in policies
    assert "no_pruning" in policies
    assert "random" in policies
    assert rows
    assert all(0 <= row.token_reduction_ratio <= 1 for row in rows)


def test_benchmark_writes_csv_and_markdown(tmp_path: Path):
    rows, summaries = run_benchmark(demo_records(), prune_ratio=0.45, seed=1)
    details = tmp_path / "details.csv"
    summary = tmp_path / "summary.csv"
    report = tmp_path / "report.md"

    write_csv(details, rows)
    write_csv(summary, summaries)
    write_markdown_report(report, summaries, details)

    assert "policy,record_id" in details.read_text(encoding="utf-8")
    assert "policy,records" in summary.read_text(encoding="utf-8")
    assert "Pruning Benchmark Report" in report.read_text(encoding="utf-8")
