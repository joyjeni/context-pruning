from pathlib import Path

import pytest

from acpa_gemma import benchmark


def test_benchmark_raises_clear_error_for_empty_dataset(tmp_path: Path):
    with pytest.raises(RuntimeError, match="No AgentEval records were loaded"):
        benchmark.main(
            [
                "--input",
                str(tmp_path),
                "--details-output",
                str(tmp_path / "details.csv"),
                "--summary-output",
                str(tmp_path / "summary.csv"),
                "--report-output",
                str(tmp_path / "report.md"),
            ]
        )
