from pathlib import Path

import pytest

from acpa_gemma.config import AppConfig
from acpa_gemma.pipeline import TrustSafetyPipeline


def test_real_run_raises_clear_error_for_empty_dataset(tmp_path: Path):
    pipeline = TrustSafetyPipeline(AppConfig(), dry_run=False)

    with pytest.raises(RuntimeError, match="No Agentic Eval records were loaded"):
        pipeline.run_to_file(input_dir=str(tmp_path), output_path=str(tmp_path / "out.jsonl"))
