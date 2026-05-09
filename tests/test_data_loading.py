import json
from pathlib import Path

from acpa_gemma.data import (
    dataset_diagnostics,
    extract_json_rows,
    load_agentic_eval_dataset,
)


def test_extract_json_rows_from_common_shapes():
    assert extract_json_rows({"records": [{"prompt": "hello"}]}) == [{"prompt": "hello"}]
    assert extract_json_rows({"examples": [{"query": "hi"}]}) == [{"query": "hi"}]
    assert extract_json_rows({"prompt": "single object"}) == [{"prompt": "single object"}]


def test_loads_ndjson_records(tmp_path: Path):
    data_file = tmp_path / "records.ndjson"
    data_file.write_text(
        json.dumps({"id": "r1", "prompt": "check prompt injection"}) + "\n",
        encoding="utf-8",
    )

    records = load_agentic_eval_dataset(tmp_path)

    assert len(records) == 1
    assert records[0].record_id == "r1"


def test_dataset_diagnostics_reports_supported_files(tmp_path: Path):
    (tmp_path / "records.jsonl").write_text('{"prompt": "hello"}\n', encoding="utf-8")
    (tmp_path / "readme.txt").write_text("metadata", encoding="utf-8")

    diagnostics = dataset_diagnostics(tmp_path)

    assert diagnostics["exists"] is True
    assert diagnostics["total_files"] == 2
    assert diagnostics["supported_files"] == 1
