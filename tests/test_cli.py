import json

from acpa_gemma.cli import main


def test_cli_reports_missing_dataset_without_traceback(tmp_path, capsys):
    exit_code = main(
        [
            "--input",
            str(tmp_path / "missing"),
            "--output",
            str(tmp_path / "out.jsonl"),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.err)

    assert exit_code == 1
    assert "No Agentic Eval records were loaded" in payload["error"]
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_cli_reports_missing_api_key_without_traceback(tmp_path, capsys):
    data_file = tmp_path / "records.ndjson"
    data_file.write_text(
        '{"id": "r1", "prompt": "check prompt injection"}\n',
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--input",
            str(tmp_path),
            "--output",
            str(tmp_path / "out.jsonl"),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.err)

    assert exit_code == 1
    assert "Missing Gemma API key" in payload["error"]
    assert "Traceback" not in captured.err
    assert captured.out == ""
