import json
import zipfile
from pathlib import Path

from acpa_gemma.cuad import (
    SOTA_BASELINE_POLICIES,
    UsageDrivenContextPruner,
    evaluate_usage_pruning,
    gold_answer_sections,
    load_cuad_dataset,
    main,
)


def write_cuad_json(path: Path) -> None:
    context = (
        "MASTER SERVICES AGREEMENT\n\n"
        "The term of this Agreement is three years from the Effective Date.\n\n"
        "The governing law is Delaware law.\n\n"
        "The confidentiality obligations survive termination."
    )
    payload = {
        "version": "test",
        "data": [
            {
                "title": "demo_contract",
                "paragraphs": [
                    {
                        "context": context,
                        "qas": [
                            {
                                "id": "q_doc",
                                "question": "Highlight the document name.",
                                "answers": [
                                    {
                                        "text": "MASTER SERVICES AGREEMENT",
                                        "answer_start": context.index("MASTER SERVICES AGREEMENT"),
                                    }
                                ],
                                "is_impossible": False,
                            },
                            {
                                "id": "q_term",
                                "question": "Highlight the term.",
                                "answers": [
                                    {
                                        "text": "three years",
                                        "answer_start": context.index("three years"),
                                    }
                                ],
                                "is_impossible": False,
                            },
                            {
                                "id": "q_law",
                                "question": "Highlight the governing law.",
                                "answers": [
                                    {
                                        "text": "Delaware law",
                                        "answer_start": context.index("Delaware law"),
                                    }
                                ],
                                "is_impossible": False,
                            },
                        ],
                    }
                ],
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_loads_cuad_zip_and_maps_answers_to_sections(tmp_path: Path):
    json_path = tmp_path / "CUADv1.json"
    write_cuad_json(json_path)
    zip_path = tmp_path / "data.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(json_path, "CUADv1.json")

    contracts = load_cuad_dataset(zip_path, max_section_chars=80)

    assert len(contracts) == 1
    assert contracts[0].questions[1].answer_texts == ["three years"]
    gold_sections = gold_answer_sections(contracts[0], contracts[0].questions[1])
    assert len(gold_sections) == 1
    assert "three years" in gold_sections[0].text


def test_usage_pruner_prioritizes_repeated_correct_sections(tmp_path: Path):
    json_path = tmp_path / "CUADv1.json"
    write_cuad_json(json_path)
    contract = load_cuad_dataset(json_path, max_section_chars=80)[0]
    pruner = UsageDrivenContextPruner()

    pruner.observe_correct_answer(contract, contract.questions[1], question_index=0)
    retained = pruner.retain_sections(contract.sections, prune_ratio=0.5, current_index=1)

    assert any("three years" in section.text for section in retained)


def test_evaluate_usage_pruning_reports_degradation_rows(tmp_path: Path):
    json_path = tmp_path / "CUADv1.json"
    write_cuad_json(json_path)
    contracts = load_cuad_dataset(json_path, max_section_chars=80)

    rows, details = evaluate_usage_pruning(
        contracts,
        prune_ratios=[0.0, 0.5],
        train_fraction=0.67,
        degradation_tolerance=0.05,
    )

    assert {"usage_driven", "hybrid_usage_bm25", *SOTA_BASELINE_POLICIES} == {
        row.policy for row in rows
    }
    usage_rows = [row for row in rows if row.policy == "usage_driven"]
    assert [row.prune_ratio for row in usage_rows] == [0.0, 0.5]
    assert usage_rows[0].citation_accuracy >= usage_rows[1].citation_accuracy
    assert any(row.baseline_policy for row in rows if row.policy == "usage_driven")
    assert all(
        row.sota_baselines_compared == len(SOTA_BASELINE_POLICIES)
        for row in rows
        if row.policy == "usage_driven"
    )
    assert details
    assert {row.policy for row in details}


def test_cuad_cli_writes_outputs(tmp_path: Path):
    json_path = tmp_path / "CUADv1.json"
    write_cuad_json(json_path)

    exit_code = main(
        [
            "--input",
            str(json_path),
            "--max-section-chars",
            "80",
            "--prune-ratios",
            "0,0.5",
            "--policies",
            "usage_driven,bm25_query_relevance,late_interaction_maxsim",
            "--summary-output",
            str(tmp_path / "summary.csv"),
            "--details-output",
            str(tmp_path / "details.csv"),
            "--report-output",
            str(tmp_path / "report.md"),
        ]
    )

    assert exit_code == 0
    summary = (tmp_path / "summary.csv").read_text(encoding="utf-8")
    assert "policy,policy_family,prune_ratio" in summary
    assert "combined_improvement_pct" in summary
    assert "sota_win_rate" in summary
    assert "CUAD Usage-Driven Context Pruning Report" in (
        tmp_path / "report.md"
    ).read_text(encoding="utf-8")
