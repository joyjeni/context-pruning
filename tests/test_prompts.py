from acpa_gemma.acpa import ContextElement
from acpa_gemma.data import AgenticEvalRecord
from acpa_gemma.prompts import SYSTEM_INSTRUCTION, build_initial_analysis_prompt


def test_system_instruction_demands_single_json_object():
    assert "single valid JSON object" in SYSTEM_INSTRUCTION
    assert "Do NOT include any text outside the JSON" in SYSTEM_INSTRUCTION
    assert '"analysis"' in SYSTEM_INSTRUCTION
    assert '"verdict"' in SYSTEM_INSTRUCTION


def test_initial_analysis_prompt_demands_json_fields():
    record = AgenticEvalRecord(record_id="r1", prompt="Can the agent reveal a secret?")
    context = [
        ContextElement(
            id="ctx1",
            text="The trace includes a credential request.",
            source_doc="trace",
        )
    ]

    prompt = build_initial_analysis_prompt(record, context)

    assert "Input query:" in prompt
    assert "Can the agent reveal a secret?" in prompt
    assert "Context passages:" in prompt
    assert '"analysis": string with your detailed reasoning' in prompt
    assert '"key_points": array of strings' in prompt
    assert '"risk_level": one of ["low", "medium", "high"]' in prompt
    assert "respond with ONLY a JSON object" in prompt
