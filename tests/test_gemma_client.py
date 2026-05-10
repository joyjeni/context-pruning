import pytest

from acpa_gemma.config import GemmaConfig
from acpa_gemma.gemma_client import GemmaClient, JSON_ONLY_SYSTEM_INSTRUCTION


class FakeGemmaClient(GemmaClient):
    def __init__(self, text: str):
        super().__init__(GemmaConfig(api_key="test-key"))
        self.text = text
        self.last_system_instruction = None

    def generate(self, prompt: str, system_instruction: str | None = None) -> str:
        self.last_system_instruction = system_instruction
        return self.text


def test_generate_json_prepends_json_only_instruction():
    client = FakeGemmaClient('{"risk_level": "low"}')

    payload = client.generate_json("prompt", system_instruction="Assess safety.")

    assert payload == {"risk_level": "low"}
    assert client.last_system_instruction is not None
    assert client.last_system_instruction.startswith(JSON_ONLY_SYSTEM_INSTRUCTION)
    assert "Assess safety." in client.last_system_instruction


def test_generate_json_extracts_json_from_extra_text():
    client = FakeGemmaClient('Here is the result:\n{"risk_level": "medium"}\nDone.')

    assert client.generate_json("prompt") == {"risk_level": "medium"}


def test_generate_json_reports_non_json_output():
    client = FakeGemmaClient("I cannot comply.")

    with pytest.raises(ValueError, match="Gemma output must be a JSON object"):
        client.generate_json("prompt")
