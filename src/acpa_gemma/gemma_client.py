"""Gemma 4 API wrapper."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from acpa_gemma.config import GemmaConfig, get_api_key, AppConfig


class GemmaGenerationError(RuntimeError):
    """Raised when Gemma generation cannot be completed."""


JSON_ONLY_SYSTEM_INSTRUCTION = (
    "You are a JSON-only assistant. "
    "You MUST respond with a single valid JSON object. "
    "Do not include explanations, markdown, or any text before or after the JSON. "
    "If you want to include text, put it inside JSON string values."
)


class GemmaClient:
    """Small wrapper around the Google GenAI SDK for Gemma 4 models."""

    def __init__(self, config: AppConfig | GemmaConfig, dry_run: bool = False) -> None:
        if isinstance(config, AppConfig):
            self.config = config.gemma
            self.api_key = get_api_key(config)
        else:
            self.config = config
            self.api_key = config.api_key.strip()
        self.dry_run = dry_run
        self._client = None

    def generate(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """Generate text with Gemma 4."""

        if self.dry_run:
            return self._dry_run_response(prompt)

        if not self.api_key:
            raise GemmaGenerationError(
                "Missing Gemma API key. Add [gemma].api_key to configs/secrets.toml "
                "or /kaggle/working/configs/secrets.toml."
            )

        client = self._get_client()
        try:
            from google.genai import types

            response = client.models.generate_content(
                model=self.config.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    max_output_tokens=self.config.max_output_tokens,
                    response_mime_type="application/json",
                ),
            )
        except Exception as exc:  # pragma: no cover - requires network/API key
            raise GemmaGenerationError(f"Gemma generation failed: {exc}") from exc

        text = getattr(response, "text", None)
        if text:
            return text

        # Fallback for SDK response shapes that expose candidates/parts.
        try:  # pragma: no cover - SDK-version dependent
            return response.candidates[0].content.parts[0].text
        except Exception as exc:
            raise GemmaGenerationError("Gemma response did not include text") from exc

    def generate_json(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a strict JSON object from Gemma output."""

        if system_instruction:
            system_instruction = f"{JSON_ONLY_SYSTEM_INSTRUCTION} {system_instruction}"
        else:
            system_instruction = JSON_ONLY_SYSTEM_INSTRUCTION

        text = self.generate(prompt=prompt, system_instruction=system_instruction)
        try:
            return parse_json_object(text)
        except ValueError:
            print("DEBUG RAW GEMMA OUTPUT:\n", text[:1000])
            match = re.search(r"\{.*\}", text, re.S)
            if not match:
                raise ValueError(
                    "Gemma output must be a JSON object, got:\n"
                    f"{text[:500]}"
                ) from None

            json_str = match.group(0)
            try:
                payload = json.loads(json_str)
            except Exception as exc:
                raise ValueError(
                    "Failed to parse JSON from Gemma output. "
                    f"Extracted:\n{json_str[:500]}\n\n"
                    f"Original:\n{text[:500]}\n"
                    f"Error: {exc}"
                ) from exc

            if not isinstance(payload, dict):
                raise ValueError(
                    f"Gemma output must be a JSON object, got type={type(payload)}"
                )

            return payload

    def _get_client(self):
        if self._client is None:
            try:
                from google import genai
            except ModuleNotFoundError as exc:  # pragma: no cover
                raise GemmaGenerationError(
                    "google-genai is required. Install dependencies with "
                    "`pip install -e .` or `pip install -r requirements.txt`."
                ) from exc
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _dry_run_response(self, prompt: str) -> str:
        lowered = prompt.lower()
        categories = []
        if "prompt injection" in lowered or "ignore previous" in lowered:
            categories.append("prompt_injection")
        if "password" in lowered or "credential" in lowered or "secret" in lowered:
            categories.append("privacy_credentials")
        if "phishing" in lowered or "fraud" in lowered:
            categories.append("fraud")
        if "malware" in lowered or "exploit" in lowered:
            categories.append("cyber_abuse")

        risk_level = "low"
        if len(categories) >= 2:
            risk_level = "high"
        elif categories:
            risk_level = "medium"

        return json.dumps(
            {
                "risk_level": risk_level,
                "categories": categories or ["no_obvious_policy_violation"],
                "evidence": [
                    "Dry-run heuristic response; configure Gemma API key for model output."
                ],
                "explanation": "Generated without network access for local verification.",
                "mitigations": [
                    "Run with Gemma 4 using configs/secrets.toml before final submission."
                ],
            }
        )


def parse_json_object(text: str) -> Dict[str, Any]:
    """Parse a JSON object, tolerating markdown fences around the object."""

    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.I).strip()
        stripped = re.sub(r"```$", "", stripped).strip()

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.S)
        if not match:
            raise
        payload = json.loads(match.group(0))

    if not isinstance(payload, dict):
        raise ValueError("Gemma output must be a JSON object")
    return payload
