"""Agentic Eval dataset loading and context construction."""

from __future__ import annotations

from dataclasses import dataclass, field
import csv
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set

from acpa_gemma.acpa import ContextElement

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover
    pd = None  # type: ignore[assignment]


SUPPORTED_SUFFIXES = {".csv", ".json", ".jsonl", ".ndjson", ".parquet"}
JSON_ROW_KEYS = ["data", "records", "examples", "rows", "items"]

PROMPT_FIELDS = ["prompt", "instruction", "query", "question", "user_prompt", "task"]
RESPONSE_FIELDS = ["response", "answer", "completion", "output", "assistant_response"]
TRACE_FIELDS = ["trajectory", "trace", "conversation", "messages", "tool_calls", "steps"]
LABEL_FIELDS = ["label", "risk", "safety_label", "category", "target"]

SAFETY_KEYWORDS = {
    "abuse",
    "attack",
    "bypass",
    "credential",
    "cyber",
    "deception",
    "exploit",
    "fraud",
    "harm",
    "harassment",
    "hate",
    "jailbreak",
    "malware",
    "password",
    "phishing",
    "policy",
    "privacy",
    "prompt injection",
    "secret",
    "self-harm",
    "sexual",
    "violence",
    "weapon",
}


@dataclass
class AgenticEvalRecord:
    """Normalized Agentic Eval row."""

    record_id: str
    prompt: str = ""
    response: str = ""
    trajectory: str = ""
    label: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)
    source_path: str = ""

    def combined_text(self) -> str:
        sections = [
            ("prompt", self.prompt),
            ("trajectory", self.trajectory),
            ("response", self.response),
            ("label", self.label),
        ]
        return "\n\n".join(f"{name}: {value}" for name, value in sections if value)


def load_agentic_eval_dataset(
    input_dir: str | Path,
    sample_size: int = 0,
) -> List[AgenticEvalRecord]:
    """Discover and load Agentic Eval records from a Kaggle input directory."""

    root = Path(input_dir)
    if not root.exists():
        return []

    records: List[AgenticEvalRecord] = []
    for data_file in discover_dataset_files(root):
        try:
            for row_index, row in enumerate(read_rows(data_file)):
                record = normalize_row(row, row_index=row_index, source_path=data_file)
                if record.combined_text().strip() or record.raw:
                    records.append(record)
                if sample_size and len(records) >= sample_size:
                    return records
        except Exception:
            # Kaggle inputs can include metadata files with supported extensions.
            # Skip unreadable files and let diagnostics explain what was found.
            continue
    return records


def discover_dataset_files(input_dir: str | Path) -> List[Path]:
    root = Path(input_dir)
    if root.is_file() and root.suffix.lower() in SUPPORTED_SUFFIXES:
        return [root]

    return sorted(
        [
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        ]
    )


def read_rows(path: Path) -> Iterator[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if pd is not None and suffix in {".csv", ".json", ".jsonl", ".ndjson", ".parquet"}:
        if suffix == ".csv":
            frame = pd.read_csv(path)
        elif suffix == ".parquet":
            frame = pd.read_parquet(path)
        else:
            frame = pd.read_json(path, lines=suffix in {".jsonl", ".ndjson"})
        for row in frame.to_dict(orient="records"):
            yield {str(key): value for key, value in row.items()}
        return

    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            yield from csv.DictReader(handle)
    elif suffix in {".jsonl", ".ndjson"}:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
    elif suffix == ".json":
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        rows = extract_json_rows(payload)
        for row in rows:
            if isinstance(row, dict):
                yield row
    else:
        raise ValueError(f"Unsupported dataset file without pandas: {path}")


def extract_json_rows(payload: Any) -> List[Dict[str, Any]]:
    """Extract row dictionaries from common JSON dataset shapes."""

    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return []
    for key in JSON_ROW_KEYS:
        value = payload.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
    # Treat a single JSON object as a one-row dataset.
    return [payload] if payload else []


def dataset_diagnostics(input_dir: str | Path, max_files: int = 25) -> Dict[str, Any]:
    """Return human-readable diagnostics for Kaggle dataset loading."""

    root = Path(input_dir)
    exists = root.exists()
    all_files: List[Path] = []
    if exists:
        if root.is_file():
            all_files = [root]
        else:
            all_files = sorted([path for path in root.rglob("*") if path.is_file()])
    supported_files = [
        path for path in all_files if path.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    return {
        "input_dir": str(root),
        "exists": exists,
        "total_files": len(all_files),
        "supported_files": len(supported_files),
        "supported_suffixes": sorted(SUPPORTED_SUFFIXES),
        "sample_supported_files": [str(path) for path in supported_files[:max_files]],
        "sample_all_files": [str(path) for path in all_files[:max_files]],
    }


def format_dataset_diagnostics(diagnostics: Dict[str, Any]) -> str:
    lines = [
        f"input_dir={diagnostics['input_dir']}",
        f"exists={diagnostics['exists']}",
        f"total_files={diagnostics['total_files']}",
        f"supported_files={diagnostics['supported_files']}",
        f"supported_suffixes={diagnostics['supported_suffixes']}",
        "sample_supported_files:",
    ]
    lines.extend(f"  - {path}" for path in diagnostics["sample_supported_files"])
    if not diagnostics["sample_supported_files"]:
        lines.append("  (none)")
    lines.append("sample_all_files:")
    lines.extend(f"  - {path}" for path in diagnostics["sample_all_files"])
    if not diagnostics["sample_all_files"]:
        lines.append("  (none)")
    return "\n".join(lines)


def normalize_row(
    row: Dict[str, Any],
    row_index: int,
    source_path: str | Path,
) -> AgenticEvalRecord:
    lower_map = {key.lower(): key for key in row}

    def pick(fields: Sequence[str]) -> str:
        for field_name in fields:
            original = lower_map.get(field_name.lower())
            if original is not None:
                return stringify(row.get(original))
        return ""

    record_id = pick(["id", "record_id", "example_id", "sample_id"])
    if not record_id:
        record_id = f"{Path(source_path).stem}_{row_index}"

    return AgenticEvalRecord(
        record_id=record_id,
        prompt=pick(PROMPT_FIELDS),
        response=pick(RESPONSE_FIELDS),
        trajectory=pick(TRACE_FIELDS),
        label=pick(LABEL_FIELDS),
        raw=row,
        source_path=str(source_path),
    )


def build_context_elements(
    record: AgenticEvalRecord,
    timestamp: int = 0,
    max_chunk_chars: int = 900,
) -> List[ContextElement]:
    """Convert a normalized record into scored ACPA context elements."""

    elements: List[ContextElement] = []
    fields = {
        "prompt": record.prompt,
        "trajectory": record.trajectory,
        "response": record.response,
        "label": record.label,
    }
    for field_name, value in fields.items():
        for chunk_index, chunk in enumerate(chunk_text(value, max_chunk_chars=max_chunk_chars)):
            element_id = f"{record.record_id}:{field_name}:{chunk_index}"
            elements.append(
                ContextElement(
                    id=element_id,
                    text=chunk,
                    source_doc=field_name,
                    timestamp=timestamp,
                    importance_score=estimate_importance(chunk),
                    metadata={
                        "record_id": record.record_id,
                        "field": field_name,
                        "chunk_index": chunk_index,
                    },
                )
            )
    return elements


def extract_citations(record: AgenticEvalRecord, context: Iterable[ContextElement]) -> Set[str]:
    """Extract evidence markers that should be pinned during pruning."""

    citations: Set[str] = set()
    citation_patterns = [
        r"\[[A-Za-z0-9_.:-]{1,40}\]",
        r"\bCVE-\d{4}-\d{4,7}\b",
        r"https?://[^\s)]+",
        r"\b(?:tool|step|turn|message)[ _-]?\d+\b",
    ]
    for text in [record.combined_text(), *(element.text for element in context)]:
        for pattern in citation_patterns:
            citations.update(match.group(0) for match in re.finditer(pattern, text, re.I))

    # Pin explicit labels and source fields when present; they are compact but
    # important for explaining the final safety judgment.
    if record.label:
        citations.add(record.label[:80])
    return {citation for citation in citations if citation}


def chunk_text(text: str, max_chunk_chars: int = 900) -> List[str]:
    text = stringify(text).strip()
    if not text:
        return []

    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n{2,}", text) if paragraph.strip()]
    chunks: List[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 2 <= max_chunk_chars:
            current = f"{current}\n\n{paragraph}".strip()
            continue
        if current:
            chunks.append(current)
        if len(paragraph) <= max_chunk_chars:
            current = paragraph
        else:
            chunks.extend(split_long_text(paragraph, max_chunk_chars))
            current = ""
    if current:
        chunks.append(current)
    return chunks


def split_long_text(text: str, max_chunk_chars: int) -> List[str]:
    words = text.split()
    chunks: List[str] = []
    current_words: List[str] = []
    current_len = 0
    for word in words:
        if current_words and current_len + len(word) + 1 > max_chunk_chars:
            chunks.append(" ".join(current_words))
            current_words = []
            current_len = 0
        current_words.append(word)
        current_len += len(word) + 1
    if current_words:
        chunks.append(" ".join(current_words))
    return chunks


def estimate_importance(text: str) -> float:
    """Lightweight importance score for safety-relevant context."""

    normalized = text.lower()
    keyword_hits = sum(1 for keyword in SAFETY_KEYWORDS if keyword in normalized)
    density = min(1.0, len(set(normalized.split())) / 120)
    citation_bonus = 0.2 if re.search(r"\[[^\]]+\]|https?://|CVE-\d{4}-", text) else 0.0
    keyword_score = min(0.8, keyword_hits * 0.16)
    return min(1.0, 0.15 + density * 0.35 + keyword_score + citation_bonus)


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd is not None and pd.isna(value):
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=True)
    return str(value)
