"""Usage-driven context pruning for the CUAD contract QA dataset.

CUAD is distributed as SQuAD-style JSON. This module learns which contract
sections are repeatedly needed for correct answers, prunes low-utility sections,
and reports how context removal affects citation retention and answer coverage.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
from statistics import mean
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Set, Tuple
import zipfile


CUAD_DATA_URL = "https://github.com/TheAtticusProject/cuad/raw/main/data.zip"
DEFAULT_CUAD_MEMBER = "CUADv1.json"
TOKEN_RE = re.compile(r"[A-Za-z0-9_./%-]+")


@dataclass
class CuadSection:
    id: str
    contract_id: str
    text: str
    start_char: int
    end_char: int
    ordinal: int
    access_count: int = 0
    correct_access_count: int = 0
    last_access_index: int = -1
    utility_score: float = 0.0


@dataclass
class CuadQuestion:
    id: str
    contract_id: str
    question: str
    answers: List[Dict[str, Any]]
    is_impossible: bool = False

    @property
    def answer_texts(self) -> List[str]:
        return [str(answer.get("text", "")) for answer in self.answers if answer.get("text")]


@dataclass
class CuadContract:
    id: str
    title: str
    context: str
    sections: List[CuadSection]
    questions: List[CuadQuestion]


@dataclass
class SectionUsageStats:
    section_id: str
    access_count: int = 0
    correct_access_count: int = 0
    last_access_index: int = -1
    questions: Set[str] = field(default_factory=set)


@dataclass
class CuadEvaluationRow:
    prune_ratio: float
    contracts: int
    questions: int
    answerable_questions: int
    original_sections: int
    retained_sections: int
    context_removed_ratio: float
    citation_accuracy: float
    answer_quality: float
    citation_degradation: float
    answer_quality_degradation: float
    significant_degradation: bool


@dataclass
class CuadDetailRow:
    prune_ratio: float
    contract_id: str
    question_id: str
    gold_section_ids: str
    retained_gold_section_ids: str
    citation_preserved: bool
    answer_quality: float


class UsageDrivenContextPruner:
    """Learns and applies section utility from repeated correct-answer access."""

    def __init__(
        self,
        correct_weight: float = 3.0,
        access_weight: float = 1.0,
        recency_weight: float = 0.15,
        length_penalty: float = 0.05,
    ) -> None:
        self.correct_weight = correct_weight
        self.access_weight = access_weight
        self.recency_weight = recency_weight
        self.length_penalty = length_penalty
        self.stats: Dict[str, SectionUsageStats] = {}

    def observe_correct_answer(
        self,
        contract: CuadContract,
        question: CuadQuestion,
        question_index: int,
    ) -> None:
        """Record sections that contain gold answer spans for a correct answer."""

        for section in gold_answer_sections(contract, question):
            stats = self.stats.setdefault(
                section.id, SectionUsageStats(section_id=section.id)
            )
            stats.access_count += 1
            stats.correct_access_count += 1
            stats.last_access_index = question_index
            stats.questions.add(question.id)

    def score_section(self, section: CuadSection, current_index: int) -> float:
        stats = self.stats.get(section.id)
        if not stats:
            return -self.length_penalty * token_count(section.text)

        age = max(0, current_index - stats.last_access_index)
        recency = 1.0 / (1.0 + age)
        score = (
            self.correct_weight * stats.correct_access_count
            + self.access_weight * stats.access_count
            + self.recency_weight * recency
            - self.length_penalty * (token_count(section.text) / 100)
        )
        return score

    def rank_sections(
        self,
        sections: Sequence[CuadSection],
        current_index: int,
    ) -> List[CuadSection]:
        ranked: List[CuadSection] = []
        for section in sections:
            copy = CuadSection(**asdict(section))
            stats = self.stats.get(section.id)
            if stats:
                copy.access_count = stats.access_count
                copy.correct_access_count = stats.correct_access_count
                copy.last_access_index = stats.last_access_index
            copy.utility_score = self.score_section(section, current_index)
            ranked.append(copy)
        return sorted(ranked, key=lambda item: (item.utility_score, -item.ordinal), reverse=True)

    def retain_sections(
        self,
        sections: Sequence[CuadSection],
        prune_ratio: float,
        current_index: int,
    ) -> List[CuadSection]:
        if not 0 <= prune_ratio < 1:
            raise ValueError("prune_ratio must be in [0, 1)")
        if not sections:
            return []

        n_retain = max(1, round(len(sections) * (1 - prune_ratio)))
        retained = self.rank_sections(sections, current_index=current_index)[:n_retain]
        return sorted(retained, key=lambda item: item.ordinal)


def load_cuad_dataset(
    input_path: str | Path,
    json_member: str = DEFAULT_CUAD_MEMBER,
    max_contracts: int = 0,
    max_section_chars: int = 1200,
) -> List[CuadContract]:
    """Load CUAD contracts from a zip, JSON file, or directory containing JSON."""

    payload = load_cuad_payload(input_path, json_member=json_member)
    contracts: List[CuadContract] = []
    for doc_index, doc in enumerate(payload.get("data", [])):
        title = str(doc.get("title") or f"contract_{doc_index}")
        for paragraph_index, paragraph in enumerate(doc.get("paragraphs", [])):
            context = str(paragraph.get("context") or "")
            contract_id = normalize_id(f"{title}_{paragraph_index}")
            sections = split_contract_sections(
                contract_id=contract_id,
                context=context,
                max_section_chars=max_section_chars,
            )
            questions = [
                CuadQuestion(
                    id=str(qa.get("id") or f"{contract_id}_q{qa_index}"),
                    contract_id=contract_id,
                    question=str(qa.get("question") or ""),
                    answers=[
                        answer
                        for answer in qa.get("answers", [])
                        if isinstance(answer, dict)
                    ],
                    is_impossible=bool(qa.get("is_impossible", False)),
                )
                for qa_index, qa in enumerate(paragraph.get("qas", []))
            ]
            contracts.append(
                CuadContract(
                    id=contract_id,
                    title=title,
                    context=context,
                    sections=sections,
                    questions=questions,
                )
            )
            if max_contracts and len(contracts) >= max_contracts:
                return contracts
    return contracts


def load_cuad_payload(input_path: str | Path, json_member: str) -> Dict[str, Any]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(
            f"CUAD input not found: {path}. Download from {CUAD_DATA_URL} or pass a local data.zip."
        )

    if path.is_dir():
        candidates = [path / json_member, path / "CUADv1.json", path / "test.json"]
        for candidate in candidates:
            if candidate.exists():
                return json.loads(candidate.read_text(encoding="utf-8"))
        json_files = sorted(path.glob("*.json"))
        if json_files:
            return json.loads(json_files[0].read_text(encoding="utf-8"))
        raise FileNotFoundError(f"No CUAD JSON files found in {path}")

    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            member = json_member if json_member in names else ""
            if not member:
                for candidate in [DEFAULT_CUAD_MEMBER, "test.json", "train_separate_questions.json"]:
                    if candidate in names:
                        member = candidate
                        break
            if not member:
                raise FileNotFoundError(
                    f"No CUAD JSON member found in {path}; available={sorted(names)[:10]}"
                )
            return json.loads(archive.read(member).decode("utf-8"))

    return json.loads(path.read_text(encoding="utf-8"))


def split_contract_sections(
    contract_id: str,
    context: str,
    max_section_chars: int = 1200,
) -> List[CuadSection]:
    spans = paragraph_spans(context)
    sections: List[CuadSection] = []
    for start, end in spans:
        text = context[start:end].strip()
        if not text:
            continue
        if len(text) <= max_section_chars:
            sections.append(
                build_section(contract_id, context, start, end, len(sections))
            )
            continue
        for chunk_start, chunk_end in split_long_span(context, start, end, max_section_chars):
            sections.append(
                build_section(contract_id, context, chunk_start, chunk_end, len(sections))
            )
    if not sections and context.strip():
        sections.append(build_section(contract_id, context, 0, len(context), 0))
    return sections


def paragraph_spans(context: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    start = 0
    for match in re.finditer(r"\n\s*\n+", context):
        end = match.start()
        if context[start:end].strip():
            spans.append((start, end))
        start = match.end()
    if context[start:].strip():
        spans.append((start, len(context)))
    return spans or [(0, len(context))]


def split_long_span(
    context: str,
    start: int,
    end: int,
    max_section_chars: int,
) -> Iterator[Tuple[int, int]]:
    cursor = start
    while cursor < end:
        target = min(end, cursor + max_section_chars)
        if target < end:
            boundary = max(
                context.rfind(". ", cursor, target),
                context.rfind("; ", cursor, target),
                context.rfind("\n", cursor, target),
            )
            if boundary > cursor + max_section_chars // 2:
                target = boundary + 1
        yield cursor, target
        cursor = target
        while cursor < end and context[cursor].isspace():
            cursor += 1


def build_section(
    contract_id: str,
    context: str,
    start: int,
    end: int,
    ordinal: int,
) -> CuadSection:
    return CuadSection(
        id=f"{contract_id}:section:{ordinal}",
        contract_id=contract_id,
        text=context[start:end].strip(),
        start_char=start,
        end_char=end,
        ordinal=ordinal,
    )


def gold_answer_sections(contract: CuadContract, question: CuadQuestion) -> List[CuadSection]:
    section_ids: Set[str] = set()
    sections: List[CuadSection] = []
    for answer in question.answers:
        try:
            answer_start = int(answer.get("answer_start", -1))
        except (TypeError, ValueError):
            answer_start = -1
        answer_text = str(answer.get("text") or "")
        answer_end = answer_start + len(answer_text)
        for section in contract.sections:
            if answer_start >= 0 and spans_overlap(
                answer_start,
                answer_end,
                section.start_char,
                section.end_char,
            ):
                if section.id not in section_ids:
                    section_ids.add(section.id)
                    sections.append(section)
            elif answer_text and answer_text.lower() in section.text.lower():
                if section.id not in section_ids:
                    section_ids.add(section.id)
                    sections.append(section)
    return sections


def evaluate_usage_pruning(
    contracts: Sequence[CuadContract],
    prune_ratios: Sequence[float],
    train_fraction: float = 0.6,
    degradation_tolerance: float = 0.05,
) -> Tuple[List[CuadEvaluationRow], List[CuadDetailRow]]:
    pruner = UsageDrivenContextPruner()
    train_items: List[Tuple[CuadContract, CuadQuestion]] = []
    eval_items: List[Tuple[CuadContract, CuadQuestion]] = []

    for contract in contracts:
        answerable = [question for question in contract.questions if question.answer_texts]
        if not answerable:
            continue
        split_index = max(1, min(len(answerable), round(len(answerable) * train_fraction)))
        train_items.extend((contract, question) for question in answerable[:split_index])
        eval_items.extend((contract, question) for question in answerable[split_index:])
        if split_index == len(answerable):
            eval_items.append((contract, answerable[-1]))

    for index, (contract, question) in enumerate(train_items):
        pruner.observe_correct_answer(contract, question, question_index=index)

    baseline_details = evaluate_at_ratio(
        contracts,
        eval_items,
        pruner,
        prune_ratio=0.0,
        current_index=len(train_items),
    )
    baseline_citation = safe_mean([1.0 if row.citation_preserved else 0.0 for row in baseline_details])
    baseline_quality = safe_mean([row.answer_quality for row in baseline_details])

    rows: List[CuadEvaluationRow] = []
    details: List[CuadDetailRow] = []
    for prune_ratio in prune_ratios:
        ratio_details = evaluate_at_ratio(
            contracts,
            eval_items,
            pruner,
            prune_ratio=prune_ratio,
            current_index=len(train_items),
        )
        details.extend(ratio_details)
        citation_accuracy = safe_mean([1.0 if row.citation_preserved else 0.0 for row in ratio_details])
        answer_quality = safe_mean([row.answer_quality for row in ratio_details])
        original_sections = sum(len(contract.sections) for contract in contracts)
        retained_sections = retained_section_count(
            contracts,
            pruner,
            prune_ratio=prune_ratio,
            current_index=len(train_items),
        )
        citation_drop = max(0.0, baseline_citation - citation_accuracy)
        quality_drop = max(0.0, baseline_quality - answer_quality)
        rows.append(
            CuadEvaluationRow(
                prune_ratio=prune_ratio,
                contracts=len(contracts),
                questions=len(eval_items),
                answerable_questions=len(eval_items),
                original_sections=original_sections,
                retained_sections=retained_sections,
                context_removed_ratio=1 - safe_ratio(retained_sections, original_sections),
                citation_accuracy=citation_accuracy,
                answer_quality=answer_quality,
                citation_degradation=citation_drop,
                answer_quality_degradation=quality_drop,
                significant_degradation=(
                    citation_drop > degradation_tolerance
                    or quality_drop > degradation_tolerance
                ),
            )
        )
    return rows, details


def evaluate_at_ratio(
    contracts: Sequence[CuadContract],
    eval_items: Sequence[Tuple[CuadContract, CuadQuestion]],
    pruner: UsageDrivenContextPruner,
    prune_ratio: float,
    current_index: int,
) -> List[CuadDetailRow]:
    retained_by_contract = {
        contract.id: pruner.retain_sections(
            contract.sections,
            prune_ratio=prune_ratio,
            current_index=current_index,
        )
        for contract in contracts
    }
    rows: List[CuadDetailRow] = []
    for contract, question in eval_items:
        gold_sections = gold_answer_sections(contract, question)
        gold_ids = {section.id for section in gold_sections}
        retained = retained_by_contract.get(contract.id, [])
        retained_ids = {section.id for section in retained}
        retained_gold_ids = gold_ids & retained_ids
        answer_quality = max(
            [answer_quality_score(answer_text, retained) for answer_text in question.answer_texts]
            or [0.0]
        )
        rows.append(
            CuadDetailRow(
                prune_ratio=prune_ratio,
                contract_id=contract.id,
                question_id=question.id,
                gold_section_ids=";".join(sorted(gold_ids)),
                retained_gold_section_ids=";".join(sorted(retained_gold_ids)),
                citation_preserved=bool(retained_gold_ids),
                answer_quality=answer_quality,
            )
        )
    return rows


def retained_section_count(
    contracts: Sequence[CuadContract],
    pruner: UsageDrivenContextPruner,
    prune_ratio: float,
    current_index: int,
) -> int:
    return sum(
        len(
            pruner.retain_sections(
                contract.sections,
                prune_ratio=prune_ratio,
                current_index=current_index,
            )
        )
        for contract in contracts
    )


def write_csv(path: str | Path, rows: Sequence[Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_as_dicts = [asdict(row) for row in rows]
    if not rows_as_dicts:
        output_path.write_text("", encoding="utf-8")
        return
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_as_dicts[0].keys()))
        writer.writeheader()
        writer.writerows(rows_as_dicts)


def write_markdown_report(path: str | Path, rows: Sequence[CuadEvaluationRow]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    safe_rows = [row for row in rows if not row.significant_degradation]
    best_safe = max(safe_rows, key=lambda row: row.context_removed_ratio, default=None)
    lines = [
        "# CUAD Usage-Driven Context Pruning Report",
        "",
        "This offline evaluation learns which contract sections are repeatedly used",
        "for correct CUAD answers, prunes low-utility sections, and measures when",
        "citation accuracy or answer-span coverage degrades.",
        "",
    ]
    if best_safe:
        lines.extend(
            [
                "## Removal before significant degradation",
                "",
                (
                    f"Maximum safe context removal: **{best_safe.context_removed_ratio:.1%}** "
                    f"at prune_ratio={best_safe.prune_ratio:.2f}."
                ),
                "",
            ]
        )
    lines.extend(
        [
            "| Prune ratio | Removed | Citation accuracy | Answer quality | Significant degradation |",
            "|---:|---:|---:|---:|:---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row.prune_ratio:.2f} | {row.context_removed_ratio:.3f} | "
            f"{row.citation_accuracy:.3f} | {row.answer_quality:.3f} | "
            f"{'yes' if row.significant_degradation else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "- **Citation accuracy**: fraction of held-out answerable questions whose",
            "  gold answer section remains after pruning.",
            "- **Answer quality**: token-F1 proxy between gold answer spans and retained",
            "  context. It measures answer-span coverage without calling an LLM.",
            "- **Significant degradation**: citation or answer-quality drop above the",
            "  configured tolerance relative to no pruning.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_prune_ratios(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def answer_quality_score(answer: str, retained_sections: Sequence[CuadSection]) -> float:
    """Return answer-span coverage proxy over retained sections."""

    normalized_answer = answer.strip().lower()
    if not normalized_answer:
        return 0.0
    for section in retained_sections:
        if normalized_answer in section.text.lower():
            return 1.0
    retained_text = "\n".join(section.text for section in retained_sections)
    return token_recall(answer, retained_text)


def token_recall(answer: str, context: str) -> float:
    answer_tokens = token_set(answer)
    context_tokens = token_set(context)
    if not answer_tokens:
        return 0.0
    overlap = len(answer_tokens & context_tokens)
    return overlap / len(answer_tokens)


def token_set(text: str) -> Set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text) if token.strip()}


def token_count(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def spans_overlap(left_start: int, left_end: int, right_start: int, right_end: int) -> bool:
    return left_start < right_end and right_start < left_end


def safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def safe_mean(values: Iterable[float]) -> float:
    values_list = list(values)
    return mean(values_list) if values_list else 0.0


def normalize_id(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.:-]+", "_", value).strip("_")
    return normalized[:180] or "contract"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate usage-driven context pruning on CUAD."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to CUAD data.zip, CUAD JSON file, or directory containing CUAD JSON.",
    )
    parser.add_argument("--json-member", default=DEFAULT_CUAD_MEMBER)
    parser.add_argument("--max-contracts", type=int, default=0)
    parser.add_argument("--max-section-chars", type=int, default=1200)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--degradation-tolerance", type=float, default=0.05)
    parser.add_argument(
        "--prune-ratios",
        default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8",
        help="Comma-separated prune ratios to evaluate.",
    )
    parser.add_argument("--summary-output", default="outputs/cuad_summary.csv")
    parser.add_argument("--details-output", default="outputs/cuad_details.csv")
    parser.add_argument("--report-output", default="outputs/cuad_report.md")
    return parser


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    contracts = load_cuad_dataset(
        args.input,
        json_member=args.json_member,
        max_contracts=args.max_contracts,
        max_section_chars=args.max_section_chars,
    )
    if not contracts:
        raise RuntimeError("No CUAD contracts were loaded.")

    rows, details = evaluate_usage_pruning(
        contracts,
        prune_ratios=parse_prune_ratios(args.prune_ratios),
        train_fraction=args.train_fraction,
        degradation_tolerance=args.degradation_tolerance,
    )
    write_csv(args.summary_output, rows)
    write_csv(args.details_output, details)
    write_markdown_report(args.report_output, rows)

    best_safe = max(
        [row for row in rows if not row.significant_degradation],
        key=lambda row: row.context_removed_ratio,
        default=None,
    )
    print(
        json.dumps(
            {
                "contracts": len(contracts),
                "summary_output": args.summary_output,
                "details_output": args.details_output,
                "report_output": args.report_output,
                "max_safe_context_removed_ratio": (
                    best_safe.context_removed_ratio if best_safe else 0.0
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
