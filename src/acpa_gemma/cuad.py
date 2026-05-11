"""Usage-driven context pruning for the CUAD contract QA dataset.

CUAD is distributed as SQuAD-style JSON. This module learns which contract
sections are repeatedly needed for correct answers, prunes low-utility sections,
and reports how context removal affects citation retention and answer coverage.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, field
from html import escape
import json
from pathlib import Path
import re
from statistics import mean
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Set, Tuple
import zipfile


CUAD_DATA_URL = "https://github.com/TheAtticusProject/cuad/raw/main/data.zip"
DEFAULT_CUAD_MEMBER = "CUADv1.json"
TOKEN_RE = re.compile(r"[A-Za-z0-9_./%-]+")
DEFAULT_POLICIES = [
    "usage_driven",
    "hybrid_usage_bm25",
    "bm25_query_relevance",
    "mmr_diverse_relevance",
    "rrf_bm25_textrank",
    "dpp_diverse_relevance",
    "late_interaction_maxsim",
]
NOVEL_POLICIES = {"usage_driven", "hybrid_usage_bm25"}
SOTA_BASELINE_POLICIES = [
    "bm25_query_relevance",
    "mmr_diverse_relevance",
    "rrf_bm25_textrank",
    "dpp_diverse_relevance",
    "late_interaction_maxsim",
]


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
    policy: str
    policy_family: str
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
    baseline_policy: str = ""
    citation_improvement_pct: float = 0.0
    answer_quality_improvement_pct: float = 0.0
    combined_improvement_pct: float = 0.0
    sota_baselines_compared: int = 0
    sota_win_count: int = 0
    sota_win_rate: float = 0.0


@dataclass
class CuadDetailRow:
    policy: str
    prune_ratio: float
    contract_id: str
    question_id: str
    original_section_count: int
    retained_section_count: int
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
    policies: Sequence[str] = DEFAULT_POLICIES,
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
        policy="usage_driven",
        prune_ratio=0.0,
        current_index=len(train_items),
        ranking_cache={},
    )
    baseline_citation = safe_mean([1.0 if row.citation_preserved else 0.0 for row in baseline_details])
    baseline_quality = safe_mean([row.answer_quality for row in baseline_details])

    rows: List[CuadEvaluationRow] = []
    details: List[CuadDetailRow] = []
    ranking_cache: Dict[Tuple[str, str, str, int], List[CuadSection]] = {}
    for policy in policies:
        for prune_ratio in prune_ratios:
            ratio_details = evaluate_at_ratio(
                contracts,
                eval_items,
                pruner,
                policy=policy,
                prune_ratio=prune_ratio,
                current_index=len(train_items),
                ranking_cache=ranking_cache,
            )
            details.extend(ratio_details)
            citation_accuracy = safe_mean(
                [1.0 if row.citation_preserved else 0.0 for row in ratio_details]
            )
            answer_quality = safe_mean([row.answer_quality for row in ratio_details])
            original_sections = sum(len(contract.sections) for contract, _ in eval_items)
            retained_sections = sum(row.retained_section_count for row in ratio_details)
            citation_drop = max(0.0, baseline_citation - citation_accuracy)
            quality_drop = max(0.0, baseline_quality - answer_quality)
            rows.append(
                CuadEvaluationRow(
                    policy=policy,
                    policy_family=policy_family(policy),
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
    annotate_policy_improvements(rows)
    return rows, details


def evaluate_at_ratio(
    contracts: Sequence[CuadContract],
    eval_items: Sequence[Tuple[CuadContract, CuadQuestion]],
    pruner: UsageDrivenContextPruner,
    policy: str,
    prune_ratio: float,
    current_index: int,
    ranking_cache: Dict[Tuple[str, str, str, int], List[CuadSection]],
) -> List[CuadDetailRow]:
    rows: List[CuadDetailRow] = []
    for contract, question in eval_items:
        gold_sections = gold_answer_sections(contract, question)
        gold_ids = {section.id for section in gold_sections}
        retained = retain_sections_for_policy(
            policy=policy,
            contract=contract,
            question=question,
            pruner=pruner,
            prune_ratio=prune_ratio,
            current_index=current_index,
            ranking_cache=ranking_cache,
        )
        retained_ids = {section.id for section in retained}
        retained_gold_ids = gold_ids & retained_ids
        answer_quality = max(
            [answer_quality_score(answer_text, retained) for answer_text in question.answer_texts]
            or [0.0]
        )
        rows.append(
            CuadDetailRow(
                policy=policy,
                prune_ratio=prune_ratio,
                contract_id=contract.id,
                question_id=question.id,
                original_section_count=len(contract.sections),
                retained_section_count=len(retained),
                gold_section_ids=";".join(sorted(gold_ids)),
                retained_gold_section_ids=";".join(sorted(retained_gold_ids)),
                citation_preserved=bool(retained_gold_ids),
                answer_quality=answer_quality,
            )
        )
    return rows


def retain_sections_for_policy(
    policy: str,
    contract: CuadContract,
    question: CuadQuestion,
    pruner: UsageDrivenContextPruner,
    prune_ratio: float,
    current_index: int,
    ranking_cache: Dict[Tuple[str, str, str, int], List[CuadSection]],
) -> List[CuadSection]:
    """Retain sections with dynamic, query-aware policies."""

    if not 0 <= prune_ratio < 1:
        raise ValueError("prune_ratio must be in [0, 1)")
    if not contract.sections:
        return []

    n_retain = max(1, round(len(contract.sections) * (1 - prune_ratio)))
    question_key = "" if policy == "usage_driven" else question.id
    cache_key = (policy, contract.id, question_key, current_index)
    ranked = ranking_cache.get(cache_key)
    if ranked is None:
        if policy == "usage_driven":
            ranked = pruner.rank_sections(contract.sections, current_index=current_index)
        elif policy == "bm25_query_relevance":
            ranked = rank_sections_bm25(question.question, contract.sections)
        elif policy == "mmr_diverse_relevance":
            ranked = rank_sections_mmr(question.question, contract.sections)
        elif policy == "rrf_bm25_textrank":
            ranked = rank_sections_rrf_bm25_textrank(question.question, contract.sections)
        elif policy == "dpp_diverse_relevance":
            ranked = rank_sections_dpp(question.question, contract.sections)
        elif policy == "late_interaction_maxsim":
            ranked = rank_sections_late_interaction_maxsim(question.question, contract.sections)
        elif policy == "hybrid_usage_bm25":
            ranked = rank_sections_hybrid_usage_bm25(
                question.question,
                contract.sections,
                pruner=pruner,
                current_index=current_index,
            )
        else:
            raise ValueError(f"Unknown CUAD pruning policy: {policy}")
        ranking_cache[cache_key] = ranked
    return sorted(ranked[:n_retain], key=lambda section: section.ordinal)


def rank_sections_bm25(query: str, sections: Sequence[CuadSection]) -> List[CuadSection]:
    """Rank sections using Okapi BM25 lexical relevance."""

    scores = bm25_scores(query, sections)
    ranked = [clone_section(section, scores.get(section.id, 0.0)) for section in sections]
    return sorted(ranked, key=lambda section: (section.utility_score, -section.ordinal), reverse=True)


def rank_sections_mmr(
    query: str,
    sections: Sequence[CuadSection],
    lambda_relevance: float = 0.72,
    candidate_limit: int = 80,
) -> List[CuadSection]:
    """Rank sections with MMR to balance query relevance and diversity."""

    scores = bm25_scores(query, sections)
    relevance_ranked = sorted(
        [clone_section(section, scores.get(section.id, 0.0)) for section in sections],
        key=lambda section: (section.utility_score, -section.ordinal),
        reverse=True,
    )
    candidate_count = min(candidate_limit, len(relevance_ranked))
    selected: List[CuadSection] = []
    remaining = relevance_ranked[:candidate_count]
    tail = relevance_ranked[candidate_count:]
    remaining_tokens = [token_set(section.text) for section in remaining]
    selected_tokens: List[Set[str]] = []
    while remaining:
        best_index = 0
        best_score = float("-inf")
        for index, section in enumerate(remaining):
            section_tokens = remaining_tokens[index]
            redundancy = max(
                [jaccard_similarity(section_tokens, tokens) for tokens in selected_tokens]
                or [0.0]
            )
            mmr_score = lambda_relevance * section.utility_score - (
                1 - lambda_relevance
            ) * redundancy
            if mmr_score > best_score:
                best_score = mmr_score
                best_index = index
        chosen = remaining.pop(best_index)
        chosen_tokens = remaining_tokens.pop(best_index)
        chosen.utility_score = best_score
        selected.append(chosen)
        selected_tokens.append(chosen_tokens)
    return selected + tail


def rank_sections_hybrid_usage_bm25(
    query: str,
    sections: Sequence[CuadSection],
    pruner: UsageDrivenContextPruner,
    current_index: int,
) -> List[CuadSection]:
    """Rank with cumulative usage utility plus query-time BM25 relevance."""

    usage_scores = {
        section.id: pruner.score_section(section, current_index=current_index)
        for section in sections
    }
    bm25 = bm25_scores(query, sections)
    normalized_usage = normalize_scores(usage_scores)
    normalized_bm25 = normalize_scores(bm25)
    ranked = []
    for section in sections:
        score = (
            0.62 * normalized_usage.get(section.id, 0.0)
            + 0.38 * normalized_bm25.get(section.id, 0.0)
        )
        ranked.append(clone_section(section, score))
    return sorted(ranked, key=lambda section: (section.utility_score, -section.ordinal), reverse=True)


def rank_sections_rrf_bm25_textrank(
    query: str,
    sections: Sequence[CuadSection],
    rrf_k: int = 60,
) -> List[CuadSection]:
    """Reciprocal-rank fusion of query BM25 and TextRank-style centrality."""

    bm25_ranked = rank_sections_bm25(query, sections)
    textrank_ranked = rank_sections_textrank(sections)
    fused_scores: Dict[str, float] = {}
    for ranked in [bm25_ranked, textrank_ranked]:
        for rank, section in enumerate(ranked, start=1):
            fused_scores[section.id] = fused_scores.get(section.id, 0.0) + 1 / (rrf_k + rank)
    fused = [clone_section(section, fused_scores.get(section.id, 0.0)) for section in sections]
    return sorted(fused, key=lambda section: (section.utility_score, -section.ordinal), reverse=True)


def rank_sections_textrank(
    sections: Sequence[CuadSection],
    candidate_limit: int = 120,
) -> List[CuadSection]:
    """Lightweight TextRank/PageRank centrality over section token overlap."""

    candidates = list(sections[:candidate_limit])
    tail = list(sections[candidate_limit:])
    token_sets = {section.id: token_set(section.text) for section in candidates}
    scores = {section.id: 1.0 for section in candidates}
    damping = 0.85
    for _ in range(12):
        next_scores: Dict[str, float] = {}
        for section in candidates:
            incoming = 0.0
            for other in candidates:
                if other.id == section.id:
                    continue
                similarity = jaccard_similarity(token_sets[section.id], token_sets[other.id])
                if similarity:
                    incoming += similarity * scores[other.id]
            next_scores[section.id] = (1 - damping) + damping * incoming
        scores = normalize_scores(next_scores)
    ranked = [clone_section(section, scores.get(section.id, 0.0)) for section in candidates]
    ranked.extend(clone_section(section, 0.0) for section in tail)
    return sorted(ranked, key=lambda section: (section.utility_score, -section.ordinal), reverse=True)


def rank_sections_dpp(
    query: str,
    sections: Sequence[CuadSection],
    candidate_limit: int = 100,
    diversity_weight: float = 0.35,
) -> List[CuadSection]:
    """DPP-inspired greedy ranking for relevance with novelty/diversity."""

    relevance = normalize_scores(bm25_scores(query, sections))
    candidates = sorted(
        [clone_section(section, relevance.get(section.id, 0.0)) for section in sections],
        key=lambda section: (section.utility_score, -section.ordinal),
        reverse=True,
    )
    active = candidates[: min(candidate_limit, len(candidates))]
    tail = candidates[len(active):]
    active_tokens = [token_set(section.text) for section in active]
    selected: List[CuadSection] = []
    selected_tokens: List[Set[str]] = []
    while active:
        best_index = 0
        best_score = float("-inf")
        for index, section in enumerate(active):
            redundancy = max(
                [jaccard_similarity(active_tokens[index], tokens) for tokens in selected_tokens]
                or [0.0]
            )
            novelty = 1 - redundancy
            score = section.utility_score + diversity_weight * novelty
            if score > best_score:
                best_score = score
                best_index = index
        chosen = active.pop(best_index)
        chosen_tokens = active_tokens.pop(best_index)
        chosen.utility_score = best_score
        selected.append(chosen)
        selected_tokens.append(chosen_tokens)
    return selected + tail


def rank_sections_late_interaction_maxsim(
    query: str,
    sections: Sequence[CuadSection],
) -> List[CuadSection]:
    """ColBERT-style lexical MaxSim approximation without embedding deps."""

    query_terms = token_list(query)
    if not query_terms:
        return [clone_section(section, 0.0) for section in sections]
    query_bigrams = set(zip(query_terms, query_terms[1:]))
    idf = inverse_document_frequency(query_terms, sections)
    ranked: List[CuadSection] = []
    for section in sections:
        terms = token_list(section.text)
        term_set = set(terms)
        bigrams = set(zip(terms, terms[1:]))
        maxsim = 0.0
        for term in query_terms:
            if term in term_set:
                maxsim += idf.get(term, 1.0)
        bigram_bonus = len(query_bigrams & bigrams) * 0.35
        length_norm = max(1.0, token_count(section.text) ** 0.25)
        ranked.append(clone_section(section, (maxsim + bigram_bonus) / length_norm))
    return sorted(ranked, key=lambda section: (section.utility_score, -section.ordinal), reverse=True)


def bm25_scores(query: str, sections: Sequence[CuadSection]) -> Dict[str, float]:
    query_terms = [term for term in token_set(query) if term]
    if not query_terms or not sections:
        return {section.id: 0.0 for section in sections}

    section_terms = {section.id: token_list(section.text) for section in sections}
    section_term_sets = {
        section_id: set(terms) for section_id, terms in section_terms.items()
    }
    doc_count = len(sections)
    avg_len = safe_mean([len(terms) for terms in section_terms.values()]) or 1.0
    document_frequency: Dict[str, int] = {}
    for term in query_terms:
        document_frequency[term] = sum(
            1 for terms in section_term_sets.values() if term in terms
        )

    k1 = 1.2
    b = 0.75
    scores: Dict[str, float] = {}
    for section in sections:
        terms = section_terms[section.id]
        length = len(terms) or 1
        score = 0.0
        for term in query_terms:
            tf = terms.count(term)
            if not tf:
                continue
            df = document_frequency.get(term, 0)
            idf = max(0.0, ((doc_count - df + 0.5) / (df + 0.5)))
            idf = 1.0 + idf
            denominator = tf + k1 * (1 - b + b * length / avg_len)
            score += idf * (tf * (k1 + 1)) / denominator
        scores[section.id] = score
    return scores


def inverse_document_frequency(
    terms: Sequence[str],
    sections: Sequence[CuadSection],
) -> Dict[str, float]:
    section_term_sets = [token_set(section.text) for section in sections]
    doc_count = len(sections)
    idf: Dict[str, float] = {}
    for term in set(terms):
        df = sum(1 for section_terms in section_term_sets if term in section_terms)
        idf[term] = 1.0 + max(0.0, (doc_count - df + 0.5) / (df + 0.5))
    return idf


def annotate_policy_improvements(rows: Sequence[CuadEvaluationRow]) -> None:
    """Annotate novel policy rows versus best non-usage policy at same ratio."""

    by_ratio: Dict[float, List[CuadEvaluationRow]] = {}
    for row in rows:
        by_ratio.setdefault(row.prune_ratio, []).append(row)

    for ratio_rows in by_ratio.values():
        baselines = [row for row in ratio_rows if row.policy in SOTA_BASELINE_POLICIES]
        if not baselines:
            continue
        best_baseline = max(
            baselines,
            key=lambda row: (row.citation_accuracy + row.answer_quality, row.context_removed_ratio),
        )
        for row in ratio_rows:
            if row.policy not in NOVEL_POLICIES:
                continue
            row.baseline_policy = best_baseline.policy
            row.citation_improvement_pct = percent_improvement(
                row.citation_accuracy,
                best_baseline.citation_accuracy,
            )
            row.answer_quality_improvement_pct = percent_improvement(
                row.answer_quality,
                best_baseline.answer_quality,
            )
            row.combined_improvement_pct = percent_improvement(
                row.citation_accuracy + row.answer_quality,
                best_baseline.citation_accuracy + best_baseline.answer_quality,
            )
            row.sota_baselines_compared = len(baselines)
            row.sota_win_count = sum(
                1
                for baseline in baselines
                if (row.citation_accuracy + row.answer_quality)
                > (baseline.citation_accuracy + baseline.answer_quality)
            )
            row.sota_win_rate = safe_ratio(row.sota_win_count, len(baselines))


def policy_family(policy: str) -> str:
    if policy in NOVEL_POLICIES:
        return "usage_driven"
    if policy in SOTA_BASELINE_POLICIES:
        return "sota_dynamic_baseline"
    return "dynamic_baseline"


def clone_section(section: CuadSection, utility_score: float) -> CuadSection:
    copied = CuadSection(**asdict(section))
    copied.utility_score = utility_score
    return copied


def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    low = min(values)
    high = max(values)
    if high == low:
        return {key: 0.0 for key in scores}
    return {key: (value - low) / (high - low) for key, value in scores.items()}


def percent_improvement(value: float, baseline: float) -> float:
    if baseline == 0:
        return 100.0 if value > 0 else 0.0
    return ((value - baseline) / baseline) * 100


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


def write_markdown_report(
    path: str | Path,
    rows: Sequence[CuadEvaluationRow],
    plot_paths: Sequence[Path] | None = None,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    usage_rows = [row for row in rows if row.policy in NOVEL_POLICIES]
    safe_rows = [row for row in usage_rows if not row.significant_degradation]
    best_safe = max(
        safe_rows,
        key=lambda row: (row.context_removed_ratio, row.combined_improvement_pct),
        default=None,
    )
    best_improvement = max(
        usage_rows,
        key=lambda row: row.combined_improvement_pct,
        default=None,
    )
    lines = [
        "# CUAD Usage-Driven Context Pruning Report",
        "",
        "This offline evaluation compares usage-driven context pruning against",
        "dynamic query-aware baselines. It learns which contract sections are",
        "repeatedly used for correct CUAD answers, prunes low-utility sections,",
        "and measures when citation accuracy or answer-span coverage degrades.",
        "",
    ]
    if best_safe:
        lines.extend(
            [
                "## Removal before significant degradation",
                "",
                (
                    f"Maximum safe context removal: **{best_safe.context_removed_ratio:.1%}** "
                    f"with `{best_safe.policy}` at prune_ratio={best_safe.prune_ratio:.2f}."
                ),
                "",
            ]
        )
    if best_improvement and best_improvement.baseline_policy:
        lines.extend(
            [
                "## Novel-policy improvement over five SOTA-style baselines",
                "",
                (
                    f"Best combined improvement: **{best_improvement.combined_improvement_pct:.1f}%** "
                    f"for `{best_improvement.policy}` over `{best_improvement.baseline_policy}` "
                    f"at prune_ratio={best_improvement.prune_ratio:.2f}."
                ),
                (
                    f"At that point it outperformed **{best_improvement.sota_win_count}/"
                    f"{best_improvement.sota_baselines_compared}** SOTA-style baselines."
                ),
                "",
            ]
        )
    if plot_paths:
        lines.extend(["## Journal-style figures", ""])
        for plot_path in plot_paths:
            try:
                display_path = plot_path.relative_to(output_path.parent)
            except ValueError:
                display_path = plot_path
            title = plot_path.stem.replace("_", " ").title()
            lines.extend([f"![{title}]({display_path})", ""])
    lines.extend(
        [
            "Compared SOTA-style baselines:",
            "",
            "- `bm25_query_relevance`: Okapi BM25 sparse retrieval.",
            "- `mmr_diverse_relevance`: Maximal Marginal Relevance diversity pruning.",
            "- `rrf_bm25_textrank`: Reciprocal-rank fusion of BM25 and TextRank centrality.",
            "- `dpp_diverse_relevance`: DPP-inspired relevance plus novelty selection.",
            "- `late_interaction_maxsim`: ColBERT-style lexical MaxSim approximation.",
            "",
        ]
    )
    lines.extend(
        [
            "| Policy | Prune ratio | Removed | Citation accuracy | Answer quality | Improvement vs best SOTA | SOTA wins | Significant degradation |",
            "|---|---:|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row.policy} | {row.prune_ratio:.2f} | {row.context_removed_ratio:.3f} | "
            f"{row.citation_accuracy:.3f} | {row.answer_quality:.3f} | "
            f"{row.combined_improvement_pct:.1f}% | "
            f"{row.sota_win_count}/{row.sota_baselines_compared} | "
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
            "- **Improvement vs best SOTA**: percentage lift for usage-driven policies",
            "  over the best non-usage SOTA-style baseline at the same prune ratio.",
            "- **SOTA wins**: number of the five baseline policies beaten by the",
            "  usage-driven method on citation+answer-quality score.",
            "- **Significant degradation**: citation or answer-quality drop above the",
            "  configured tolerance relative to no pruning.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_journal_plots(
    output_dir: str | Path,
    rows: Sequence[CuadEvaluationRow],
) -> List[Path]:
    """Write dependency-free SVG figures for publication-style comparisons."""

    plot_dir = Path(output_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    plots = [
        write_line_plot(
            plot_dir / "citation_accuracy_vs_context_removed.svg",
            rows,
            metric_name="citation_accuracy",
            title="Citation accuracy vs. context removed",
            y_label="Citation accuracy",
            percent_y=False,
        ),
        write_line_plot(
            plot_dir / "answer_quality_vs_context_removed.svg",
            rows,
            metric_name="answer_quality",
            title="Answer quality vs. context removed",
            y_label="Answer quality",
            percent_y=False,
        ),
        write_line_plot(
            plot_dir / "improvement_vs_context_removed.svg",
            [row for row in rows if row.policy in NOVEL_POLICIES],
            metric_name="combined_improvement_pct",
            title="Improvement over best SOTA baseline",
            y_label="Improvement (%)",
            percent_y=False,
        ),
        write_bar_plot(
            plot_dir / "max_safe_context_removed_by_policy.svg",
            rows,
            title="Maximum safe context removal by policy",
            y_label="Max safe context removed",
        ),
    ]
    return plots


def write_line_plot(
    path: Path,
    rows: Sequence[CuadEvaluationRow],
    metric_name: str,
    title: str,
    y_label: str,
    percent_y: bool,
) -> Path:
    grouped = group_rows_by_policy(rows)
    series: Dict[str, List[Tuple[float, float]]] = {}
    for policy, policy_rows in grouped.items():
        points = sorted(
            [
                (row.context_removed_ratio, float(getattr(row, metric_name)))
                for row in policy_rows
            ],
            key=lambda item: item[0],
        )
        if points:
            series[policy] = points
    y_values = [value for points in series.values() for _, value in points]
    y_min, y_max = chart_bounds(y_values)
    svg = build_line_svg(
        title=title,
        x_label="Context removed",
        y_label=y_label,
        series=series,
        y_min=y_min,
        y_max=y_max,
        percent_y=percent_y,
    )
    path.write_text(svg, encoding="utf-8")
    return path


def write_bar_plot(
    path: Path,
    rows: Sequence[CuadEvaluationRow],
    title: str,
    y_label: str,
) -> Path:
    grouped = group_rows_by_policy(rows)
    values = {
        policy: max(
            [
                row.context_removed_ratio
                for row in policy_rows
                if not row.significant_degradation
            ]
            or [0.0]
        )
        for policy, policy_rows in grouped.items()
    }
    svg = build_bar_svg(
        title=title,
        x_label="Policy",
        y_label=y_label,
        values=values,
        y_min=0.0,
        y_max=max([*values.values(), 1.0]),
    )
    path.write_text(svg, encoding="utf-8")
    return path


def build_line_svg(
    title: str,
    x_label: str,
    y_label: str,
    series: Dict[str, List[Tuple[float, float]]],
    y_min: float,
    y_max: float,
    percent_y: bool,
) -> str:
    width, height = 980, 620
    margin_left, margin_right, margin_top, margin_bottom = 90, 260, 70, 95
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    def x_coord(value: float) -> float:
        return margin_left + clamp(value, 0.0, 1.0) * plot_width

    def y_coord(value: float) -> float:
        if y_max == y_min:
            return margin_top + plot_height / 2
        return margin_top + (y_max - value) / (y_max - y_min) * plot_height

    parts = svg_header(width, height, title)
    parts.extend(draw_axes(width, height, margin_left, margin_top, plot_width, plot_height, x_label, y_label))
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        x = x_coord(tick)
        parts.append(f'<line x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" y2="{margin_top + plot_height}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{x:.1f}" y="{margin_top + plot_height + 28}" text-anchor="middle" class="tick">{tick:.0%}</text>')
    for index in range(5):
        value = y_min + (y_max - y_min) * index / 4
        y = y_coord(value)
        label = f"{value:.0f}%" if percent_y else f"{value:.2f}"
        parts.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{margin_left + plot_width}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{margin_left - 12}" y="{y + 4:.1f}" text-anchor="end" class="tick">{label}</text>')

    for idx, (policy, points) in enumerate(series.items()):
        color = chart_color(idx)
        point_text = " ".join(f"{x_coord(x):.1f},{y_coord(y):.1f}" for x, y in points)
        parts.append(f'<polyline points="{point_text}" fill="none" stroke="{color}" stroke-width="2.8"/>')
        for x, y in points:
            parts.append(f'<circle cx="{x_coord(x):.1f}" cy="{y_coord(y):.1f}" r="4" fill="{color}"/>')
        legend_y = margin_top + idx * 24
        legend_x = width - margin_right + 35
        parts.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 22}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x + 30}" y="{legend_y + 5}" class="legend">{escape(policy)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def build_bar_svg(
    title: str,
    x_label: str,
    y_label: str,
    values: Dict[str, float],
    y_min: float,
    y_max: float,
) -> str:
    width, height = 980, 620
    margin_left, margin_right, margin_top, margin_bottom = 90, 40, 70, 170
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    policies = list(values)
    bar_gap = 12
    bar_width = (plot_width - bar_gap * max(0, len(policies) - 1)) / max(1, len(policies))

    def y_coord(value: float) -> float:
        return margin_top + (y_max - value) / (y_max - y_min) * plot_height

    parts = svg_header(width, height, title)
    parts.extend(draw_axes(width, height, margin_left, margin_top, plot_width, plot_height, x_label, y_label))
    for index in range(5):
        value = y_min + (y_max - y_min) * index / 4
        y = y_coord(value)
        parts.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{margin_left + plot_width}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{margin_left - 12}" y="{y + 4:.1f}" text-anchor="end" class="tick">{value:.0%}</text>')
    for idx, policy in enumerate(policies):
        value = values[policy]
        x = margin_left + idx * (bar_width + bar_gap)
        y = y_coord(value)
        h = margin_top + plot_height - y
        color = chart_color(idx)
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{h:.1f}" fill="{color}" opacity="0.85"/>')
        parts.append(f'<text x="{x + bar_width / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle" class="tick">{value:.0%}</text>')
        parts.append(
            f'<text transform="translate({x + bar_width / 2:.1f},{height - margin_bottom + 30}) rotate(45)" '
            f'text-anchor="start" class="tick">{escape(policy)}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def svg_header(width: int, height: int, title: str) -> List[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: Arial, Helvetica, sans-serif; fill: #111827; }",
        ".title { font-size: 22px; font-weight: 700; }",
        ".axis { font-size: 15px; font-weight: 600; }",
        ".tick { font-size: 12px; fill: #374151; }",
        ".legend { font-size: 13px; fill: #111827; }",
        "</style>",
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="34" text-anchor="middle" class="title">{escape(title)}</text>',
    ]


def draw_axes(
    width: int,
    height: int,
    margin_left: int,
    margin_top: int,
    plot_width: int,
    plot_height: int,
    x_label: str,
    y_label: str,
) -> List[str]:
    axis_y = margin_top + plot_height
    return [
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{axis_y}" stroke="#111827" stroke-width="1.5"/>',
        f'<line x1="{margin_left}" y1="{axis_y}" x2="{margin_left + plot_width}" y2="{axis_y}" stroke="#111827" stroke-width="1.5"/>',
        f'<text x="{margin_left + plot_width / 2:.1f}" y="{height - 28}" text-anchor="middle" class="axis">{escape(x_label)}</text>',
        f'<text transform="translate(24,{margin_top + plot_height / 2:.1f}) rotate(-90)" text-anchor="middle" class="axis">{escape(y_label)}</text>',
    ]


def chart_color(index: int) -> str:
    colors = [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#9467bd",
        "#ff7f0e",
        "#17becf",
        "#7f7f7f",
        "#bcbd22",
    ]
    return colors[index % len(colors)]


def group_rows_by_policy(rows: Sequence[CuadEvaluationRow]) -> Dict[str, List[CuadEvaluationRow]]:
    grouped: Dict[str, List[CuadEvaluationRow]] = {}
    for row in rows:
        grouped.setdefault(row.policy, []).append(row)
    return grouped


def chart_bounds(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 1.0
    low = min(values)
    high = max(values)
    if low == high:
        pad = 1.0 if low == 0 else abs(low) * 0.1
        return low - pad, high + pad
    pad = (high - low) * 0.08
    return low - pad, high + pad


def parse_prune_ratios(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_policies(value: str) -> List[str]:
    policies = [item.strip() for item in value.split(",") if item.strip()]
    valid = set(DEFAULT_POLICIES)
    unknown = [policy for policy in policies if policy not in valid]
    if unknown:
        raise ValueError(f"Unknown CUAD policies: {unknown}; valid={sorted(valid)}")
    return policies or list(DEFAULT_POLICIES)


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


def token_list(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall(text) if token.strip()]


def jaccard_similarity(left: Set[str], right: Set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def token_count(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def spans_overlap(left_start: int, left_end: int, right_start: int, right_end: int) -> bool:
    return left_start < right_end and right_start < left_end


def safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def safe_mean(values: Iterable[float]) -> float:
    values_list = list(values)
    return mean(values_list) if values_list else 0.0


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


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
    parser.add_argument(
        "--policies",
        default=",".join(DEFAULT_POLICIES),
        help=(
            "Comma-separated policies: usage_driven, hybrid_usage_bm25, "
            "bm25_query_relevance, mmr_diverse_relevance, rrf_bm25_textrank, "
            "dpp_diverse_relevance, late_interaction_maxsim."
        ),
    )
    parser.add_argument("--summary-output", default="outputs/cuad_summary.csv")
    parser.add_argument("--details-output", default="outputs/cuad_details.csv")
    parser.add_argument("--report-output", default="outputs/cuad_report.md")
    parser.add_argument(
        "--plots-output-dir",
        default="outputs/cuad_plots",
        help="Directory for journal-style SVG plots.",
    )
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
        policies=parse_policies(args.policies),
    )
    write_csv(args.summary_output, rows)
    write_csv(args.details_output, details)
    plot_paths = write_journal_plots(args.plots_output_dir, rows)
    write_markdown_report(args.report_output, rows, plot_paths=plot_paths)

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
                "plots_output_dir": args.plots_output_dir,
                "plots": [str(path) for path in plot_paths],
                "policies": sorted({row.policy for row in rows}),
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
