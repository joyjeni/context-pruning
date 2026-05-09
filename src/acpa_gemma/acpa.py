"""Adaptive Context Pruning Algorithm (ACPA).

This module implements the LFU/LRU hybrid cache eviction algorithm for Trust &
Safety research workflows. Context is represented as small elements
that can be accessed by Gemma agent stages, scored, pinned, retained, or
evicted.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import logging
import math
import re
from statistics import mean
from typing import DefaultDict, Dict, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


TOKEN_RE = re.compile(r"[A-Za-z0-9_#@./:-]+")


def _tokens(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


@dataclass
class ContextElement:
    """Represents a single piece of context.

    Attributes:
        id: Unique identifier.
        text: The actual content.
        source_doc: Source document or field name.
        timestamp: Pipeline stage number.
        access_count: How many times agents accessed this element.
        importance_score: TF-IDF, semantic, or heuristic importance in [0, 1].
        in_dependency_graph: Whether this is pinned memory.
        cache_priority: LFU weight multiplier.
        embedding: Optional semantic embedding vector.
        metadata: Additional metadata for traceability.
    """

    id: str
    text: str
    source_doc: str = ""
    timestamp: int = 0
    access_count: int = 0
    importance_score: float = 0.0
    in_dependency_graph: bool = False
    cache_priority: float = 1.0
    embedding: List[float] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"ContextElement(id={self.id!r}, access={self.access_count}, "
            f"importance={self.importance_score:.2f})"
        )


class AccessPatternTracker:
    """Tracks which agents access which context elements.

    The access matrix is the LFU foundation: each context hit increases an
    element's retention score during competitive eviction.
    """

    def __init__(self) -> None:
        self.access_matrix: DefaultDict[str, DefaultDict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.agent_outputs: Dict[str, str] = {}
        self.access_history: List[Dict[str, object]] = []

    def track_access(
        self,
        context_element: ContextElement,
        agent_id: str,
        agent_output: str,
        use_semantic: bool = False,
    ) -> bool:
        """Track whether an agent referenced a context element.

        Keyword matching is intentionally lightweight for Kaggle notebooks.
        Semantic matching can be plugged in later by supplying embeddings.
        """

        if not context_element.text or not agent_output:
            self.agent_outputs[agent_id] = agent_output
            return False

        if use_semantic and context_element.embedding:
            referenced = self._semantic_match(context_element, agent_output)
        else:
            referenced = self._keyword_match(context_element.text, agent_output)

        if referenced:
            self.access_matrix[context_element.id][agent_id] += 1
            context_element.access_count += 1
            self.access_history.append(
                {
                    "context_id": context_element.id,
                    "agent_id": agent_id,
                    "timestamp": context_element.timestamp,
                }
            )
            logger.debug("Cache hit: agent %s accessed %s", agent_id, context_element.id)

        self.agent_outputs[agent_id] = agent_output
        return referenced

    def _keyword_match(self, context_text: str, agent_output: str) -> bool:
        context_tokens = [token for token in _tokens(context_text) if len(token) > 3]
        output_tokens = set(_tokens(agent_output))
        if not context_tokens or not output_tokens:
            return False

        first_terms = set(context_tokens[:12])
        overlap = first_terms & output_tokens
        if overlap:
            return True

        # Also catch longer exact snippets that survive model paraphrasing less
        # often but provide high-confidence cache hits when present.
        snippet = " ".join(context_text.split()[:8]).lower()
        return len(snippet) > 24 and snippet in agent_output.lower()

    def _semantic_match(self, element: ContextElement, output: str) -> bool:
        """Use cosine similarity for embedding-backed matching when available."""

        output_embedding = element.metadata.get("output_embedding")
        if not output_embedding or not element.embedding:
            return False

        return cosine_similarity(element.embedding, output_embedding) >= 0.72

    def get_access_count(self, element_id: str) -> int:
        """Get total cache hits for a context element."""

        return sum(self.access_matrix[element_id].values())

    def get_hot_elements(self, threshold: int = 2) -> Set[str]:
        """Get frequently accessed elements."""

        return {
            element_id
            for element_id in self.access_matrix
            if self.get_access_count(element_id) >= threshold
        }


class AdaptiveContextPruner:
    """Adaptive Context Pruning with LFU/LRU hybrid eviction.

    Retention score:

        score = alpha*frequency + beta*importance + gamma*recency + delta*dependency
    """

    def __init__(
        self,
        alpha: float = 1.5,
        beta: float = 1.0,
        gamma: float = 0.5,
        delta: float = 10.0,
        prune_ratio: float = 0.45,
        cache_threshold: int = 2,
        priority_boost: float = 1.5,
    ) -> None:
        if not 0 <= prune_ratio < 1:
            raise ValueError("prune_ratio must be in [0, 1)")
        if cache_threshold < 1:
            raise ValueError("cache_threshold must be >= 1")

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.prune_ratio = prune_ratio
        self.cache_threshold = cache_threshold
        self.priority_boost = priority_boost
        self.access_tracker = AccessPatternTracker()
        self.eviction_history: List[Dict[str, float]] = []

    def track_context_access(
        self,
        context: Iterable[ContextElement],
        agent_id: str,
        agent_output: str,
        use_semantic: bool = False,
    ) -> None:
        """Track which context elements this agent accessed."""

        for element in context:
            self.access_tracker.track_access(
                element,
                agent_id=agent_id,
                agent_output=agent_output,
                use_semantic=use_semantic,
            )

    def apply_frequency_based_prioritization(
        self, context: Iterable[ContextElement]
    ) -> None:
        """Boost priority for frequently accessed elements."""

        hot_elements = self.access_tracker.get_hot_elements(self.cache_threshold)
        for element in context:
            if element.id in hot_elements:
                element.cache_priority *= self.priority_boost
                logger.debug(
                    "Priority boost: %s -> %.2f", element.id, element.cache_priority
                )

    def compute_retention_scores(
        self,
        context: Iterable[ContextElement],
        current_timestamp: int,
    ) -> Dict[str, float]:
        """Compute retention score for each context element."""

        scores: Dict[str, float] = {}
        for element in context:
            frequency_score = element.access_count * element.cache_priority
            importance_score = clamp(element.importance_score, 0.0, 1.0)
            age = max(0, current_timestamp - element.timestamp)
            recency_score = 0.9**age
            dependency_score = self.delta if element.in_dependency_graph else 0.0

            scores[element.id] = (
                self.alpha * frequency_score
                + self.beta * importance_score
                + self.gamma * recency_score
                + dependency_score
            )

        return scores

    def identify_dependency_graphs(
        self,
        context: Iterable[ContextElement],
        citations: Set[str],
    ) -> List[ContextElement]:
        """Mark citation nodes as pinned memory."""

        normalized_citations = {citation.lower() for citation in citations if citation}
        marked: List[ContextElement] = []
        for element in context:
            haystack = element.text.lower()
            if any(citation in haystack for citation in normalized_citations):
                element.in_dependency_graph = True
                logger.debug("Dependency graph: %s contains citation", element.id)
            marked.append(element)
        return marked

    def competitive_eviction(
        self,
        context: List[ContextElement],
        current_timestamp: int,
        citations: Optional[Set[str]] = None,
    ) -> Tuple[List[ContextElement], Dict[str, float]]:
        """Remove the coldest non-pinned context elements."""

        if not context:
            return [], {
                "n_original": 0,
                "n_retained": 0,
                "n_evicted": 0,
                "eviction_ratio": 0.0,
                "n_dependencies_preserved": 0,
                "avg_score_retained": 0.0,
                "avg_score_evicted": 0.0,
                "timestamp": float(current_timestamp),
            }

        working_context = list(context)
        if citations:
            working_context = self.identify_dependency_graphs(working_context, citations)

        self.apply_frequency_based_prioritization(working_context)
        scores = self.compute_retention_scores(working_context, current_timestamp)
        sorted_elements = sorted(
            working_context,
            key=lambda element: scores[element.id],
            reverse=True,
        )

        dependency_elements = [
            element for element in sorted_elements if element.in_dependency_graph
        ]
        non_dependency = [
            element for element in sorted_elements if not element.in_dependency_graph
        ]

        n_total = len(sorted_elements)
        n_retain = max(1, math.ceil(n_total * (1 - self.prune_ratio)))
        n_non_dep_retain = max(0, n_retain - len(dependency_elements))

        retained = dependency_elements + non_dependency[:n_non_dep_retain]
        evicted = non_dependency[n_non_dep_retain:]
        stats = {
            "n_original": n_total,
            "n_retained": len(retained),
            "n_evicted": len(evicted),
            "eviction_ratio": 1 - (len(retained) / n_total),
            "n_dependencies_preserved": len(dependency_elements),
            "avg_score_retained": mean([scores[element.id] for element in retained])
            if retained
            else 0.0,
            "avg_score_evicted": mean([scores[element.id] for element in evicted])
            if evicted
            else 0.0,
            "timestamp": float(current_timestamp),
        }

        self.eviction_history.append(stats)
        logger.info(
            "Eviction @ t=%s: %s -> %s elements (%.1f%% removed)",
            current_timestamp,
            n_total,
            len(retained),
            stats["eviction_ratio"] * 100,
        )
        return retained, stats

    def get_eviction_statistics(self) -> Dict[str, float]:
        """Get aggregate statistics across all evictions."""

        if not self.eviction_history:
            return {}

        return {
            "total_evictions": len(self.eviction_history),
            "avg_eviction_ratio": mean(
                [stats["eviction_ratio"] for stats in self.eviction_history]
            ),
            "total_dependencies_preserved": sum(
                stats["n_dependencies_preserved"] for stats in self.eviction_history
            ),
            "final_context_size": self.eviction_history[-1]["n_retained"],
        }


def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    """Return cosine similarity for two vectors."""

    left_values = list(left)
    right_values = list(right)
    if len(left_values) != len(right_values) or not left_values:
        return 0.0

    dot = sum(a * b for a, b in zip(left_values, right_values))
    left_norm = math.sqrt(sum(a * a for a in left_values))
    right_norm = math.sqrt(sum(b * b for b in right_values))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
