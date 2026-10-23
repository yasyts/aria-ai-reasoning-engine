"""
Analogical reasoning strategy.

Solves new problems by finding similar past problems in its
knowledge bank and adapting those solutions.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from typing import Any

from ai_reasoning_engine.core.problem import Problem, ProblemType
from ai_reasoning_engine.core.solution import Solution
from ai_reasoning_engine.strategies.base import BaseStrategy


class AnalogicalStrategy(BaseStrategy):
    """Reasoning by analogy — find similar past problems and adapt."""

    def __init__(self) -> None:
        self._knowledge: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "Analogical"

    @property
    def description(self) -> str:
        return "Finds similar past problems and adapts their solutions"

    @property
    def supported_types(self) -> list[ProblemType]:
        return list(ProblemType)  # Can attempt any type

    def add_knowledge(self, problem_text: str, solution_answer: Any) -> None:
        """Store a solved problem–solution pair for future analogies."""
        self._knowledge.append({
            "text": problem_text,
            "tokens": self._tokenize(problem_text),
            "answer": solution_answer,
        })

    def load_knowledge(self, entries: list[dict[str, Any]]) -> None:
        """Bulk-load knowledge entries."""
        for entry in entries:
            self._knowledge.append({
                "text": entry.get("text", ""),
                "tokens": self._tokenize(entry.get("text", "")),
                "answer": entry.get("answer"),
            })

    # ------------------------------------------------------------------ #

    def solve(self, problem: Problem) -> Solution:
        start = time.perf_counter()
        steps: list[str] = []
        text = problem.description

        if not self._knowledge:
            steps.append("Knowledge bank is empty — no analogies available")
            elapsed = time.perf_counter() - start
            return Solution(
                answer=None,
                strategy_name=self.name,
                confidence=0.05,
                reasoning_steps=steps,
                execution_time=elapsed,
            )

        steps.append(f"Searching {len(self._knowledge)} known problems for best analogy")

        best_sim = 0.0
        best_entry: dict[str, Any] | None = None
        query_tokens = self._tokenize(text)

        for entry in self._knowledge:
            sim = self._similarity(query_tokens, entry["tokens"])
            if sim > best_sim:
                best_sim = sim
                best_entry = entry

        if best_entry is None or best_sim < 0.1:
            steps.append("No sufficiently similar problem found")
            elapsed = time.perf_counter() - start
            return Solution(
                answer=None, strategy_name=self.name,
                confidence=0.05, reasoning_steps=steps,
                execution_time=elapsed,
            )

        steps.append(f"Best match (similarity {best_sim:.2f}): '{best_entry['text'][:80]}'")
        steps.append(f"Adapting solution: {best_entry['answer']}")

        confidence = min(best_sim, 0.95)
        elapsed = time.perf_counter() - start

        return Solution(
            answer=best_entry["answer"],
            strategy_name=self.name,
            confidence=confidence,
            reasoning_steps=steps,
            execution_time=elapsed,
            metadata={"analogy_similarity": best_sim, "source_problem": best_entry["text"][:120]},
        )

    # ------------------------------------------------------------------ #
    #                       text similarity helpers                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Lowercase, remove punctuation, split into word set."""
        words = re.findall(r'\b\w+\b', text.lower())
        stop = {'the', 'a', 'an', 'is', 'are', 'of', 'in', 'to', 'and', 'or', 'for', 'it', 'on'}
        return {w for w in words if w not in stop and len(w) > 1}

    @staticmethod
    def _similarity(tokens_a: set[str], tokens_b: set[str]) -> float:
        """Jaccard similarity between two token sets."""
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return len(intersection) / len(union)

    @staticmethod
    def _cosine_similarity(text_a: str, text_b: str) -> float:
        """Cosine similarity using word frequency vectors."""
        words_a = re.findall(r'\b\w+\b', text_a.lower())
        words_b = re.findall(r'\b\w+\b', text_b.lower())
        counter_a = Counter(words_a)
        counter_b = Counter(words_b)
        all_words = set(counter_a) | set(counter_b)
        dot = sum(counter_a.get(w, 0) * counter_b.get(w, 0) for w in all_words)
        mag_a = sum(v ** 2 for v in counter_a.values()) ** 0.5
        mag_b = sum(v ** 2 for v in counter_b.values()) ** 0.5
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)
