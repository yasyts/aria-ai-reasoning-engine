"""
Persistent knowledge base for ARIA.

Stores solved problem–solution pairs and provides similarity search
to support analogical reasoning and self-improvement.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from ai_reasoning_engine.core.problem import Problem
from ai_reasoning_engine.core.solution import Solution


class KnowledgeBase:
    """JSON-backed store of solved problems for analogy and learning."""

    def __init__(self, data_dir: str = "aria_data") -> None:
        self._data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self._file = os.path.join(data_dir, "knowledge_base.json")
        self._entries: list[dict[str, Any]] = []
        self.load()

    # ------------------------------------------------------------------ #
    #                        public interface                             #
    # ------------------------------------------------------------------ #

    def store(self, problem: Problem, solution: Solution) -> None:
        """Save a problem–solution pair."""
        entry = {
            "problem_text": problem.description,
            "problem_type": problem.problem_type.name,
            "answer": solution._serialize_answer(solution.answer),
            "strategy": solution.strategy_name,
            "score": solution.score,
            "confidence": solution.confidence,
        }
        self._entries.append(entry)
        self.save()

    def find_similar(self, problem_text: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Find most similar past problems by word overlap."""
        if not self._entries:
            return []

        query_tokens = self._tokenize(problem_text)
        scored: list[tuple[float, dict[str, Any]]] = []

        for entry in self._entries:
            sim = self._compute_similarity(query_tokens, self._tokenize(entry["problem_text"]))
            scored.append((sim, entry))

        scored.sort(key=lambda x: -x[0])
        return [entry for _, entry in scored[:top_k] if _ > 0.05]

    def get_all(self) -> list[dict[str, Any]]:
        return list(self._entries)

    def get_count(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries = []
        self.save()

    # ------------------------------------------------------------------ #
    #                        similarity                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        stop = {'the', 'a', 'an', 'is', 'are', 'of', 'in', 'to', 'and', 'or', 'for', 'it', 'on', 'what'}
        words = re.findall(r'\b\w+\b', text.lower())
        return {w for w in words if w not in stop and len(w) > 1}

    @staticmethod
    def _compute_similarity(tokens_a: set[str], tokens_b: set[str]) -> float:
        """Jaccard similarity."""
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return len(intersection) / len(union)

    # ------------------------------------------------------------------ #
    #                        persistence                                  #
    # ------------------------------------------------------------------ #

    def save(self) -> None:
        try:
            with open(self._file, "w", encoding="utf-8") as f:
                json.dump(self._entries, f, indent=2)
        except OSError:
            pass

    def load(self) -> None:
        if not os.path.exists(self._file):
            return
        try:
            with open(self._file, "r", encoding="utf-8") as f:
                self._entries = json.load(f)
        except (OSError, json.JSONDecodeError):
            self._entries = []
