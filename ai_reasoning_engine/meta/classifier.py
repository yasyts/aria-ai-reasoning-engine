"""
Problem classifier for ARIA.

Classifies natural-language problem text into ProblemType
using weighted keyword scoring.
"""

from __future__ import annotations

import re

from ai_reasoning_engine.core.problem import ProblemType


_KEYWORD_WEIGHTS: dict[ProblemType, dict[str, float]] = {
    ProblemType.MATHEMATICAL: {
        "calculate": 2.0, "compute": 2.0, "solve": 1.5, "equation": 2.0,
        "sum": 1.5, "product": 1.5, "factorial": 2.0, "percent": 1.5,
        "average": 1.5, "mean": 1.5, "median": 1.5, "roots": 1.8,
        "quadratic": 2.0, "gcd": 2.0, "lcm": 2.0, "prime": 1.5,
        "arithmetic": 2.0, "derivative": 2.0, "integral": 2.0,
        "standard deviation": 2.0, "permutation": 1.8, "combination": 1.8,
    },
    ProblemType.LOGICAL: {
        "if": 0.8, "then": 0.8, "therefore": 2.0, "implies": 2.0,
        "all": 0.8, "none": 0.8, "true": 1.2, "false": 1.2,
        "syllogism": 2.5, "deduce": 2.0, "logic": 2.0, "boolean": 2.0,
        "inference": 2.0, "proposition": 2.0, "premise": 2.0,
    },
    ProblemType.OPTIMIZATION: {
        "minimize": 2.5, "maximize": 2.5, "optimal": 2.0, "shortest": 2.0,
        "cheapest": 2.0, "fastest": 2.0, "efficient": 1.5,
        "knapsack": 3.0, "traveling": 2.0, "salesman": 2.5,
        "path": 1.5, "route": 1.5, "optimize": 2.5,
    },
    ProblemType.PATTERN: {
        "pattern": 2.5, "recognize": 1.5, "detect": 1.5,
        "palindrome": 2.5, "regex": 2.0, "symmetry": 2.0,
        "repetition": 2.0, "grid": 1.5, "matrix": 1.5,
    },
    ProblemType.SEQUENCE: {
        "sequence": 2.5, "next": 1.5, "series": 2.0, "predict": 1.5,
        "continue": 1.2, "fibonacci": 3.0, "progression": 2.5,
        "nth term": 2.5,
    },
    ProblemType.TEXT_ANALYSIS: {
        "analyze": 1.5, "summarize": 2.0, "meaning": 1.5,
        "sentiment": 2.5, "extract": 1.2, "classify text": 2.5,
        "word count": 2.0, "frequency": 1.5,
    },
}


class ProblemClassifier:
    """Classifies problem text into a ProblemType using keyword scores."""

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r'\b\w+\b', text.lower())

    @classmethod
    def classify(cls, text: str) -> ProblemType:
        ptype, _ = cls.get_confidence(text)
        return ptype

    @classmethod
    def get_confidence(cls, text: str) -> tuple[ProblemType, float]:
        text_lower = text.lower()
        tokens = set(cls._tokenize(text))

        scores: dict[ProblemType, float] = {}
        for ptype, keywords in _KEYWORD_WEIGHTS.items():
            score = 0.0
            for kw, weight in keywords.items():
                kw_tokens = set(kw.lower().split())
                if kw_tokens.issubset(tokens) or kw.lower() in text_lower:
                    score += weight
            scores[ptype] = score

        # Boost for numeric patterns
        is_stats = any(w in text_lower for w in ["mean", "average", "median", "mode", "standard deviation", "variance", "std dev"])
        if re.search(r'\d+\s*[+\-*/^]\s*\d+', text):
            scores[ProblemType.MATHEMATICAL] = scores.get(ProblemType.MATHEMATICAL, 0) + 5.0
        if is_stats:
            scores[ProblemType.MATHEMATICAL] = scores.get(ProblemType.MATHEMATICAL, 0) + 6.0
        if re.search(r'\d+\s*,\s*\d+\s*,\s*\d+', text) and not is_stats:
            scores[ProblemType.SEQUENCE] = scores.get(ProblemType.SEQUENCE, 0) + 4.0

        if not scores or max(scores.values()) == 0:
            return ProblemType.GENERAL, 0.0

        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        total = sum(scores.values())
        conf = scores[best] / total if total > 0 else 0.0
        return best, min(conf, 1.0)
