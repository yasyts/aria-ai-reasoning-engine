"""
Problem representation for ARIA.

Defines the Problem dataclass and ProblemType enum for classifying
problems into categories that strategies can target.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any


class ProblemType(Enum):
    """Categories of problems the engine can solve."""
    MATHEMATICAL = auto()
    LOGICAL = auto()
    OPTIMIZATION = auto()
    PATTERN = auto()
    SEQUENCE = auto()
    TEXT_ANALYSIS = auto()
    GENERAL = auto()


# Keyword sets for auto-classification (keyword -> weight)
_CLASSIFICATION_KEYWORDS: dict[ProblemType, dict[str, float]] = {
    ProblemType.MATHEMATICAL: {
        "calculate": 2.0, "compute": 2.0, "solve": 1.5, "equation": 2.0,
        "sum": 1.5, "product": 1.5, "integral": 2.0, "derivative": 2.0,
        "factorial": 2.0, "percent": 1.5, "average": 1.5, "mean": 1.5,
        "median": 1.5, "roots": 1.8, "quadratic": 2.0, "algebra": 2.0,
        "gcd": 2.0, "lcm": 2.0, "prime": 1.5, "remainder": 1.5,
        "modulo": 1.5, "divide": 1.2, "multiply": 1.2, "subtract": 1.2,
        "add": 1.0, "arithmetic": 2.0, "standard deviation": 2.0,
        "permutation": 1.8, "combination": 1.8, "logarithm": 2.0,
        "sqrt": 1.5, "square root": 1.5, "exponent": 1.5, "power": 1.0,
    },
    ProblemType.LOGICAL: {
        "if": 0.8, "then": 0.8, "therefore": 2.0, "implies": 2.0,
        "all": 0.8, "some": 0.5, "none": 0.8, "true": 1.2, "false": 1.2,
        "valid": 1.5, "invalid": 1.5, "syllogism": 2.5, "deduce": 2.0,
        "conclude": 1.8, "premise": 2.0, "logic": 2.0, "boolean": 2.0,
        "and": 0.3, "or": 0.3, "not": 0.3, "contradiction": 2.0,
        "inference": 2.0, "proposition": 2.0, "proof": 1.5,
    },
    ProblemType.OPTIMIZATION: {
        "minimize": 2.5, "maximize": 2.5, "optimal": 2.0, "best": 1.0,
        "shortest": 2.0, "cheapest": 2.0, "fastest": 2.0, "efficient": 1.5,
        "knapsack": 3.0, "traveling": 2.0, "salesman": 2.5, "path": 1.5,
        "route": 1.5, "distance": 1.2, "cost": 1.2, "weight": 0.8,
        "capacity": 1.5, "constraint": 1.5, "optimize": 2.5, "budget": 1.5,
    },
    ProblemType.PATTERN: {
        "pattern": 2.5, "recognize": 1.5, "detect": 1.5, "repetition": 2.0,
        "palindrome": 2.5, "regex": 2.0, "match": 1.0, "grid": 1.5,
        "matrix": 1.5, "symmetry": 2.0, "repeating": 2.0,
    },
    ProblemType.SEQUENCE: {
        "sequence": 2.5, "next": 1.5, "series": 2.0, "predict": 1.5,
        "continue": 1.2, "fibonacci": 3.0, "progression": 2.5,
        "nth term": 2.5, "arithmetic progression": 3.0,
        "geometric progression": 3.0, "term": 1.0,
    },
    ProblemType.TEXT_ANALYSIS: {
        "analyze": 1.5, "summarize": 2.0, "meaning": 1.5, "sentiment": 2.5,
        "extract": 1.2, "classify text": 2.5, "word count": 2.0,
        "frequency": 1.5, "paragraph": 1.2, "text": 0.8,
        "language": 1.2, "grammar": 2.0,
    },
}


@dataclass
class Problem:
    """Represents a problem to be solved by the ARIA engine."""

    description: str
    problem_type: ProblemType = ProblemType.GENERAL
    data: dict[str, Any] = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at: datetime = field(default_factory=datetime.now)

    @classmethod
    def from_text(cls, text: str) -> Problem:
        """Create a Problem from natural language text, auto-classifying its type."""
        problem_type, confidence = cls._classify(text)
        data = {
            "raw_text": text,
            "classification_confidence": confidence,
            "numbers": cls._extract_numbers(text),
        }
        return cls(
            description=text,
            problem_type=problem_type,
            data=data,
        )

    @classmethod
    def _classify(cls, text: str) -> tuple[ProblemType, float]:
        """Classify text into a ProblemType using weighted keyword scoring."""
        text_lower = text.lower()
        tokens = set(re.findall(r'\b\w+\b', text_lower))

        # Check for math operators as strong signal
        has_math_ops = bool(re.search(r'[\d]+\s*[+\-*/^]\s*[\d]+', text))
        has_equation = '=' in text and any(c.isalpha() for c in text)

        scores: dict[ProblemType, float] = {}
        for ptype, keywords in _CLASSIFICATION_KEYWORDS.items():
            score = 0.0
            for keyword, weight in keywords.items():
                kw_tokens = set(keyword.lower().split())
                if kw_tokens.issubset(tokens) or keyword.lower() in text_lower:
                    score += weight
            scores[ptype] = score

        # Boost math score if operators or statistical indicators present
        is_stats = any(w in text_lower for w in ["mean", "average", "median", "mode", "standard deviation", "variance", "std dev"])
        if has_math_ops:
            scores[ProblemType.MATHEMATICAL] = scores.get(ProblemType.MATHEMATICAL, 0) + 5.0
        if has_equation:
            scores[ProblemType.MATHEMATICAL] = scores.get(ProblemType.MATHEMATICAL, 0) + 3.0
        if is_stats:
            scores[ProblemType.MATHEMATICAL] = scores.get(ProblemType.MATHEMATICAL, 0) + 6.0

        # Check for sequence indicators (only if not primarily a statistical query)
        if re.search(r'\d+\s*,\s*\d+\s*,\s*\d+', text) and not is_stats:
            scores[ProblemType.SEQUENCE] = scores.get(ProblemType.SEQUENCE, 0) + 4.0
            scores[ProblemType.PATTERN] = scores.get(ProblemType.PATTERN, 0) + 2.0

        # Find best match
        if not scores or max(scores.values()) == 0:
            return ProblemType.GENERAL, 0.0

        best_type = max(scores, key=scores.get)  # type: ignore[arg-type]
        best_score = scores[best_type]
        total = sum(scores.values())
        confidence = best_score / total if total > 0 else 0.0

        return best_type, min(confidence, 1.0)

    @staticmethod
    def _extract_numbers(text: str) -> list[float]:
        """Extract all numbers from text."""
        numbers = re.findall(r'-?\d+\.?\d*', text)
        return [float(n) for n in numbers]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "description": self.description,
            "problem_type": self.problem_type.name,
            "data": self.data,
            "constraints": self.constraints,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Problem:
        """Deserialize from dictionary."""
        return cls(
            description=d["description"],
            problem_type=ProblemType[d["problem_type"]],
            data=d.get("data", {}),
            constraints=d.get("constraints", []),
            id=d.get("id", str(uuid.uuid4())[:8]),
            created_at=datetime.fromisoformat(d["created_at"]) if "created_at" in d else datetime.now(),
        )

    def __str__(self) -> str:
        return f"Problem[{self.id}|{self.problem_type.name}]: {self.description[:80]}"
