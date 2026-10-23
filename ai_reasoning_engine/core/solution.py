"""
Solution representation for ARIA.

Defines the Solution dataclass that holds a strategy's answer,
reasoning chain, confidence, and evaluation score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Solution:
    """Represents a candidate solution produced by a reasoning strategy."""

    answer: Any
    strategy_name: str
    confidence: float = 0.0  # 0.0 – 1.0
    reasoning_steps: list[str] = field(default_factory=list)
    execution_time: float = 0.0  # seconds
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0  # set by evaluator

    def __post_init__(self) -> None:
        """Clamp confidence to [0, 1]."""
        self.confidence = max(0.0, min(1.0, self.confidence))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "answer": self._serialize_answer(self.answer),
            "strategy_name": self.strategy_name,
            "confidence": self.confidence,
            "reasoning_steps": self.reasoning_steps,
            "execution_time": self.execution_time,
            "metadata": self.metadata,
            "score": self.score,
        }

    @staticmethod
    def _serialize_answer(answer: Any) -> Any:
        """Safely serialize answer to JSON-compatible type."""
        if isinstance(answer, (int, float, str, bool, type(None))):
            return answer
        if isinstance(answer, (list, tuple)):
            return [Solution._serialize_answer(a) for a in answer]
        if isinstance(answer, dict):
            return {str(k): Solution._serialize_answer(v) for k, v in answer.items()}
        return str(answer)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Solution:
        """Deserialize from dictionary."""
        return cls(
            answer=d["answer"],
            strategy_name=d["strategy_name"],
            confidence=d.get("confidence", 0.0),
            reasoning_steps=d.get("reasoning_steps", []),
            execution_time=d.get("execution_time", 0.0),
            metadata=d.get("metadata", {}),
            score=d.get("score", 0.0),
        )

    def __str__(self) -> str:
        conf_pct = f"{self.confidence * 100:.0f}%"
        return (
            f"Solution[{self.strategy_name}] "
            f"answer={self.answer} "
            f"confidence={conf_pct} "
            f"score={self.score:.3f}"
        )

    def summary(self) -> str:
        """Return a human-readable multi-line summary."""
        lines = [
            f"Strategy : {self.strategy_name}",
            f"Answer   : {self.answer}",
            f"Confidence: {self.confidence * 100:.1f}%",
            f"Score    : {self.score:.3f}",
            f"Time     : {self.execution_time:.4f}s",
            f"Steps    : {len(self.reasoning_steps)}",
        ]
        if self.reasoning_steps:
            lines.append("Reasoning:")
            for i, step in enumerate(self.reasoning_steps, 1):
                lines.append(f"  {i}. {step}")
        return "\n".join(lines)
