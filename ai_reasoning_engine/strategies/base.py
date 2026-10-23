"""
Abstract base class for all reasoning strategies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ai_reasoning_engine.core.problem import Problem, ProblemType
from ai_reasoning_engine.core.solution import Solution


class BaseStrategy(ABC):
    """Abstract base for reasoning strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """What this strategy does."""
        ...

    @property
    @abstractmethod
    def supported_types(self) -> list[ProblemType]:
        """Problem types this strategy can handle."""
        ...

    @abstractmethod
    def solve(self, problem: Problem) -> Solution:
        """Produce a Solution for the given Problem."""
        ...

    def can_handle(self, problem: Problem) -> bool:
        """Return True if this strategy supports the problem type."""
        return problem.problem_type in self.supported_types

    def __repr__(self) -> str:
        return f"<{self.name}: {self.description}>"
