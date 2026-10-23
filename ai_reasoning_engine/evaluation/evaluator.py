"""
Multi-criteria solution evaluator and ranker for ARIA.

Scores solutions on confidence, consistency (consensus),
efficiency, and completeness, then picks the best one.
"""

from __future__ import annotations

from typing import Any

from ai_reasoning_engine.core.problem import Problem
from ai_reasoning_engine.core.solution import Solution


class SolutionEvaluator:
    """Evaluates and ranks candidate solutions using weighted multi-criteria scoring."""

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self._weights = weights or {
            "confidence": 0.35,
            "consistency": 0.25,
            "efficiency": 0.20,
            "completeness": 0.20,
        }

    # ------------------------------------------------------------------ #
    #                        public interface                             #
    # ------------------------------------------------------------------ #

    def evaluate(self, solutions: list[Solution], problem: Problem) -> list[Solution]:
        """Score all solutions and return them sorted best-first."""
        if not solutions:
            return []

        # Filter out None-answer solutions with very low confidence
        valid = [s for s in solutions if s.answer is not None or s.confidence > 0.3]
        if not valid:
            valid = solutions  # keep all if nothing is good

        # Compute raw scores
        times = [s.execution_time for s in valid]
        max_time = max(times) if times else 1.0

        for sol in valid:
            conf_score = sol.confidence
            cons_score = self._compute_consistency(sol, valid)
            eff_score = 1.0 - (sol.execution_time / (max_time + 1e-9)) if max_time > 0 else 1.0
            comp_score = min(len(sol.reasoning_steps) / 5.0, 1.0)

            sol.score = (
                self._weights["confidence"] * conf_score
                + self._weights["consistency"] * cons_score
                + self._weights["efficiency"] * eff_score
                + self._weights["completeness"] * comp_score
            )

        valid.sort(key=lambda s: -s.score)
        return valid

    def get_best(self, solutions: list[Solution], problem: Problem) -> Solution:
        """Return the single best solution."""
        ranked = self.evaluate(solutions, problem)
        if not ranked:
            return Solution(
                answer=None, strategy_name="none",
                confidence=0.0, reasoning_steps=["No solutions available"],
            )
        return ranked[0]

    # ------------------------------------------------------------------ #
    #                          internals                                  #
    # ------------------------------------------------------------------ #

    def _compute_consistency(self, solution: Solution, all_solutions: list[Solution]) -> float:
        """How many other solutions agree with this one (consensus score)."""
        if len(all_solutions) <= 1:
            return 0.5  # neutral when only one solution

        matches = sum(
            1 for other in all_solutions
            if other is not solution and self._answers_match(solution.answer, other.answer)
        )
        return matches / (len(all_solutions) - 1)

    def _answers_match(self, a: Any, b: Any) -> bool:
        """Fuzzy comparison of two answers."""
        if a is None or b is None:
            return a is b

        # Exact match
        if a == b:
            return True

        # Numeric tolerance
        try:
            fa, fb = float(a), float(b)
            return abs(fa - fb) < max(abs(fa), abs(fb), 1.0) * 0.01  # 1% tolerance
        except (TypeError, ValueError):
            pass

        # String similarity
        sa, sb = str(a).lower().strip(), str(b).lower().strip()
        if sa == sb:
            return True

        # Check if one contains the other
        if sa in sb or sb in sa:
            return True

        # Token overlap
        ta = set(sa.split())
        tb = set(sb.split())
        if ta and tb:
            overlap = len(ta & tb) / max(len(ta), len(tb))
            return overlap > 0.7

        return False

    @staticmethod
    def _normalize(values: list[float]) -> list[float]:
        """Min-max normalisation to [0, 1]."""
        if not values:
            return []
        lo, hi = min(values), max(values)
        if hi - lo < 1e-12:
            return [0.5] * len(values)
        return [(v - lo) / (hi - lo) for v in values]
