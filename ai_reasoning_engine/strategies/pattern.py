"""
Pattern recognition strategy.

Detects numeric sequences (arithmetic, geometric, Fibonacci, polynomial),
string patterns, and predicts next values.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from typing import Any

from ai_reasoning_engine.core.problem import Problem, ProblemType
from ai_reasoning_engine.core.solution import Solution
from ai_reasoning_engine.strategies.base import BaseStrategy


class PatternStrategy(BaseStrategy):
    """Recognises patterns in sequences, strings, and grids."""

    @property
    def name(self) -> str:
        return "Pattern"

    @property
    def description(self) -> str:
        return "Sequence prediction, pattern recognition, string analysis"

    @property
    def supported_types(self) -> list[ProblemType]:
        return [ProblemType.PATTERN, ProblemType.SEQUENCE, ProblemType.GENERAL]

    def solve(self, problem: Problem) -> Solution:
        start = time.perf_counter()
        steps: list[str] = []
        text = problem.description

        try:
            for solver in [
                self._try_numeric_sequence,
                self._try_fibonacci_like,
                self._try_string_pattern,
            ]:
                result = solver(text, steps)
                if result is not None:
                    elapsed = time.perf_counter() - start
                    return Solution(
                        answer=result,
                        strategy_name=self.name,
                        confidence=0.85,
                        reasoning_steps=steps,
                        execution_time=elapsed,
                    )

            steps.append("No recognisable pattern found")
            elapsed = time.perf_counter() - start
            return Solution(
                answer=None, strategy_name=self.name,
                confidence=0.1, reasoning_steps=steps,
                execution_time=elapsed,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - start
            steps.append(f"Error: {exc}")
            return Solution(
                answer=None, strategy_name=self.name,
                confidence=0.05, reasoning_steps=steps,
                execution_time=elapsed,
            )

    # ------------------------------------------------------------------ #
    #                      Numeric Sequences                              #
    # ------------------------------------------------------------------ #

    def _try_numeric_sequence(self, text: str, steps: list[str]) -> Any:
        """Detect arithmetic, geometric, or polynomial sequences."""
        nums = [float(n) for n in re.findall(r'-?\d+\.?\d*', text)]
        if len(nums) < 3:
            return None

        steps.append(f"Sequence: {nums}")

        # 1) Arithmetic: constant first difference
        diffs = [nums[i + 1] - nums[i] for i in range(len(nums) - 1)]
        if len(set(diffs)) == 1:
            d = diffs[0]
            next_val = nums[-1] + d
            steps.append(f"Arithmetic sequence with common difference d = {d}")
            steps.append(f"Next value: {nums[-1]} + {d} = {next_val}")
            return {"pattern": "arithmetic", "common_difference": d, "next_value": next_val}

        # 2) Geometric: constant ratio
        if all(nums[i] != 0 for i in range(len(nums) - 1)):
            ratios = [nums[i + 1] / nums[i] for i in range(len(nums) - 1)]
            # Allow small floating-point tolerance
            if all(abs(r - ratios[0]) < 1e-9 for r in ratios):
                r = ratios[0]
                next_val = nums[-1] * r
                steps.append(f"Geometric sequence with common ratio r = {r}")
                steps.append(f"Next value: {nums[-1]} × {r} = {next_val}")
                return {"pattern": "geometric", "common_ratio": r, "next_value": next_val}

        # 3) Quadratic (second differences constant)
        if len(nums) >= 4:
            second_diffs = [diffs[i + 1] - diffs[i] for i in range(len(diffs) - 1)]
            if len(set(round(d, 8) for d in second_diffs)) == 1:
                sd = second_diffs[0]
                next_diff = diffs[-1] + sd
                next_val = nums[-1] + next_diff
                steps.append(f"Quadratic sequence (2nd diff = {sd})")
                steps.append(f"Next difference: {next_diff}, next value: {next_val}")
                return {"pattern": "quadratic", "second_difference": sd, "next_value": next_val}

        # 4) Higher-order polynomial via finite differences
        result = self._finite_differences(nums, steps)
        if result is not None:
            return result

        return None

    def _finite_differences(self, nums: list[float], steps: list[str]) -> Any:
        """Use method of finite differences for polynomial sequences."""
        table: list[list[float]] = [nums[:]]
        while len(table[-1]) > 1:
            row = table[-1]
            new_row = [row[i + 1] - row[i] for i in range(len(row) - 1)]
            table.append(new_row)
            if all(abs(v) < 1e-9 for v in new_row):
                break

        # If we reached a constant row, extrapolate
        if len(table[-1]) >= 1 and len(set(round(v, 8) for v in table[-1])) == 1:
            order = len(table) - 1
            steps.append(f"Polynomial of degree {order - 1} (constant at level {order})")

            # Extrapolate: add one element to each row bottom-up
            for i in range(len(table) - 1, 0, -1):
                table[i].append(table[i][-1])
                table[i - 1].append(table[i - 1][-1] + table[i][-1])

            next_val = table[0][-1]
            steps.append(f"Next value: {next_val}")
            return {"pattern": f"polynomial_degree_{order - 1}", "next_value": next_val}

        return None

    # ------------------------------------------------------------------ #
    #                      Fibonacci-like                                 #
    # ------------------------------------------------------------------ #

    def _try_fibonacci_like(self, text: str, steps: list[str]) -> Any:
        """Detect sequences where each term = sum of previous two."""
        nums = [float(n) for n in re.findall(r'-?\d+\.?\d*', text)]
        if len(nums) < 4:
            return None

        # Check if a[n] = a[n-1] + a[n-2]
        is_fib = all(
            abs(nums[i] - (nums[i - 1] + nums[i - 2])) < 1e-9
            for i in range(2, len(nums))
        )
        if is_fib:
            next_val = nums[-1] + nums[-2]
            steps.append("Fibonacci-like sequence: each term = sum of two preceding terms")
            steps.append(f"Next value: {nums[-1]} + {nums[-2]} = {next_val}")
            return {"pattern": "fibonacci", "next_value": next_val}

        # Check tribonacci: a[n] = a[n-1] + a[n-2] + a[n-3]
        if len(nums) >= 5:
            is_trib = all(
                abs(nums[i] - (nums[i - 1] + nums[i - 2] + nums[i - 3])) < 1e-9
                for i in range(3, len(nums))
            )
            if is_trib:
                next_val = nums[-1] + nums[-2] + nums[-3]
                steps.append("Tribonacci sequence: each term = sum of three preceding terms")
                steps.append(f"Next value: {next_val}")
                return {"pattern": "tribonacci", "next_value": next_val}

        return None

    # ------------------------------------------------------------------ #
    #                       String Patterns                               #
    # ------------------------------------------------------------------ #

    def _try_string_pattern(self, text: str, steps: list[str]) -> Any:
        """Palindrome detection, character frequency, repetition patterns."""
        text_lower = text.lower()

        # Palindrome check
        if 'palindrome' in text_lower:
            # Extract quoted string or last word
            quoted = re.findall(r'["\'](.+?)["\']', text)
            subject = quoted[0] if quoted else text.split()[-1]
            clean = re.sub(r'[^a-z0-9]', '', subject.lower())
            is_pal = clean == clean[::-1]
            steps.append(f"Checking if '{subject}' is a palindrome")
            steps.append(f"Cleaned: '{clean}' → {'Yes' if is_pal else 'No'}")
            return {"is_palindrome": is_pal, "text": subject}

        # Character frequency
        if 'frequency' in text_lower or 'count' in text_lower:
            quoted = re.findall(r'["\'](.+?)["\']', text)
            subject = quoted[0] if quoted else text
            freq = dict(Counter(subject.lower()))
            freq = dict(sorted(freq.items(), key=lambda x: -x[1]))
            steps.append(f"Character frequency of '{subject[:50]}'")
            return {"frequency": freq}

        return None
