"""
Mathematical reasoning strategy.

Solves arithmetic, algebra, quadratic equations, statistics,
factorials, permutations, combinations, GCD/LCM, and expression evaluation.
"""

from __future__ import annotations

import math
import re
import time
from typing import Any

from ai_reasoning_engine.core.problem import Problem, ProblemType
from ai_reasoning_engine.core.solution import Solution
from ai_reasoning_engine.strategies.base import BaseStrategy


class MathematicalStrategy(BaseStrategy):
    """Solves mathematical problems using symbolic + numeric methods."""

    @property
    def name(self) -> str:
        return "Mathematical"

    @property
    def description(self) -> str:
        return "Arithmetic, algebra, statistics, and combinatorics"

    @property
    def supported_types(self) -> list[ProblemType]:
        return [ProblemType.MATHEMATICAL, ProblemType.SEQUENCE, ProblemType.GENERAL]

    # ------------------------------------------------------------------ #
    #                         public entry point                          #
    # ------------------------------------------------------------------ #

    def solve(self, problem: Problem) -> Solution:
        start = time.perf_counter()
        steps: list[str] = []
        text = problem.description

        try:
            # Try specialized solvers in order (quadratic before linear)
            for solver in [
                self._try_quadratic,
                self._try_linear_equation,
                self._try_factorial,
                self._try_gcd_lcm,
                self._try_permutation_combination,
                self._try_statistics,
                self._try_expression,
            ]:
                result = solver(text, steps)
                if result is not None:
                    elapsed = time.perf_counter() - start
                    return Solution(
                        answer=result,
                        strategy_name=self.name,
                        confidence=0.9,
                        reasoning_steps=steps,
                        execution_time=elapsed,
                    )

            # Fallback: extract numbers and try basic ops
            answer = self._fallback(text, steps)
            elapsed = time.perf_counter() - start
            return Solution(
                answer=answer,
                strategy_name=self.name,
                confidence=0.4 if answer is not None else 0.1,
                reasoning_steps=steps,
                execution_time=elapsed,
            )

        except Exception as exc:
            elapsed = time.perf_counter() - start
            steps.append(f"Error: {exc}")
            return Solution(
                answer=None,
                strategy_name=self.name,
                confidence=0.05,
                reasoning_steps=steps,
                execution_time=elapsed,
            )

    # ------------------------------------------------------------------ #
    #                        specialised solvers                          #
    # ------------------------------------------------------------------ #

    def _try_linear_equation(self, text: str, steps: list[str]) -> Any:
        """Solve  ax + b = c  style equations."""
        if '^' in text or '²' in text:
            return None

        # Patterns: "solve 2x + 5 = 15",  "2*x + 5 = 15"
        m = re.search(
            r'(-?\d*\.?\d*)\s*\*?\s*[xX]\s*([+\-])\s*(\d+\.?\d*)\s*=\s*(-?\d+\.?\d*)',
            text,
        )
        if not m:
            return None

        a_str, op, b_str, c_str = m.groups()
        a = float(a_str) if a_str and a_str != '-' else (-1.0 if a_str == '-' else 1.0)
        b = float(b_str) if op == '+' else -float(b_str)
        c = float(c_str)

        steps.append(f"Detected linear equation: {a}x + {b} = {c}")
        if a == 0:
            steps.append("Coefficient of x is 0 — no unique solution")
            return None
        x = (c - b) / a
        steps.append(f"x = ({c} - {b}) / {a} = {x}")
        return x

    def _try_quadratic(self, text: str, steps: list[str]) -> Any:
        """Solve  ax² + bx + c = 0  using the quadratic formula."""
        m = re.search(
            r'(-?\d*\.?\d*)\s*[xX]\s*[\^²]\s*2?\s*'
            r'([+\-])\s*(\d*\.?\d*)\s*[xX]\s*'
            r'([+\-])\s*(\d+\.?\d*)\s*=\s*0',
            text,
        )
        if not m:
            # Simpler pattern: x^2 - 5x + 6 = 0
            m = re.search(
                r'[xX]\s*\^\s*2\s*([+\-])\s*(\d+\.?\d*)\s*[xX]\s*([+\-])\s*(\d+\.?\d*)\s*=\s*0',
                text,
            )
            if m:
                a = 1.0
                op_b, b_val, op_c, c_val = m.groups()
                b = float(b_val) if op_b == '+' else -float(b_val)
                c = float(c_val) if op_c == '+' else -float(c_val)
            else:
                return None
        else:
            a_str, op_b, b_str, op_c, c_str = m.groups()
            a = float(a_str) if a_str and a_str not in ('', '-') else (-1.0 if a_str == '-' else 1.0)
            b = float(b_str) if op_b == '+' else -float(b_str if b_str else '1')
            c = float(c_str) if op_c == '+' else -float(c_str)

        steps.append(f"Detected quadratic: {a}x² + {b}x + {c} = 0")
        discriminant = b * b - 4 * a * c
        steps.append(f"Discriminant = {b}² - 4·{a}·{c} = {discriminant}")

        if discriminant < 0:
            steps.append("Discriminant < 0 → complex roots")
            real_part = -b / (2 * a)
            imag_part = math.sqrt(-discriminant) / (2 * a)
            return f"{real_part} ± {imag_part}i"
        elif discriminant == 0:
            x = -b / (2 * a)
            steps.append(f"Double root: x = {x}")
            return x
        else:
            sqrt_d = math.sqrt(discriminant)
            x1 = (-b + sqrt_d) / (2 * a)
            x2 = (-b - sqrt_d) / (2 * a)
            steps.append(f"x₁ = {x1}, x₂ = {x2}")
            return (x1, x2)

    def _try_factorial(self, text: str, steps: list[str]) -> Any:
        """Compute factorial."""
        m = re.search(r'factorial\s*(?:of)?\s*(\d+)', text, re.IGNORECASE)
        if not m:
            m = re.search(r'(\d+)\s*!', text)
        if not m:
            return None

        n = int(m.group(1))
        steps.append(f"Computing {n}!")
        if n > 170:
            steps.append("n too large for float; using exact integer")
        result = math.factorial(n)
        steps.append(f"{n}! = {result}")
        return result

    def _try_gcd_lcm(self, text: str, steps: list[str]) -> Any:
        """Compute GCD or LCM."""
        text_lower = text.lower()
        nums = [int(x) for x in re.findall(r'\d+', text)]
        if len(nums) < 2:
            return None

        if 'gcd' in text_lower or 'greatest common' in text_lower:
            result = nums[0]
            for n in nums[1:]:
                result = math.gcd(result, n)
            steps.append(f"GCD of {nums} = {result}")
            return result

        if 'lcm' in text_lower or 'least common' in text_lower:
            result = nums[0]
            for n in nums[1:]:
                result = result * n // math.gcd(result, n)
            steps.append(f"LCM of {nums} = {result}")
            return result

        return None

    def _try_permutation_combination(self, text: str, steps: list[str]) -> Any:
        """P(n,r) or C(n,r)."""
        text_lower = text.lower()
        nums = [int(x) for x in re.findall(r'\d+', text)]
        if len(nums) < 2:
            return None

        n, r = nums[0], nums[1]
        if r > n:
            return None

        if 'permutation' in text_lower or re.search(r'[pP]\s*\(\s*\d', text):
            result = math.perm(n, r)
            steps.append(f"P({n},{r}) = {result}")
            return result

        if 'combination' in text_lower or re.search(r'[cC]\s*\(\s*\d', text):
            result = math.comb(n, r)
            steps.append(f"C({n},{r}) = {result}")
            return result

        return None

    def _try_statistics(self, text: str, steps: list[str]) -> Any:
        """Mean, median, mode, standard deviation."""
        text_lower = text.lower()
        stat_keywords = ['mean', 'average', 'median', 'mode', 'standard deviation', 'std dev', 'variance']
        if not any(kw in text_lower for kw in stat_keywords):
            return None

        nums = re.findall(r'-?\d+\.?\d*', text)
        data = [float(n) for n in nums]
        if len(data) < 2:
            return None

        steps.append(f"Data set: {data}")
        results: dict[str, float] = {}

        if 'mean' in text_lower or 'average' in text_lower:
            mean = sum(data) / len(data)
            results['mean'] = round(mean, 4)
            steps.append(f"Mean = {results['mean']}")

        if 'median' in text_lower:
            sorted_d = sorted(data)
            mid = len(sorted_d) // 2
            if len(sorted_d) % 2 == 0:
                median = (sorted_d[mid - 1] + sorted_d[mid]) / 2
            else:
                median = sorted_d[mid]
            results['median'] = round(median, 4)
            steps.append(f"Median = {results['median']}")

        if 'mode' in text_lower:
            from collections import Counter
            counts = Counter(data)
            max_count = max(counts.values())
            modes = [v for v, c in counts.items() if c == max_count]
            results['mode'] = modes[0] if len(modes) == 1 else modes  # type: ignore[assignment]
            steps.append(f"Mode = {results['mode']}")

        if 'standard deviation' in text_lower or 'std' in text_lower:
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)
            std = math.sqrt(variance)
            results['standard_deviation'] = round(std, 4)
            results['mean'] = round(mean, 4)
            steps.append(f"Standard deviation = {results['standard_deviation']}")

        if results:
            return results if len(results) > 1 else list(results.values())[0]
        return None

    def _try_expression(self, text: str, steps: list[str]) -> Any:
        """Evaluate a mathematical expression safely."""
        # Extract expression: look for math-like substring
        expr = re.sub(r'[Cc]alculate|[Cc]ompute|[Ww]hat\s+is|[Ff]ind|[Ss]olve|the\s+value\s+of', '', text)
        expr = expr.strip().rstrip('?.')

        # Clean for eval
        expr_clean = re.sub(r'[^0-9+\-*/().%^ ]', '', expr)
        expr_clean = expr_clean.replace('^', '**')

        if not expr_clean or not re.search(r'\d', expr_clean):
            return None

        steps.append(f"Evaluating expression: {expr_clean}")
        try:
            # Safe eval with only math builtins
            allowed = {
                "__builtins__": {},
                "abs": abs, "round": round, "min": min, "max": max,
                "sqrt": math.sqrt, "pow": pow, "log": math.log,
                "sin": math.sin, "cos": math.cos, "pi": math.pi, "e": math.e,
            }
            result = eval(expr_clean, allowed)  # noqa: S307
            steps.append(f"Result = {result}")
            return result
        except Exception:
            return None

    def _fallback(self, text: str, steps: list[str]) -> Any:
        """Last resort: extract all numbers and try sum/product."""
        nums = [float(n) for n in re.findall(r'-?\d+\.?\d*', text)]
        if not nums:
            steps.append("No numbers found in problem text")
            return None
        steps.append(f"Extracted numbers: {nums}")
        if len(nums) == 1:
            return nums[0]
        # Return sum as default operation
        total = sum(nums)
        steps.append(f"Default operation (sum): {total}")
        return total
