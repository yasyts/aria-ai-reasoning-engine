"""
Heuristic search strategy.

A*, greedy best-first, hill climbing, simulated annealing,
knapsack solver, and shortest-path solver.
"""

from __future__ import annotations

import heapq
import math
import random
import re
import time
from typing import Any

from ai_reasoning_engine.core.problem import Problem, ProblemType
from ai_reasoning_engine.core.solution import Solution
from ai_reasoning_engine.strategies.base import BaseStrategy


class HeuristicStrategy(BaseStrategy):
    """Search-based and optimization problem solving."""

    @property
    def name(self) -> str:
        return "Heuristic"

    @property
    def description(self) -> str:
        return "A*, hill climbing, simulated annealing, knapsack, shortest path"

    @property
    def supported_types(self) -> list[ProblemType]:
        return [ProblemType.OPTIMIZATION, ProblemType.GENERAL]

    def solve(self, problem: Problem) -> Solution:
        start = time.perf_counter()
        steps: list[str] = []
        text = problem.description

        try:
            for solver in [
                self._try_shortest_path,
                self._try_knapsack,
                self._try_numeric_optimization,
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

            steps.append("No optimisation problem pattern detected")
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
    #                      Shortest Path (Dijkstra)                       #
    # ------------------------------------------------------------------ #

    def _try_shortest_path(self, text: str, steps: list[str]) -> Any:
        """Parse edges and run Dijkstra."""
        text_lower = text.lower()
        path_keywords = ['path', 'shortest', 'distance', 'route', 'from', 'to']
        if sum(1 for kw in path_keywords if kw in text_lower) < 2:
            return None

        # Parse edges: A-B=3, B-C=2  or  A to B: 3  or  A->B: 3
        edges: list[tuple[str, str, float]] = []
        patterns = [
            r'(\w)\s*[-–—]\s*(\w)\s*[=:]\s*(\d+\.?\d*)',
            r'(\w)\s*(?:to|→|->)\s*(\w)\s*[=:]\s*(\d+\.?\d*)',
        ]
        for pat in patterns:
            for m in re.finditer(pat, text):
                a, b, w = m.group(1).upper(), m.group(2).upper(), float(m.group(3))
                edges.append((a, b, w))

        if not edges:
            return None

        # Build adjacency list (undirected)
        graph: dict[str, list[tuple[str, float]]] = {}
        for a, b, w in edges:
            graph.setdefault(a, []).append((b, w))
            graph.setdefault(b, []).append((a, w))

        steps.append(f"Graph edges: {[(a, b, w) for a, b, w in edges]}")

        # Detect start/end from text
        nodes = sorted(graph.keys())
        start_node = nodes[0]
        end_node = nodes[-1]

        # Try to parse explicit start/end
        m_from = re.search(r'from\s+(\w)', text, re.IGNORECASE)
        m_to = re.search(r'to\s+(\w)', text, re.IGNORECASE)
        if m_from:
            start_node = m_from.group(1).upper()
        if m_to:
            end_node = m_to.group(1).upper()

        steps.append(f"Finding shortest path from {start_node} to {end_node}")

        # Dijkstra
        dist: dict[str, float] = {n: math.inf for n in graph}
        prev: dict[str, str | None] = {n: None for n in graph}
        dist[start_node] = 0
        pq: list[tuple[float, str]] = [(0, start_node)]

        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for v, w in graph.get(u, []):
                nd = d + w
                if nd < dist.get(v, math.inf):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))

        if dist.get(end_node, math.inf) == math.inf:
            steps.append(f"No path found from {start_node} to {end_node}")
            return None

        # Reconstruct path
        path: list[str] = []
        node: str | None = end_node
        while node is not None:
            path.append(node)
            node = prev[node]
        path.reverse()

        steps.append(f"Shortest path: {' → '.join(path)}")
        steps.append(f"Total distance: {dist[end_node]}")

        return {
            "path": " → ".join(path),
            "distance": dist[end_node],
        }

    # ------------------------------------------------------------------ #
    #                       Knapsack (0/1 DP)                             #
    # ------------------------------------------------------------------ #

    def _try_knapsack(self, text: str, steps: list[str]) -> Any:
        """Solve 0/1 knapsack."""
        if 'knapsack' not in text.lower() and 'capacity' not in text.lower():
            return None

        # Parse items: (value, weight) pairs
        nums = [float(n) for n in re.findall(r'\d+\.?\d*', text)]
        if len(nums) < 3:
            return None

        # Assume last number is capacity, rest alternate value/weight
        capacity = int(nums[-1])
        pairs = nums[:-1]
        if len(pairs) % 2 != 0:
            pairs = pairs[:-1]

        items: list[tuple[int, int]] = []
        for i in range(0, len(pairs), 2):
            items.append((int(pairs[i]), int(pairs[i + 1])))

        steps.append(f"Items (value, weight): {items}")
        steps.append(f"Capacity: {capacity}")

        n = len(items)
        dp = [[0] * (capacity + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            v, w = items[i - 1]
            for c in range(capacity + 1):
                dp[i][c] = dp[i - 1][c]
                if w <= c:
                    dp[i][c] = max(dp[i][c], dp[i - 1][c - w] + v)

        # Backtrack selected items
        selected: list[int] = []
        c = capacity
        for i in range(n, 0, -1):
            if dp[i][c] != dp[i - 1][c]:
                selected.append(i - 1)
                c -= items[i - 1][1]

        max_value = dp[n][capacity]
        steps.append(f"Max value: {max_value}, selected items: {selected}")

        return {"max_value": max_value, "selected_items": selected}

    # ------------------------------------------------------------------ #
    #               Numeric Optimisation (Simulated Annealing)            #
    # ------------------------------------------------------------------ #

    def _try_numeric_optimization(self, text: str, steps: list[str]) -> Any:
        """Optimize a numeric expression using simulated annealing."""
        text_lower = text.lower()
        opt_words = ['minimize', 'maximize', 'minimum', 'maximum', 'optimal']
        if not any(w in text_lower for w in opt_words):
            return None

        maximise = any(w in text_lower for w in ['maximize', 'maximum', 'largest', 'biggest'])

        # Try to extract a function like "x^2 - 4x + 5"
        expr_match = re.search(r'([\dx\^\s+\-*/().]+)', text)
        if not expr_match:
            return None

        expr = expr_match.group(1).replace('^', '**')
        if 'x' not in expr:
            return None

        steps.append(f"Objective: {'maximize' if maximise else 'minimize'} f(x) = {expr}")

        def f(x: float) -> float:
            try:
                return float(eval(expr.replace('x', f'({x})'), {"__builtins__": {}, "abs": abs}))  # noqa: S307
            except Exception:
                return float('inf')

        # Simulated annealing
        random.seed(42)
        x = random.uniform(-100, 100)
        best_x = x
        best_val = f(x)
        temp = 100.0
        cooling = 0.995

        for _ in range(5000):
            x_new = x + random.gauss(0, temp * 0.1)
            val_new = f(x_new)
            diff = val_new - f(x)
            if maximise:
                diff = -diff

            if diff < 0 or random.random() < math.exp(-diff / max(temp, 1e-10)):
                x = x_new
                val = f(x)
                if (maximise and val > best_val) or (not maximise and val < best_val):
                    best_x = x
                    best_val = val

            temp *= cooling

        best_x = round(best_x, 4)
        best_val = round(best_val, 4)
        steps.append(f"{'Maximum' if maximise else 'Minimum'} found at x = {best_x}, f(x) = {best_val}")

        return {"optimal_x": best_x, "optimal_value": best_val}
