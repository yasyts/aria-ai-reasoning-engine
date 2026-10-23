"""
Multi-Armed Bandit (UCB1) algorithm selector.

Learns which reasoning strategy works best for each problem type
using the Upper Confidence Bound (UCB1) exploration/exploitation algorithm.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

from ai_reasoning_engine.core.problem import ProblemType


class AlgorithmSelector:
    """UCB1-based meta-learner that selects the best strategy per problem type."""

    def __init__(self, data_dir: str = "aria_data") -> None:
        self._data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self._file = os.path.join(data_dir, "algorithm_weights.json")
        self._c = 1.414  # exploration constant (√2)

        # Per (strategy, problem_type): {total_reward, trials}
        self._stats: dict[str, dict[str, dict[str, float]]] = {}
        self._total_trials = 0
        self.load()

    # ------------------------------------------------------------------ #
    #                        public interface                             #
    # ------------------------------------------------------------------ #

    def get_strategy_weights(self, problem_type: ProblemType) -> dict[str, float]:
        """Return {strategy_name: UCB1_score} for this problem type."""
        pt_key = problem_type.name
        weights: dict[str, float] = {}

        for strategy_name in self._all_strategy_names():
            stats = self._get_stats(strategy_name, pt_key)
            if stats["trials"] == 0:
                weights[strategy_name] = float("inf")  # force exploration
            else:
                avg = stats["total_reward"] / stats["trials"]
                exploration = self._c * math.sqrt(
                    math.log(max(self._total_trials, 1)) / stats["trials"]
                )
                weights[strategy_name] = avg + exploration

        return weights

    def get_top_strategies(self, problem_type: ProblemType, n: int = 5) -> list[str]:
        """Return top N strategy names by UCB1 weight."""
        weights = self.get_strategy_weights(problem_type)
        ranked = sorted(weights.items(), key=lambda x: -x[1])
        return [name for name, _ in ranked[:n]]

    def update(self, strategy_name: str, problem_type: ProblemType, reward: float) -> None:
        """Record the reward (0-1) from a strategy on a problem type."""
        pt_key = problem_type.name
        stats = self._get_stats(strategy_name, pt_key)
        stats["total_reward"] += reward
        stats["trials"] += 1
        self._set_stats(strategy_name, pt_key, stats)
        self._total_trials += 1

    def register_strategy(self, name: str) -> None:
        """Ensure a strategy is tracked (called at engine init)."""
        if name not in self._stats:
            self._stats[name] = {}

    def reset(self) -> None:
        """Reset all weights."""
        self._stats = {}
        self._total_trials = 0
        self.save()

    # ------------------------------------------------------------------ #
    #                          persistence                                #
    # ------------------------------------------------------------------ #

    def save(self) -> None:
        data = {"stats": self._stats, "total_trials": self._total_trials}
        try:
            with open(self._file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except OSError:
            pass

    def load(self) -> None:
        if not os.path.exists(self._file):
            return
        try:
            with open(self._file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._stats = data.get("stats", {})
            self._total_trials = data.get("total_trials", 0)
        except (OSError, json.JSONDecodeError):
            pass

    # ------------------------------------------------------------------ #
    #                          internals                                  #
    # ------------------------------------------------------------------ #

    def _all_strategy_names(self) -> list[str]:
        return list(self._stats.keys()) if self._stats else []

    def _get_stats(self, strategy: str, pt_key: str) -> dict[str, float]:
        return self._stats.setdefault(strategy, {}).setdefault(
            pt_key, {"total_reward": 0.0, "trials": 0}
        )

    def _set_stats(self, strategy: str, pt_key: str, stats: dict[str, float]) -> None:
        self._stats.setdefault(strategy, {})[pt_key] = stats
