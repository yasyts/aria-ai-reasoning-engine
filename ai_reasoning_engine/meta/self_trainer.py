"""
Self-training engine for ARIA.

Analyses accumulated performance data and adjusts strategy weights
in the AlgorithmSelector to continuously improve.
"""

from __future__ import annotations

from typing import Any

from ai_reasoning_engine.meta.algorithm_selector import AlgorithmSelector
from ai_reasoning_engine.meta.performance_tracker import PerformanceTracker


class SelfTrainer:
    """Self-improvement loop that tunes the meta-learner from experience."""

    def __init__(
        self,
        algorithm_selector: AlgorithmSelector,
        performance_tracker: PerformanceTracker,
        knowledge_base: Any = None,
        train_interval: int = 20,
    ) -> None:
        self._selector = algorithm_selector
        self._tracker = performance_tracker
        self._knowledge_base = knowledge_base
        self._train_interval = train_interval
        self._last_train_count = 0
        self.training_count = 0
        self._last_report: dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    #                        public interface                             #
    # ------------------------------------------------------------------ #

    def should_train(self) -> bool:
        """True if enough new data since last training."""
        current = self._tracker.get_history_count()
        return (current - self._last_train_count) >= self._train_interval

    def train(self) -> dict[str, Any]:
        """Core self-training loop."""
        analysis = self._analyze_strategy_performance()
        adjustments = self._compute_reward_adjustments(analysis)
        self._apply_adjustments(adjustments)

        self._last_train_count = self._tracker.get_history_count()
        self.training_count += 1

        # Find knowledge gaps
        gaps = self._find_knowledge_gaps(analysis)

        report = {
            "training_round": self.training_count,
            "strategies_analysed": len(analysis),
            "adjustments_applied": len(adjustments),
            "strategies_boosted": [a for a in adjustments if a[2] > 0],
            "strategies_penalized": [a for a in adjustments if a[2] < 0],
            "knowledge_gaps": gaps,
            "improvement_trend": self._tracker.get_improvement_trend(),
        }
        self._last_report = report
        self._selector.save()
        return report

    def get_training_report(self) -> dict[str, Any]:
        return self._last_report

    # ------------------------------------------------------------------ #
    #                          internals                                  #
    # ------------------------------------------------------------------ #

    def _analyze_strategy_performance(self) -> dict[str, dict[str, Any]]:
        """Compute per-strategy-per-type metrics from history."""
        history = self._tracker.get_recent(200)
        analysis: dict[str, dict[str, Any]] = {}

        for entry in history:
            key = f"{entry['strategy']}|{entry['problem_type']}"
            if key not in analysis:
                analysis[key] = {
                    "strategy": entry["strategy"],
                    "problem_type": entry["problem_type"],
                    "scores": [],
                    "times": [],
                }
            analysis[key]["scores"].append(entry["score"])
            analysis[key]["times"].append(entry["execution_time"])

        # Compute averages
        for key, data in analysis.items():
            n = len(data["scores"])
            data["avg_score"] = sum(data["scores"]) / n
            data["avg_time"] = sum(data["times"]) / n
            data["count"] = n
            # Trend: is it improving?
            if n >= 4:
                first_half = data["scores"][:n // 2]
                second_half = data["scores"][n // 2:]
                data["trend"] = (sum(second_half) / len(second_half)) - (sum(first_half) / len(first_half))
            else:
                data["trend"] = 0.0

        return analysis

    def _compute_reward_adjustments(
        self, analysis: dict[str, dict[str, Any]]
    ) -> list[tuple[str, str, float]]:
        """Determine weight changes: (strategy, problem_type, adjustment)."""
        adjustments: list[tuple[str, str, float]] = []

        # Find global average
        all_scores = [d["avg_score"] for d in analysis.values()]
        global_avg = sum(all_scores) / len(all_scores) if all_scores else 0.5

        for key, data in analysis.items():
            strategy = data["strategy"]
            pt = data["problem_type"]
            avg = data["avg_score"]
            trend = data["trend"]

            if avg > global_avg + 0.1 and data["count"] >= 3:
                # Boost strong performers
                boost = min(0.15, (avg - global_avg) * 0.5)
                adjustments.append((strategy, pt, boost))
            elif avg < global_avg - 0.1 and data["count"] >= 3:
                # Penalize weak performers
                penalty = max(-0.15, (avg - global_avg) * 0.5)
                adjustments.append((strategy, pt, penalty))
            elif trend > 0.1:
                # Improving strategy gets a small boost
                adjustments.append((strategy, pt, 0.05))

        return adjustments

    def _apply_adjustments(self, adjustments: list[tuple[str, str, float]]) -> None:
        """Apply reward adjustments to the algorithm selector."""
        from ai_reasoning_engine.core.problem import ProblemType

        for strategy, pt_name, adjustment in adjustments:
            try:
                pt = ProblemType[pt_name]
                # Apply as a synthetic reward update
                if adjustment > 0:
                    self._selector.update(strategy, pt, min(1.0, 0.7 + adjustment))
                else:
                    self._selector.update(strategy, pt, max(0.0, 0.3 + adjustment))
            except (KeyError, ValueError):
                pass

    def _find_knowledge_gaps(self, analysis: dict[str, dict[str, Any]]) -> list[str]:
        """Identify problem types where all strategies score poorly."""
        from ai_reasoning_engine.core.problem import ProblemType

        type_best: dict[str, float] = {}
        for data in analysis.values():
            pt = data["problem_type"]
            if pt not in type_best or data["avg_score"] > type_best[pt]:
                type_best[pt] = data["avg_score"]

        gaps: list[str] = []
        for pt, best_score in type_best.items():
            if best_score < 0.4:
                gaps.append(f"{pt}: best strategy scores only {best_score:.2f}")

        return gaps
