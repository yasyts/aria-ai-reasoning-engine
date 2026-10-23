"""
Performance tracker for ARIA.

Records every solve attempt and computes per-strategy and
per-problem-type statistics plus improvement trends.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

from ai_reasoning_engine.core.problem import Problem
from ai_reasoning_engine.core.solution import Solution


class PerformanceTracker:
    """Tracks strategy performance over time."""

    def __init__(self, data_dir: str = "aria_data") -> None:
        self._data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self._file = os.path.join(data_dir, "performance_history.json")
        self._history: list[dict[str, Any]] = []
        self.load()

    # ------------------------------------------------------------------ #
    #                        recording                                    #
    # ------------------------------------------------------------------ #

    def record(self, problem: Problem, solution: Solution) -> None:
        """Store a solve result."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "problem_id": problem.id,
            "problem_type": problem.problem_type.name,
            "problem_text": problem.description[:200],
            "strategy": solution.strategy_name,
            "score": solution.score,
            "confidence": solution.confidence,
            "execution_time": solution.execution_time,
            "answer": str(solution.answer)[:200] if solution.answer is not None else None,
        }
        self._history.append(entry)
        self.save()

    # ------------------------------------------------------------------ #
    #                        statistics                                   #
    # ------------------------------------------------------------------ #

    def get_strategy_stats(self) -> dict[str, dict[str, Any]]:
        """Per-strategy: avg score, avg time, count, success rate."""
        stats: dict[str, dict[str, Any]] = {}
        for entry in self._history:
            s = entry["strategy"]
            if s not in stats:
                stats[s] = {"scores": [], "times": [], "count": 0, "successes": 0}
            stats[s]["scores"].append(entry["score"])
            stats[s]["times"].append(entry["execution_time"])
            stats[s]["count"] += 1
            if entry["score"] > 0.5:
                stats[s]["successes"] += 1

        result: dict[str, dict[str, Any]] = {}
        for s, data in stats.items():
            n = data["count"]
            result[s] = {
                "avg_score": sum(data["scores"]) / n if n else 0,
                "avg_time": sum(data["times"]) / n if n else 0,
                "count": n,
                "success_rate": data["successes"] / n if n else 0,
            }
        return result

    def get_problem_type_stats(self) -> dict[str, dict[str, Any]]:
        """Per problem type: best strategy, avg score."""
        by_type: dict[str, dict[str, list[float]]] = {}
        for entry in self._history:
            pt = entry["problem_type"]
            strat = entry["strategy"]
            by_type.setdefault(pt, {}).setdefault(strat, []).append(entry["score"])

        result: dict[str, dict[str, Any]] = {}
        for pt, strats in by_type.items():
            avg_scores = {s: sum(v) / len(v) for s, v in strats.items()}
            best = max(avg_scores, key=avg_scores.get)  # type: ignore[arg-type]
            all_scores = [sc for scores in strats.values() for sc in scores]
            result[pt] = {
                "best_strategy": best,
                "best_avg_score": avg_scores[best],
                "overall_avg_score": sum(all_scores) / len(all_scores),
                "total_attempts": len(all_scores),
            }
        return result

    def get_overall_stats(self) -> dict[str, Any]:
        """Global summary statistics."""
        if not self._history:
            return {"total": 0, "avg_score": 0, "avg_time": 0, "success_rate": 0}

        scores = [e["score"] for e in self._history]
        times = [e["execution_time"] for e in self._history]
        successes = sum(1 for s in scores if s > 0.5)

        return {
            "total": len(self._history),
            "avg_score": sum(scores) / len(scores),
            "avg_time": sum(times) / len(times),
            "success_rate": successes / len(scores),
        }

    def get_improvement_trend(self, window: int = 10) -> list[float]:
        """Rolling average of scores to show improvement over time."""
        scores = [e["score"] for e in self._history]
        if len(scores) < window:
            return [sum(scores) / len(scores)] if scores else []

        trend: list[float] = []
        for i in range(len(scores) - window + 1):
            chunk = scores[i : i + window]
            trend.append(sum(chunk) / len(chunk))
        return trend

    def get_history_count(self) -> int:
        return len(self._history)

    def get_recent(self, n: int = 10) -> list[dict[str, Any]]:
        """Return last N history entries."""
        return self._history[-n:]

    # ------------------------------------------------------------------ #
    #                        persistence                                  #
    # ------------------------------------------------------------------ #

    def save(self) -> None:
        try:
            with open(self._file, "w", encoding="utf-8") as f:
                json.dump(self._history, f, indent=2)
        except OSError:
            pass

    def load(self) -> None:
        if not os.path.exists(self._file):
            return
        try:
            with open(self._file, "r", encoding="utf-8") as f:
                self._history = json.load(f)
        except (OSError, json.JSONDecodeError):
            self._history = []
