"""
ARIA Engine — the main orchestrator.

Ties together all strategies, the meta-learner, evaluator,
performance tracker, self-trainer, and knowledge base.
"""

from __future__ import annotations

import time
from typing import Any

from ai_reasoning_engine.core.problem import Problem, ProblemType
from ai_reasoning_engine.core.solution import Solution
from ai_reasoning_engine.strategies.base import BaseStrategy
from ai_reasoning_engine.strategies.mathematical import MathematicalStrategy
from ai_reasoning_engine.strategies.logical import LogicalStrategy
from ai_reasoning_engine.strategies.heuristic import HeuristicStrategy
from ai_reasoning_engine.strategies.pattern import PatternStrategy
from ai_reasoning_engine.strategies.analogical import AnalogicalStrategy
from ai_reasoning_engine.strategies.real_world import RealWorldStrategy
from ai_reasoning_engine.meta.algorithm_selector import AlgorithmSelector
from ai_reasoning_engine.meta.performance_tracker import PerformanceTracker
from ai_reasoning_engine.meta.self_trainer import SelfTrainer
from ai_reasoning_engine.evaluation.evaluator import SolutionEvaluator
from ai_reasoning_engine.knowledge.knowledge_base import KnowledgeBase


class ARIAEngine:
    """
    Adaptive Reasoning & Intelligence Architecture.

    The core orchestrator that:
    1. Classifies problems automatically
    2. Selects top strategies via UCB1 meta-learner
    3. Runs all viable strategies in parallel
    4. Evaluates and ranks all solutions
    5. Picks the single best solution
    6. Records results for self-improvement
    7. Triggers self-training periodically
    """

    def __init__(self, data_dir: str = "aria_data") -> None:
        self._data_dir = data_dir
        self._log_buffer: list[str] = []

        # Sub-systems
        self._selector = AlgorithmSelector(data_dir)
        self._tracker = PerformanceTracker(data_dir)
        self._evaluator = SolutionEvaluator()
        self._knowledge = KnowledgeBase(data_dir)

        # Strategies
        self._strategies: list[BaseStrategy] = []
        self._analogical = AnalogicalStrategy()

        # Register default strategies
        self._register_defaults()

        # Self-trainer (must come after strategies are registered)
        self._trainer = SelfTrainer(
            self._selector, self._tracker, self._knowledge, train_interval=15
        )

        # Load existing knowledge into analogical strategy
        for entry in self._knowledge.get_all():
            self._analogical.add_knowledge(entry.get("problem_text", ""), entry.get("answer"))

        self._log("ARIA Engine initialised")
        self._log(f"  Strategies : {len(self._strategies)}")
        self._log(f"  Knowledge  : {self._knowledge.get_count()} entries")
        self._log(f"  History    : {self._tracker.get_history_count()} solves")

    # ------------------------------------------------------------------ #
    #                           public API                                #
    # ------------------------------------------------------------------ #

    def solve(self, problem_text: str) -> Solution:
        """
        Main entry point. Takes problem text and returns the best solution.
        """
        # 1. Create & classify problem
        problem = Problem.from_text(problem_text)
        self._log(f"\n{'='*60}")
        self._log(f"Problem: {problem.description[:80]}")
        self._log(f"Type   : {problem.problem_type.name}")

        # 2. Get strategy weights from meta-learner
        weights = self._selector.get_strategy_weights(problem.problem_type)

        # 3. Select strategies to run (all that can handle this type)
        runnable = [s for s in self._strategies if s.can_handle(problem)]
        if not runnable:
            runnable = self._strategies  # fallback: try everything

        # Sort by weight (highest first)
        runnable.sort(key=lambda s: -weights.get(s.name, 0))
        self._log(f"Running {len(runnable)} strategies: {[s.name for s in runnable]}")

        # 4. Run all strategies, collect solutions
        solutions: list[Solution] = []
        for strategy in runnable:
            try:
                self._log(f"  → {strategy.name}...", end=" ")
                sol = strategy.solve(problem)
                solutions.append(sol)
                self._log(f"done ({sol.execution_time:.4f}s, conf={sol.confidence:.2f})")
            except Exception as exc:
                self._log(f"FAILED: {exc}")
                solutions.append(Solution(
                    answer=None, strategy_name=strategy.name,
                    confidence=0.0, reasoning_steps=[f"Strategy crashed: {exc}"],
                    execution_time=0.0,
                ))

        # 5. Evaluate and rank
        ranked = self._evaluator.evaluate(solutions, problem)
        best = ranked[0] if ranked else Solution(
            answer=None, strategy_name="none",
            confidence=0.0, reasoning_steps=["All strategies failed"],
        )

        self._log(f"\nBest solution: [{best.strategy_name}] answer={best.answer} "
                   f"score={best.score:.3f}")

        # 6. Record in performance tracker
        self._tracker.record(problem, best)

        # 7. Update algorithm selector with rewards
        for sol in ranked:
            self._selector.update(sol.strategy_name, problem.problem_type, sol.score)
        self._selector.save()

        # 8. Self-train if enough data
        if self._trainer.should_train():
            self._log("\n🔄 Auto self-training triggered...")
            report = self._trainer.train()
            self._log(f"   Training round {report['training_round']}: "
                       f"{report['adjustments_applied']} adjustments")
            if report['knowledge_gaps']:
                self._log(f"   Knowledge gaps: {report['knowledge_gaps']}")

        # 9. Store in knowledge base
        self._knowledge.store(problem, best)
        self._analogical.add_knowledge(problem.description, best.answer)

        return best

    def solve_detailed(self, problem_text: str) -> dict[str, Any]:
        """
        Executes problem solving and returns full detailed diagnostic package
        for the web dashboard (candidates, evaluations, weights, and logs).
        """
        self.clear_log()
        # 1. Create & classify problem
        problem = Problem.from_text(problem_text)
        self._log(f"Received Problem: \"{problem.description}\"")
        self._log(f"Categorization: {problem.problem_type.name} (confidence {problem.data.get('classification_confidence', 0):.1%})")

        # 2. Get weights
        weights = self._selector.get_strategy_weights(problem.problem_type)
        # Format weights safely for JSON
        serializable_weights = {
            k: (999.0 if v == float("inf") else round(v, 4))
            for k, v in weights.items()
        }

        # 3. Select strategies
        runnable = [s for s in self._strategies if s.can_handle(problem)]
        if not runnable:
            runnable = self._strategies
        runnable.sort(key=lambda s: -weights.get(s.name, 0))
        self._log(f"Selected candidate strategies: {[s.name for s in runnable]}")

        # 4. Run strategies
        solutions: list[Solution] = []
        for strategy in runnable:
            try:
                self._log(f"Running strategy: {strategy.name}...")
                sol = strategy.solve(problem)
                solutions.append(sol)
                self._log(f"  → Result from {strategy.name}: answer={sol.answer}, conf={sol.confidence:.2f}, time={sol.execution_time:.5f}s")
            except Exception as exc:
                self._log(f"  → Strategy {strategy.name} encountered error: {exc}")
                solutions.append(Solution(
                    answer=None, strategy_name=strategy.name,
                    confidence=0.0, reasoning_steps=[f"Error: {exc}"],
                    execution_time=0.0,
                ))

        # 5. Evaluate and rank
        ranked = self._evaluator.evaluate(solutions, problem)
        best = ranked[0] if ranked else Solution(
            answer=None, strategy_name="none",
            confidence=0.0, reasoning_steps=["No valid solution was produced"],
        )

        self._log(f"Selected best solution: [{best.strategy_name}] with evaluated score {best.score:.3f}")

        # 6. Record in tracker
        self._tracker.record(problem, best)

        # 7. Update algorithm selector
        for sol in ranked:
            self._selector.update(sol.strategy_name, problem.problem_type, sol.score)
        self._selector.save()

        # 8. Check auto-train
        trained = False
        training_report = None
        if self._trainer.should_train():
            self._log("Self-training triggered automatically based on accumulated experience.")
            training_report = self._trainer.train()
            trained = True

        # 9. Store in knowledge base
        self._knowledge.store(problem, best)
        self._analogical.add_knowledge(problem.description, best.answer)

        return {
            "problem": problem.to_dict(),
            "best_solution": best.to_dict(),
            "candidate_solutions": [s.to_dict() for s in ranked],
            "strategy_weights": serializable_weights,
            "logs": list(self._log_buffer),
            "auto_trained": trained,
            "training_report": training_report,
        }

    def add_strategy(self, strategy: BaseStrategy) -> None:
        """Register a new reasoning strategy."""
        self._strategies.append(strategy)
        self._selector.register_strategy(strategy.name)
        self._log(f"Added strategy: {strategy.name}")

    def get_stats(self) -> dict[str, Any]:
        """Return comprehensive performance statistics."""
        return {
            "overall": self._tracker.get_overall_stats(),
            "by_strategy": self._tracker.get_strategy_stats(),
            "by_problem_type": self._tracker.get_problem_type_stats(),
            "knowledge_count": self._knowledge.get_count(),
            "training_rounds": self._trainer.training_count,
            "improvement_trend": self._tracker.get_improvement_trend(),
        }

    def train(self) -> dict[str, Any]:
        """Manually trigger self-training."""
        return self._trainer.train()

    def get_log(self) -> list[str]:
        """Return the internal log buffer."""
        return list(self._log_buffer)

    def clear_log(self) -> None:
        self._log_buffer.clear()

    # ------------------------------------------------------------------ #
    #                          internals                                  #
    # ------------------------------------------------------------------ #

    def _register_defaults(self) -> None:
        """Register all built-in strategies."""
        defaults: list[BaseStrategy] = [
            MathematicalStrategy(),
            LogicalStrategy(),
            HeuristicStrategy(),
            PatternStrategy(),
            RealWorldStrategy(),
            self._analogical,
        ]
        for s in defaults:
            self._strategies.append(s)
            self._selector.register_strategy(s.name)

    def _log(self, msg: str, end: str = "\n") -> None:
        self._log_buffer.append(msg)
