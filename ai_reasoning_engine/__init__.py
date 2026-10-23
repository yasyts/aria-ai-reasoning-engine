"""
ARIA — Adaptive Reasoning & Intelligence Architecture

A self-improving AI reasoning engine that uses multiple strategies,
meta-learning algorithm selection, and self-training to find optimal
solutions for any problem.
"""

from ai_reasoning_engine.core.engine import ARIAEngine
from ai_reasoning_engine.core.problem import Problem, ProblemType
from ai_reasoning_engine.core.solution import Solution

__version__ = "1.0.0"
__all__ = ["ARIAEngine", "Problem", "ProblemType", "Solution"]
