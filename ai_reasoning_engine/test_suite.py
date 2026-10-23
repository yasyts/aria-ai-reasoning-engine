"""
Comprehensive test suite for ARIA (Adaptive Reasoning & Intelligence Architecture).
Tests all strategies, meta-learning, solution evaluation, and self-improvement loop.
"""

import sys
import os
import shutil

# Ensure utf-8 output encoding on Windows consoles
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from ai_reasoning_engine.core.engine import ARIAEngine
from ai_reasoning_engine.core.problem import Problem, ProblemType

def run_tests():
    test_data_dir = os.path.join(WORKSPACE_ROOT, "test_aria_data")
    if os.path.exists(test_data_dir):
        shutil.rmtree(test_data_dir)

    print("🚀 Initializing ARIA Engine for test suite...")
    engine = ARIAEngine(data_dir=test_data_dir)

    test_cases = [
        # 1. Mathematical - Arithmetic
        {
            "prompt": "Calculate 15 * 23 + 47 - 12",
            "expected_type": ProblemType.MATHEMATICAL,
            "validate": lambda ans: abs(float(ans) - 380.0) < 1e-4,
            "desc": "Arithmetic expression evaluation",
        },
        # 2. Mathematical - Linear equation
        {
            "prompt": "Solve the equation 2x + 5 = 15",
            "expected_type": ProblemType.MATHEMATICAL,
            "validate": lambda ans: abs(float(ans) - 5.0) < 1e-4,
            "desc": "Linear algebra solver",
        },
        # 3. Mathematical - Quadratic
        {
            "prompt": "Find the roots of x^2 - 5x + 6 = 0",
            "expected_type": ProblemType.MATHEMATICAL,
            "validate": lambda ans: (isinstance(ans, (tuple, list)) and set(round(x, 1) for x in ans) == {2.0, 3.0})
                                    or ("2" in str(ans) and "3" in str(ans)),
            "desc": "Quadratic equation solver",
        },
        # 4. Mathematical - Factorial
        {
            "prompt": "What is the factorial of 10?",
            "expected_type": ProblemType.MATHEMATICAL,
            "validate": lambda ans: int(ans) == 3628800,
            "desc": "Factorial calculation",
        },
        # 5. Pattern - Geometric Sequence
        {
            "prompt": "Find the next number in the sequence: 2, 6, 18, 54",
            "expected_type": ProblemType.SEQUENCE,
            "validate": lambda ans: (isinstance(ans, dict) and abs(ans.get("next_value", 0) - 162) < 1e-4)
                                    or "162" in str(ans),
            "desc": "Geometric sequence recognition",
        },
        # 6. Logical - Syllogism / Category Inference
        {
            "prompt": "If all dogs are animals, and Rex is a dog, what is Rex?",
            "expected_type": ProblemType.LOGICAL,
            "validate": lambda ans: "animal" in str(ans).lower(),
            "desc": "Deductive logical syllogism",
        },
        # 7. Heuristic - Shortest Path
        {
            "prompt": "Find the shortest path from A to D given: A-B=3, B-C=2, A-C=6, C-D=1, B-D=5",
            "expected_type": ProblemType.OPTIMIZATION,
            "validate": lambda ans: (isinstance(ans, dict) and ans.get("distance") == 6.0)
                                    or "6" in str(ans),
            "desc": "Dijkstra shortest path search",
        },
        # 8. Mathematical - Statistics
        {
            "prompt": "What is the mean and standard deviation of: 10, 20, 30, 40, 50",
            "expected_type": ProblemType.MATHEMATICAL,
            "validate": lambda ans: (isinstance(ans, dict) and abs(ans.get("mean", 0) - 30.0) < 1e-2)
                                    or "30" in str(ans),
            "desc": "Statistical distribution analysis",
        },
    ]

    passed = 0
    total = len(test_cases)

    print("\n" + "="*70)
    print("RUNNING REASONING AND DEDUCTION TESTS")
    print("="*70)

    for i, test in enumerate(test_cases, 1):
        print(f"\n[Test {i}/{total}] {test['desc']}")
        print(f"  Input: \"{test['prompt']}\"")
        solution = engine.solve(test["prompt"])
        print(f"  Strategy selected: {solution.strategy_name}")
        print(f"  Answer: {solution.answer}")
        print(f"  Confidence: {solution.confidence:.2%}")
        print(f"  Score: {solution.score:.3f}")
        print(f"  Reasoning steps: {len(solution.reasoning_steps)}")
        for step in solution.reasoning_steps:
            print(f"    • {step}")

        try:
            is_valid = test["validate"](solution.answer)
        except Exception as e:
            is_valid = False
            print(f"  Validation error: {e}")

        if is_valid:
            print("  Status: ✅ PASSED")
            passed += 1
        else:
            print("  Status: ❌ FAILED")

    # 9. Test Self-Training and Knowledge Base
    print("\n" + "="*70)
    print("TESTING ANALOGICAL TRANSFER & SELF-TRAINING")
    print("="*70)

    # Re-ask a similar problem to test Analogical reasoning recall
    analogical_prompt = "Calculate 15 * 23 + 47 - 12 please"
    print(f"\n[Analogical Test] Querying similar problem: \"{analogical_prompt}\"")
    analogical_sol = engine.solve(analogical_prompt)
    print(f"  Solution strategy: {analogical_sol.strategy_name}")
    print(f"  Answer: {analogical_sol.answer}")
    print(f"  Score: {analogical_sol.score:.3f}")

    # Trigger manual self-train
    print("\n[Self-Training Test] Running manual training cycle...")
    train_report = engine.train()
    print(f"  Training round: {train_report.get('training_round')}")
    print(f"  Adjustments applied: {train_report.get('adjustments_applied')}")
    print(f"  Strategies boosted: {train_report.get('strategies_boosted')}")

    # Stats inspection
    stats = engine.get_stats()
    print("\n[Engine Stats Summary]")
    print(f"  Total solved: {stats['overall']['total']}")
    print(f"  Average score: {stats['overall']['avg_score']:.3f}")
    print(f"  Knowledge entries: {stats['knowledge_count']}")

    print("\n" + "="*70)
    print(f"FINAL RESULT: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("="*70)

    # Cleanup test dir
    if os.path.exists(test_data_dir):
        shutil.rmtree(test_data_dir)

    return passed == total

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
