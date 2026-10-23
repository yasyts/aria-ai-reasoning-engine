#!/usr/bin/env python3
"""
Autonomous Training Pipeline on Real-World Data for ARIA.

Feeds real-world scientific, financial, logistics, health, and engineering
benchmark datasets into ARIA, validates candidate solutions, updates UCB1
meta-learner weights, and permanently expands ARIA's long-term memory.
"""

from __future__ import annotations

import os
import sys

# Ensure UTF-8 console output on Windows
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ai_reasoning_engine.core.engine import ARIAEngine
from ai_reasoning_engine.data.real_world_datasets import REAL_WORLD_BENCHMARKS, PHYSICAL_CONSTANTS, FINANCIAL_DATA
from ai_reasoning_engine.utils.helpers import colorize, format_time


def generate_augmented_real_world_problems() -> list[dict]:
    """Generate synthetic real-world variations to augment training data."""
    augmented = list(REAL_WORLD_BENCHMARKS)

    # Physics variations (varying mass and velocity)
    physics_cases = [
        (1500, 20, 300000.0),    # 0.5 * 1500 * 400 = 300,000 J
        (800, 30, 360000.0),     # 0.5 * 800 * 900 = 360,000 J
        (2000, 15, 225000.0),    # 0.5 * 2000 * 225 = 225,000 J
    ]
    for m, v, expected in physics_cases:
        augmented.append({
            "id": f"PHYS-AUG-{m}",
            "domain": "Physics",
            "problem": f"What is the kinetic energy of a {m} kg car travelling at {v} m/s?",
            "ground_truth": expected,
            "type": "MATHEMATICAL",
            "formula": "KE = 0.5 * m * v^2",
        })

    # Financial compound interest variations
    fin_cases = [
        (5000, 5, 3, 5788.13),     # 5000 * (1.05)^3 = 5788.125
        (20000, 8, 4, 27209.78),   # 20000 * (1.08)^4 = 27209.779
        (15000, 6, 2, 16854.00),   # 15000 * (1.06)^2 = 16854.0
    ]
    for p, r, t, expected in fin_cases:
        augmented.append({
            "id": f"FIN-AUG-{p}",
            "domain": "Personal Finance",
            "problem": f"Calculate compound interest total amount on ${p} invested at {r}% annual interest compounded annually for {t} years",
            "ground_truth": expected,
            "type": "MATHEMATICAL",
            "formula": "A = P * (1 + r)^t",
        })

    # Health BMI variations
    bmi_cases = [
        (65, 1.70, 22.49),
        (85, 1.80, 26.23),
        (55, 1.62, 20.96),
    ]
    for w, h, expected in bmi_cases:
        augmented.append({
            "id": f"MED-AUG-{w}",
            "domain": "Medicine / Nutrition",
            "problem": f"Calculate the Body Mass Index (BMI) for a person weighing {w} kg with height {h} meters",
            "ground_truth": expected,
            "type": "MATHEMATICAL",
            "formula": "BMI = weight / (height^2)",
        })

    # Temperature conversion variations
    temp_cases = [
        (32.0, 0.0),
        (212.0, 100.0),
        (68.0, 20.0),
    ]
    for f, expected in temp_cases:
        augmented.append({
            "id": f"CHEM-AUG-{int(f)}",
            "domain": "Unit Conversion",
            "problem": f"Convert {f} degrees Fahrenheit to Celsius using formula (F - 32) * 5 / 9",
            "ground_truth": expected,
            "type": "MATHEMATICAL",
            "formula": "C = (F - 32) * 5 / 9",
        })

    return augmented


def run_training_pipeline() -> None:
    data_dir = os.path.join(_PROJECT_ROOT, "aria_data")
    engine = ARIAEngine(data_dir=data_dir)

    dataset = generate_augmented_real_world_problems()
    total_samples = len(dataset)

    print(colorize("\n" + "=" * 70, "cyan"))
    print(colorize(f"  🧠 ARIA REAL-WORLD DATA TRAINING PIPELINE", "bold"))
    print(colorize(f"  Ingesting {total_samples} Verified Science, Finance & Logistics Cases", "dim"))
    print(colorize("=" * 70, "cyan") + "\n")

    correct = 0
    start_knowledge = engine._knowledge.get_count()

    for idx, item in enumerate(dataset, 1):
        prompt = item["problem"]
        domain = item["domain"]
        expected = item["ground_truth"]

        sol = engine.solve(prompt)
        ans = sol.answer

        # Check accuracy
        is_match = False
        try:
            if isinstance(expected, (int, float)):
                # If ans is dict, check if any value or key matches
                if isinstance(ans, dict):
                    val = next((v for v in ans.values() if isinstance(v, (int, float))), 0)
                    is_match = abs(float(val) - float(expected)) <= max(abs(float(expected)) * 0.05, 1.0)
                elif ans is not None:
                    is_match = abs(float(ans) - float(expected)) <= max(abs(float(expected)) * 0.05, 1.0)
            elif isinstance(expected, str):
                is_match = str(expected).lower() in str(ans).lower()
        except Exception:
            is_match = False

        if is_match:
            correct += 1
            status = colorize("✅ VERIFIED", "green")
        else:
            status = colorize("🟡 APPROX", "yellow")

        ans_str = str(ans)[:40] if ans is not None else "None"
        print(f"[{idx:02d}/{total_samples}] [{domain:<20}] {status} -> Ans: {ans_str} ({sol.strategy_name}, {format_time(sol.execution_time)})")

    # Autonomous Self-Training Loop
    print(colorize("\n" + "-" * 70, "dim"))
    print(colorize("  🔄 Executing Meta-Learning Weight Adaptation Cycle...", "bold"))
    report = engine.train()
    print(f"  Training Epoch: {report.get('training_round')}")
    print(f"  Applied Prior Adjustments: {report.get('adjustments_applied')}")
    if report.get('strategies_boosted'):
        print(f"  Boosted Strategies: {report.get('strategies_boosted')}")
    if report.get('knowledge_gaps'):
        print(f"  Detected Knowledge Gaps: {report.get('knowledge_gaps')}")

    end_knowledge = engine._knowledge.get_count()
    stats = engine.get_stats()

    print(colorize("\n" + "=" * 70, "green"))
    print(colorize("  🎯 TRAINING SESSION COMPLETED", "bold"))
    print(colorize("=" * 70, "green"))
    print(f"  Total Real-World Cases Trained : {total_samples}")
    print(f"  Empirical Accuracy Rate         : {correct}/{total_samples} ({correct/total_samples*100:.1f}%)")
    print(f"  Knowledge Base Memory Growth   : {start_knowledge} → {end_knowledge} cases")
    print(f"  Global Solves Logged           : {stats['overall']['total']}")
    print(f"  Average Solve Time             : {format_time(stats['overall']['avg_time'])}")
    print(colorize("=" * 70, "green") + "\n")


if __name__ == "__main__":
    run_training_pipeline()
