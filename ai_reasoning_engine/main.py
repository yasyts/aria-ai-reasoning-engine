#!/usr/bin/env python3
"""
ARIA — Adaptive Reasoning & Intelligence Architecture
Interactive CLI

Usage:
    python main.py
"""

from __future__ import annotations

import os
import sys

# Ensure utf-8 output encoding on Windows consoles
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add project root to path so imports work when running directly
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ai_reasoning_engine.core.engine import ARIAEngine
from ai_reasoning_engine.utils.helpers import colorize, format_time


# ------------------------------------------------------------------ #
#                            constants                                #
# ------------------------------------------------------------------ #

BANNER = r"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║       █████╗ ██████╗ ██╗ █████╗                              ║
    ║      ██╔══██╗██╔══██╗██║██╔══██╗                             ║
    ║      ███████║██████╔╝██║███████║                             ║
    ║      ██╔══██║██╔══██╗██║██╔══██║                             ║
    ║      ██║  ██║██║  ██║██║██║  ██║                             ║
    ║      ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝                             ║
    ║                                                              ║
    ║   Adaptive Reasoning & Intelligence Architecture  v1.0       ║
    ║   Self-improving AI  •  Multi-strategy  •  Meta-learning     ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
"""

HELP_TEXT = """
  Commands:
    <problem>  — Type any problem to solve it
    demo       — Run demo problems to showcase abilities
    stats      — Show performance statistics
    train      — Manually trigger self-training
    history    — Show recent solved problems
    knowledge  — Show knowledge base stats
    help       — Show this help
    quit/exit  — Exit ARIA
"""

DEMO_PROBLEMS = [
    "Calculate 15 * 23 + 47 - 12",
    "Solve the equation 2x + 5 = 15",
    "Find the next number in the sequence: 2, 6, 18, 54",
    "If all dogs are animals, and Rex is a dog, what is Rex?",
    "Find the shortest path from A to D given: A-B=3, B-C=2, A-C=6, C-D=1, B-D=5",
    "What is the factorial of 10?",
    "Find the roots of x^2 - 5x + 6 = 0",
    "What is the mean and standard deviation of: 10, 20, 30, 40, 50",
]


# ------------------------------------------------------------------ #
#                          display helpers                            #
# ------------------------------------------------------------------ #

def print_header(text: str) -> None:
    print(f"\n{colorize('═' * 60, 'cyan')}")
    print(f"  {colorize(text, 'bold')}")
    print(f"{colorize('═' * 60, 'cyan')}")


def print_solution(solution, engine: ARIAEngine) -> None:
    """Pretty-print a solution with reasoning chain."""
    print()
    print(colorize("┌─────────────────────────────────────────────────────┐", "green"))
    print(colorize("│                    BEST SOLUTION                    │", "green"))
    print(colorize("└─────────────────────────────────────────────────────┘", "green"))

    print(f"  {colorize('Strategy:', 'bold')}   {solution.strategy_name}")
    print(f"  {colorize('Answer:', 'bold')}     {colorize(str(solution.answer), 'yellow')}")
    print(f"  {colorize('Confidence:', 'bold')} {_confidence_bar(solution.confidence)}")
    print(f"  {colorize('Score:', 'bold')}      {solution.score:.3f}")
    print(f"  {colorize('Time:', 'bold')}       {format_time(solution.execution_time)}")

    if solution.reasoning_steps:
        print(f"\n  {colorize('Reasoning Chain:', 'bold')}")
        for i, step in enumerate(solution.reasoning_steps, 1):
            print(f"    {colorize(f'{i}.', 'dim')} {step}")

    # Show engine log for detailed view
    log = engine.get_log()
    if log:
        print(f"\n  {colorize('Engine Log:', 'dim')}")
        for line in log[-15:]:  # last 15 lines
            print(f"    {colorize(line, 'dim')}")
    engine.clear_log()
    print()


def _confidence_bar(confidence: float) -> str:
    """Visual confidence bar."""
    filled = int(confidence * 20)
    bar = "█" * filled + "░" * (20 - filled)
    pct = f"{confidence * 100:.0f}%"
    if confidence >= 0.8:
        color = "green"
    elif confidence >= 0.5:
        color = "yellow"
    else:
        color = "red"
    return f"{colorize(bar, color)} {pct}"


def show_stats(engine: ARIAEngine) -> None:
    """Display performance statistics."""
    stats = engine.get_stats()
    print_header("Performance Statistics")

    overall = stats["overall"]
    print(f"\n  {colorize('Overall:', 'bold')}")
    print(f"    Total problems solved : {overall['total']}")
    print(f"    Average score         : {overall['avg_score']:.3f}")
    print(f"    Average time          : {format_time(overall['avg_time'])}")
    print(f"    Success rate          : {overall['success_rate'] * 100:.1f}%")
    print(f"    Knowledge entries     : {stats['knowledge_count']}")
    print(f"    Training rounds       : {stats['training_rounds']}")

    by_strat = stats["by_strategy"]
    if by_strat:
        print(f"\n  {colorize('By Strategy:', 'bold')}")
        print(f"    {'Strategy':<15} {'Avg Score':>10} {'Avg Time':>10} {'Count':>6} {'Success':>8}")
        print(f"    {'─' * 49}")
        for name, s in sorted(by_strat.items(), key=lambda x: -x[1]["avg_score"]):
            print(f"    {name:<15} {s['avg_score']:>10.3f} {format_time(s['avg_time']):>10} "
                  f"{s['count']:>6} {s['success_rate'] * 100:>7.1f}%")

    by_type = stats["by_problem_type"]
    if by_type:
        print(f"\n  {colorize('By Problem Type:', 'bold')}")
        for pt, s in by_type.items():
            print(f"    {pt}: best → {s['best_strategy']} "
                  f"(avg score {s['best_avg_score']:.3f}, {s['total_attempts']} attempts)")

    trend = stats["improvement_trend"]
    if trend and len(trend) >= 2:
        direction = "📈 Improving" if trend[-1] > trend[0] else "📉 Declining"
        print(f"\n  {colorize('Trend:', 'bold')} {direction} "
              f"({trend[0]:.3f} → {trend[-1]:.3f})")

    print()


def show_history(engine: ARIAEngine) -> None:
    """Show recent solved problems."""
    print_header("Recent History")
    recent = engine._tracker.get_recent(10)
    if not recent:
        print("  No history yet. Solve some problems first!\n")
        return

    for i, entry in enumerate(recent, 1):
        score = entry['score']
        if score >= 0.7:
            emoji = "✅"
        elif score >= 0.4:
            emoji = "🟡"
        else:
            emoji = "❌"
        print(f"  {emoji} {i}. [{entry['strategy']}] {entry.get('problem_text', '')[:60]}")
        print(f"       Score: {score:.3f} | Time: {format_time(entry['execution_time'])}")
    print()


def run_demo(engine: ARIAEngine) -> None:
    """Run all demo problems."""
    print_header("Demo Mode — Solving 8 diverse problems")
    for i, problem_text in enumerate(DEMO_PROBLEMS, 1):
        print(f"\n{colorize(f'── Demo {i}/{len(DEMO_PROBLEMS)} ──', 'magenta')}")
        print(f"  {colorize('Problem:', 'bold')} {problem_text}")
        solution = engine.solve(problem_text)
        print_solution(solution, engine)
        print(colorize("─" * 50, "dim"))


# ------------------------------------------------------------------ #
#                              main                                  #
# ------------------------------------------------------------------ #

def main() -> None:
    print(colorize(BANNER, "cyan"))

    # Initialise engine
    data_dir = os.path.join(_PROJECT_ROOT, "aria_data")
    engine = ARIAEngine(data_dir=data_dir)
    engine.clear_log()

    print(colorize("  Type a problem to solve, or 'help' for commands.\n", "dim"))

    while True:
        try:
            user_input = input(colorize("ARIA ❯ ", "cyan")).strip()
        except (EOFError, KeyboardInterrupt):
            print(colorize("\n  Goodbye! 👋\n", "cyan"))
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd in ("quit", "exit", "q"):
            print(colorize("\n  Goodbye! ARIA keeps learning. 👋\n", "cyan"))
            break
        elif cmd == "help":
            print(HELP_TEXT)
        elif cmd == "demo":
            run_demo(engine)
        elif cmd == "stats":
            show_stats(engine)
        elif cmd == "train":
            print(colorize("\n  🔄 Running self-training...", "yellow"))
            report = engine.train()
            print(f"  Training round {report['training_round']}")
            print(f"  Adjustments: {report['adjustments_applied']}")
            if report['knowledge_gaps']:
                print(f"  Knowledge gaps: {report['knowledge_gaps']}")
            print(colorize("  ✅ Training complete!\n", "green"))
        elif cmd == "history":
            show_history(engine)
        elif cmd == "knowledge":
            print_header("Knowledge Base")
            count = engine._knowledge.get_count()
            print(f"  Stored problem–solution pairs: {count}")
            if count > 0:
                print(f"  Similar search available for analogical reasoning")
            print()
        else:
            # Solve the problem
            engine.clear_log()
            solution = engine.solve(user_input)
            print_solution(solution, engine)


if __name__ == "__main__":
    main()
