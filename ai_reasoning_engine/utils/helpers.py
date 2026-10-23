"""
Utility functions for ARIA.
"""

from __future__ import annotations

import math
import re


def safe_eval(expression: str) -> float:
    """Safely evaluate a math expression (no code injection)."""
    expr = re.sub(r'[^0-9+\-*/().%^ ]', '', expression)
    expr = expr.replace('^', '**')
    allowed = {
        "__builtins__": {},
        "abs": abs, "round": round, "min": min, "max": max,
        "sqrt": math.sqrt, "pow": pow, "log": math.log,
        "sin": math.sin, "cos": math.cos, "pi": math.pi, "e": math.e,
    }
    return float(eval(expr, allowed))  # noqa: S307


def tokenize(text: str) -> list[str]:
    """Lowercase, remove punctuation, split into words."""
    return re.findall(r'\b\w+\b', text.lower())


def extract_numbers(text: str) -> list[float]:
    """Extract all numbers from text."""
    return [float(n) for n in re.findall(r'-?\d+\.?\d*', text)]


def fuzzy_match(s1: str, s2: str) -> float:
    """Simple string similarity ratio (0-1) using longest common subsequence."""
    s1, s2 = s1.lower(), s2.lower()
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    # LCS length
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    return (2.0 * lcs) / (m + n)


def format_time(seconds: float) -> str:
    """Human-readable time string."""
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.0f}µs"
    if seconds < 1.0:
        return f"{seconds * 1000:.1f}ms"
    if seconds < 60:
        return f"{seconds:.2f}s"
    m, s = divmod(seconds, 60)
    return f"{int(m)}m {s:.1f}s"


def truncate(text: str, max_len: int = 100) -> str:
    """Truncate with ellipsis."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


# ------------------------------------------------------------------ #
#                     ANSI terminal colours                           #
# ------------------------------------------------------------------ #

_COLORS = {
    "red": "\033[91m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "magenta": "\033[95m",
    "cyan": "\033[96m",
    "white": "\033[97m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "reset": "\033[0m",
}


def colorize(text: str, color: str) -> str:
    """Wrap text in ANSI colour codes."""
    code = _COLORS.get(color.lower(), "")
    reset = _COLORS["reset"]
    return f"{code}{text}{reset}" if code else text
