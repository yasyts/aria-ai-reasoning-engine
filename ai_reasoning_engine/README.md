# 🧠 ARIA — Adaptive Reasoning & Intelligence Architecture

A self-improving AI reasoning engine in Python that tackles problems using multiple concurrent strategies, selects the best algorithms through meta-learning, ranks candidate solutions through multi-criteria evaluation, and continuously trains itself on collected experience to find the **1 optimal solution**.

---

## 🌟 Key Architecture & Capabilities

```
                           ┌──────────────────────────┐
                           │    User Input Problem    │
                           └────────────┬─────────────┘
                                        │
                                        ▼
                           ┌──────────────────────────┐
                           │    Problem Classifier    │
                           └────────────┬─────────────┘
                                        │
                         ┌──────────────┴──────────────┐
                         ▼                             ▼
              ┌─────────────────────┐       ┌──────────────────────┐
              │ UCB1 Meta-Learner   │◄──────┤ Self-Training Loop   │
              │ (Algorithm Selector)│       │ (Weight Adjustments) │
              └──────────┬──────────┘       └──────────▲───────────┘
                         │                             │
        ┌────────────────┼────────────────┬────────────┴───────┐
        ▼                ▼                ▼                    ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐   ┌───────────────────┐
│ Mathematical │ │   Logical    │ │  Heuristic   │   │ Pattern / Analogy │
│  (Algebra,   │ │ (Syllogisms, │ │ (Dijkstra,   │   │ (Sequences, Past  │
│  Calculus)   │ │  Inference)  │ │   Knapsack)  │   │  Experience)      │
└───────┬──────┘ └───────┬──────┘ └───────┬──────┘   └─────────┬─────────┘
        │                │                │                    │
        └────────────────┼────────────────┴────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────────────────────┐
        │        Multi-Criteria Solution Evaluator               │
        │  • Confidence  • Consensus  • Efficiency  • Steps     │
        └────────────────────────┬───────────────────────────────┘
                                 │
                                 ▼
                     ★ 1 OPTIMAL SOLUTION ★
                                 │
                   ┌─────────────┴─────────────┐
                   ▼                           ▼
        ┌────────────────────┐      ┌────────────────────┐
        │ Performance Record │      │   Knowledge Base   │
        └────────────────────┘      └────────────────────┘
```

### 1. Multi-Strategy Reasoning
ARIA does not rely on a single algorithm. Every problem is attacked from multiple perspectives:
- **Mathematical Strategy** (`ai_reasoning_engine.strategies.mathematical`):
  - Linear and quadratic algebra ($ax^2 + bx + c = 0$)
  - Arithmetic and symbolic expression evaluation
  - Factorials, combinations ($C(n, r)$), permutations ($P(n, r)$)
  - Greatest Common Divisor (GCD) and Least Common Multiple (LCM)
  - Statistical analysis (mean, median, mode, standard deviation)
- **Logical Strategy** (`ai_reasoning_engine.strategies.logical`):
  - Deductive syllogisms ("All A are B, X is A $\implies$ X is B")
  - Boolean logic evaluation and simplification
  - Modus ponens and rule-based forward-chaining
- **Heuristic Strategy** (`ai_reasoning_engine.strategies.heuristic`):
  - Dijkstra shortest path algorithm on graph networks
  - 0/1 Knapsack optimization via Dynamic Programming
  - Continuous function optimization via Simulated Annealing
- **Pattern Strategy** (`ai_reasoning_engine.strategies.pattern`):
  - Arithmetic progression ($d$)
  - Geometric progression ($r$)
  - Polynomial sequence extrapolation using the Method of Finite Differences
  - Fibonacci and Tribonacci sequences
  - String palindromes and character frequency analysis
- **Real-World Strategy** (`ai_reasoning_engine.strategies.real_world`):
  - Applied physics: kinetic energy ($0.5mv^2$), speed of light travel times ($t = d/c$), electrical power ($P = VI$), planetary surface weights
  - Financial economics: compound interest ($A = P(1+r)^t$), 30-year amortized mortgage payments, ROI %, Rule of 72
  - Health & biometrics: Body Mass Index (BMI) and WHO weight classification
  - Global logistics: airway routing across international flight corridors
  - Ideal Gas Law and temperature unit conversions
- **Analogical Strategy** (`ai_reasoning_engine.strategies.analogical`):
  - Jaccard and cosine token similarity against historical problems
  - Adapts solutions from previously solved similar cases

### 2. Meta-Learning Algorithm Selector (UCB1)
ARIA uses the **Upper Confidence Bound (UCB1)** multi-armed bandit algorithm to balance **exploration** (trying alternative strategies to discover better methods) with **exploitation** (favoring strategies that consistently yield high scores for a given problem type):

$$\text{UCB1} = \bar{R}_i + C \sqrt{\frac{\ln N}{n_i}}$$

### 3. Multi-Criteria Solution Evaluator
All candidate solutions are scored against 4 criteria to select the single best solution:
1. **Confidence** ($35\%$): Inherent certainty of the strategy's proof or computation.
2. **Consensus / Consistency** ($25\%$): Agreement across multiple independent strategies.
3. **Efficiency** ($20\%$): Computational time and resource conservation.
4. **Completeness** ($20\%$): Depth and validity of the step-by-step reasoning chain.

### 4. Self-Training & Autonomous Improvement
- Automatically saves every interaction to persistent storage in `aria_data/`.
- Every 15 problems (or via manual `train` command), ARIA analyzes performance trends.
- Rewards strategies that outperform expectations, penalizes underperformers, and identifies domain knowledge gaps.

---

## 💻 How to Run

### Requirements
- Python 3.10+ (Standard library only; zero mandatory third-party dependencies).

### Run the Interactive CLI
```bash
python main.py
```

### Interactive CLI Commands
- `<problem>`: Type any question or problem in natural language.
- `demo`: Automatically runs 8 diverse problems across mathematics, logic, optimization, patterns, and sequences.
- `stats`: Displays overall accuracy, strategy rankings, average latency, and learning trends.
- `train`: Triggers the self-training loop to re-weight strategies based on accumulated data.
- `history`: Lists recent problems solved and their scores.
- `knowledge`: Shows the state of the long-term knowledge memory.
- `quit` or `exit`: Exits ARIA.

---

## 🧪 Running the Automated Test Suite

```bash
python test_suite.py
```

---

## 📁 Project Structure

```
ai_reasoning_engine/
├── main.py                     # Interactive CLI entry point with ASCII art & color
├── test_suite.py               # Comprehensive 8-point automated test suite
├── requirements.txt            # Minimal requirements file
├── core/
│   ├── engine.py               # ARIAEngine orchestrator
│   ├── problem.py              # Problem dataclass & weighted classifier
│   └── solution.py             # Solution dataclass & reasoning chains
├── strategies/
│   ├── base.py                 # Abstract BaseStrategy interface
│   ├── mathematical.py         # Equations, stats, combinatorics, calculus
│   ├── logical.py              # Syllogisms, boolean logic, forward chaining
│   ├── heuristic.py            # Dijkstra, knapsack, simulated annealing
│   ├── pattern.py              # Polynomial/geometric sequences, palindromes
│   └── analogical.py           # Similarity search & analogical transfer
├── meta/
│   ├── classifier.py           # Problem categorization
│   ├── algorithm_selector.py   # UCB1 multi-armed bandit meta-learner
│   ├── performance_tracker.py  # History tracking & rolling trend metrics
│   └── self_trainer.py         # Autonomous weight optimizer & gap detector
├── evaluation/
│   └── evaluator.py            # Multi-criteria solution ranker
├── knowledge/
│   └── knowledge_base.py       # Persistent JSON store for transfer learning
└── utils/
    └── helpers.py              # Safe eval, tokenization, terminal colors
```
