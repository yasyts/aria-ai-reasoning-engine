# 🧠 ARIA — Adaptive Reasoning & Intelligence Architecture

[![Live Demo](https://img.shields.io/badge/Live_Demo-GitHub_Pages-06b6d4?style=for-the-badge&logo=github)](https://yasyts.github.io/aria-ai-reasoning-engine/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![Zero Dependencies](https://img.shields.io/badge/Dependencies-Zero-10b981?style=for-the-badge)](#-zero-external-dependencies)

> **Live Interactive Website:** **[https://yasyts.github.io/aria-ai-reasoning-engine/](https://yasyts.github.io/aria-ai-reasoning-engine/)**

---

## 🌟 What is ARIA?

**ARIA** (**A**daptive **R**easoning & **I**ntelligence **A**rchitecture) is an AI reasoning engine built from first principles that:

1. **Attacks every problem with multiple concurrent strategies** (algebraic, logical deduction, graph search, pattern recognition, analogical memory, and applied real-world science).
2. **Selects optimal strategies via Meta-Learning** using the **UCB1 (Upper Confidence Bound)** Multi-Armed Bandit algorithm.
3. **Picks the single #1 perfect solution** via multi-criteria evaluation (confidence, inter-strategy consensus, computational speed, and proof completeness).
4. **Continuously trains itself on real-world data** (applied physics, finance, health, international logistics, and economic time series) and adapts its decision weights over time.

---

## 🚀 Live Web Dashboard

Try the interactive website directly in your browser:
👉 **[Open ARIA Live Dashboard](https://yasyts.github.io/aria-ai-reasoning-engine/)**

- **⚡ Reasoning Studio**: Submit any natural language problem or choose from real-world presets (Kinetic Energy, 30-year Mortgage, Compound Interest, Moon Gravity, BMI, Dijkstra Shortest Path, Linear & Quadratic equations).
- **📊 Candidate Benchmark Matrix**: Inspect side-by-side solutions from all competing strategies, including confidence %, compute latency, and evaluator scores.
- **🧠 Meta-Learner Visualizer**: Live UCB1 weights and strategy win rates.
- **🔄 Autonomous Self-Training**: Real-time training loop metrics and continuous score evolution charts.
- **🌍 Real-World Data Catalog**: Grounded physical constants, historical inflation CPI datasets, and airway distances.

---

## 🧩 Core Reasoning Strategies

| Strategy | Domain & Algorithms | Key Formulas / Proofs |
|---|---|---|
| **RealWorld** | Applied physics, financial economics, health, logistics | $KE = \frac{1}{2}mv^2$, Mortgage $P\frac{r(1+r)^n}{(1+r)^n-1}$, BMI, $PV=nRT$ |
| **Mathematical** | Symbolic & numerical algebra, statistics, quadratics | $ax^2 + bx + c = 0$, $ax + b = c$, Mean, $\sigma$, $n!$, $C(n,r)$, $\gcd/\text{lcm}$ |
| **Logical** | Categorical syllogisms, boolean logic, forward chaining | $\forall x (A(x) \to B(x))$, Modus Ponens, Contradiction Detection |
| **Heuristic** | Graph routing & combinatorial optimization | Dijkstra Shortest Path, 0/1 Knapsack DP, Simulated Annealing |
| **Pattern** | Sequence recognition & series extrapolation | Arithmetic ($d$), Geometric ($r$), Polynomial (Finite Differences), Fibonacci |
| **Analogical** | Pre-trained transfer learning from memory | Jaccard & Cosine text similarity across 39+ verified cases |

---

## 💻 Local Installation & Usage

### 1. Run the Interactive Terminal CLI
```bash
python ai_reasoning_engine/main.py
```

### 2. Run the Local Web Server & REST API
```bash
python ai_reasoning_engine/server.py --port 8000
```
Open **`http://localhost:8000`** in any web browser.

### 3. Run the Autonomous Real-World Training Pipeline
```bash
python ai_reasoning_engine/train_on_real_data.py
```

### 4. Run the Test Suite
```bash
python ai_reasoning_engine/test_suite.py
```

---

## 🛠️ Zero External Dependencies

ARIA is implemented in **pure, clean Python 3.10+ standard library**:
- `http.server.ThreadingHTTPServer` for high-concurrency web serving
- `math`, `heapq`, `re`, `json`, `time` for symbolic and numerical computation
- Zero mandatory external packages (`pip install` not required!)

---

## 📄 License

MIT License. Free to use, adapt, and build upon.
