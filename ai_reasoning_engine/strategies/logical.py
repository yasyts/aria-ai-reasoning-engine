"""
Logical reasoning strategy.

Implements deductive/inductive inference with a rule-based engine,
syllogistic reasoning, boolean logic, and forward chaining.
"""

from __future__ import annotations

import re
import time
from typing import Any

from ai_reasoning_engine.core.problem import Problem, ProblemType
from ai_reasoning_engine.core.solution import Solution
from ai_reasoning_engine.strategies.base import BaseStrategy


class LogicalStrategy(BaseStrategy):
    """Deductive and inductive logical reasoning."""

    @property
    def name(self) -> str:
        return "Logical"

    @property
    def description(self) -> str:
        return "Syllogisms, boolean logic, forward chaining inference"

    @property
    def supported_types(self) -> list[ProblemType]:
        return [ProblemType.LOGICAL, ProblemType.GENERAL]

    def solve(self, problem: Problem) -> Solution:
        start = time.perf_counter()
        steps: list[str] = []
        text = problem.description

        try:
            # Try specialized solvers
            for solver in [
                self._try_syllogism,
                self._try_boolean_eval,
                self._try_modus_ponens,
                self._try_category_inference,
            ]:
                result = solver(text, steps)
                if result is not None:
                    elapsed = time.perf_counter() - start
                    return Solution(
                        answer=result,
                        strategy_name=self.name,
                        confidence=0.85,
                        reasoning_steps=steps,
                        execution_time=elapsed,
                    )

            # Forward-chaining fallback
            answer = self._forward_chain(text, steps)
            elapsed = time.perf_counter() - start
            return Solution(
                answer=answer,
                strategy_name=self.name,
                confidence=0.4 if answer else 0.1,
                reasoning_steps=steps,
                execution_time=elapsed,
            )

        except Exception as exc:
            elapsed = time.perf_counter() - start
            steps.append(f"Error: {exc}")
            return Solution(
                answer=None, strategy_name=self.name,
                confidence=0.05, reasoning_steps=steps,
                execution_time=elapsed,
            )

    # ------------------------------------------------------------------ #
    #                            solvers                                  #
    # ------------------------------------------------------------------ #

    def _try_syllogism(self, text: str, steps: list[str]) -> Any:
        """
        All A are B.  X is A.  → X is B.
        """
        text_lower = text.lower()
        # Pattern: "all <A> are <B>"
        universal = re.findall(r'all\s+(\w+)\s+are\s+(\w+)', text_lower)
        # Pattern: "<X> is a/an <A>"  or  "<X> is <A>"
        instances = re.findall(r'(\w+)\s+is\s+(?:a|an\s+)?(\w+)', text_lower)

        if not universal or not instances:
            return None

        steps.append(f"Universal premises: {universal}")
        steps.append(f"Instance premises : {instances}")

        # Build category graph: A → B
        isa: dict[str, set[str]] = {}
        for sub, sup in universal:
            isa.setdefault(sub, set()).add(sup)

        # Transitive closure
        changed = True
        while changed:
            changed = False
            for sub in list(isa.keys()):
                new_sups: set[str] = set()
                for sup in isa[sub]:
                    if sup in isa:
                        new_sups |= isa[sup]
                before = len(isa[sub])
                isa[sub] |= new_sups
                if len(isa[sub]) > before:
                    changed = True

        # Infer
        conclusions: list[str] = []
        for entity, category in instances:
            if category in isa:
                for parent in isa[category]:
                    conclusion = f"{entity} is {parent}"
                    conclusions.append(conclusion)
                    steps.append(f"Inference: {entity} is {category}, all {category} are {parent} → {conclusion}")

        if conclusions:
            answer = "; ".join(conclusions) if len(conclusions) > 1 else conclusions[0]
            return answer
        return None

    def _try_boolean_eval(self, text: str, steps: list[str]) -> Any:
        """Evaluate boolean expressions: True AND False, NOT True, etc."""
        text_lower = text.lower()
        bool_keywords = ['true', 'false', ' and ', ' or ', ' not ']
        if sum(1 for kw in bool_keywords if kw in text_lower) < 2:
            return None

        # Extract the boolean expression
        expr = text_lower
        expr = re.sub(r'evaluate|compute|what is|find|the value of', '', expr).strip().rstrip('?.')

        expr = expr.replace(' and ', ' and ').replace(' or ', ' or ').replace(' not ', ' not ')
        expr = expr.replace('true', 'True').replace('false', 'False')

        steps.append(f"Boolean expression: {expr}")
        try:
            allowed = {"__builtins__": {}, "True": True, "False": False}
            result = eval(expr, allowed)  # noqa: S307
            steps.append(f"Result: {result}")
            return result
        except Exception:
            return None

    def _try_modus_ponens(self, text: str, steps: list[str]) -> Any:
        """
        If P then Q.  P is true.  → Q is true.
        """
        text_lower = text.lower()
        # Pattern: "if <P> then <Q>"
        rules = re.findall(r'if\s+(.+?)\s+then\s+(.+?)(?:\.|,|;|$)', text_lower)
        if not rules:
            return None

        steps.append(f"Rules found: {rules}")

        # Check which antecedents are asserted
        conclusions: list[str] = []
        for antecedent, consequent in rules:
            antecedent = antecedent.strip()
            consequent = consequent.strip()
            # Check if antecedent is stated as true
            if (
                f"{antecedent} is true" in text_lower
                or f"{antecedent}" in text_lower.replace(f"if {antecedent} then", "")
            ):
                steps.append(f"Modus ponens: '{antecedent}' is true → '{consequent}' is true")
                conclusions.append(consequent)

        if conclusions:
            return "; ".join(conclusions)
        return None

    def _try_category_inference(self, text: str, steps: list[str]) -> Any:
        """
        Handles: "If all dogs are animals, and Rex is a dog, what is Rex?"
        """
        text_lower = text.lower()

        # Pattern: "all <A> are <B>" or "every <A> is a <B>"
        universal = re.findall(r'(?:all|every)\s+(\w+)\s+(?:are|is\s+(?:a|an)?)\s+(\w+)', text_lower)
        # Pattern: "<X> is a <A>"
        instances = re.findall(r'(\w+)\s+is\s+(?:a|an)\s+(\w+)', text_lower)

        if not universal:
            return None

        steps.append(f"Universal: {universal}")
        steps.append(f"Instances: {instances}")

        # Build inheritance
        category_map: dict[str, str] = {}
        for sub, sup in universal:
            # Handle plural → singular
            sub_s = sub.rstrip('s') if sub.endswith('s') else sub
            category_map[sub_s] = sup.rstrip('s') if sup.endswith('s') else sup
            category_map[sub] = sup

        conclusions: list[str] = []
        for entity, category in instances:
            cat_key = category.rstrip('s') if category.endswith('s') else category
            if cat_key in category_map:
                parent = category_map[cat_key]
                conclusions.append(f"{entity} is an {parent}")
                steps.append(f"{entity} is a {category} → {entity} is an {parent}")
            elif category in category_map:
                parent = category_map[category]
                conclusions.append(f"{entity} is an {parent}")
                steps.append(f"{entity} is a {category} → {entity} is an {parent}")

        # Also check "what is <X>" pattern
        what_match = re.search(r'what\s+is\s+(\w+)', text_lower)
        if what_match and conclusions:
            entity = what_match.group(1)
            # Find conclusion about this entity
            for c in conclusions:
                if entity in c.lower():
                    return c

        if conclusions:
            return "; ".join(conclusions)

        return None

    def _forward_chain(self, text: str, steps: list[str]) -> Any:
        """Extract facts and rules, then forward-chain to derive new facts."""
        text_lower = text.lower()
        sentences = re.split(r'[.;,]', text_lower)
        sentences = [s.strip() for s in sentences if s.strip()]

        facts: set[str] = set()
        rules: list[tuple[str, str]] = []

        for sent in sentences:
            rule_match = re.match(r'if\s+(.+?)\s+then\s+(.+)', sent)
            if rule_match:
                rules.append((rule_match.group(1).strip(), rule_match.group(2).strip()))
            elif not sent.startswith(('what', 'find', 'determine', 'is')):
                facts.add(sent)

        steps.append(f"Initial facts: {facts}")
        steps.append(f"Rules: {rules}")

        # Forward chain
        iterations = 0
        new_fact_added = True
        while new_fact_added and iterations < 20:
            new_fact_added = False
            iterations += 1
            for antecedent, consequent in rules:
                if antecedent in facts and consequent not in facts:
                    facts.add(consequent)
                    new_fact_added = True
                    steps.append(f"Derived: {consequent}")

        if facts:
            return "; ".join(sorted(facts))
        return None
