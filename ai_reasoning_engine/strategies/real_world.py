"""
Real-world reasoning strategy for ARIA.

Applies real-world scientific laws, engineering formulas, financial models,
health/biometric benchmarks, and geographic routing to natural language problems.
"""

from __future__ import annotations

import math
import re
import time
from typing import Any

from ai_reasoning_engine.core.problem import Problem, ProblemType
from ai_reasoning_engine.core.solution import Solution
from ai_reasoning_engine.data.real_world_datasets import (
    PHYSICAL_CONSTANTS,
    FINANCIAL_DATA,
    REAL_WORLD_FLIGHT_DISTANCES,
    BMI_CATEGORIES,
)
from ai_reasoning_engine.strategies.base import BaseStrategy


class RealWorldStrategy(BaseStrategy):
    """Solves real-world applied science, finance, health, and engineering problems."""

    @property
    def name(self) -> str:
        return "RealWorld"

    @property
    def description(self) -> str:
        return "Applied physics, financial economics, health biometrics, and geospatial routing"

    @property
    def supported_types(self) -> list[ProblemType]:
        return [ProblemType.MATHEMATICAL, ProblemType.OPTIMIZATION, ProblemType.GENERAL]

    def solve(self, problem: Problem) -> Solution:
        start = time.perf_counter()
        steps: list[str] = []
        text = problem.description

        try:
            solvers = [
                self._solve_physics_kinetic_energy,
                self._solve_speed_of_light_travel,
                self._solve_planetary_weight,
                self._solve_electrical_power,
                self._solve_compound_interest,
                self._solve_mortgage_payment,
                self._solve_roi,
                self._solve_rule_of_72,
                self._solve_bmi,
                self._solve_temperature_conversion,
                self._solve_ideal_gas,
                self._solve_geographical_distance,
            ]

            for solver in solvers:
                res = solver(text, steps)
                if res is not None:
                    elapsed = time.perf_counter() - start
                    return Solution(
                        answer=res,
                        strategy_name=self.name,
                        confidence=0.95,
                        reasoning_steps=steps,
                        execution_time=elapsed,
                        metadata={"domain": "RealWorldAppliedKnowledge"},
                    )

            steps.append("No specific real-world domain formula matched query")
            elapsed = time.perf_counter() - start
            return Solution(
                answer=None,
                strategy_name=self.name,
                confidence=0.1,
                reasoning_steps=steps,
                execution_time=elapsed,
            )

        except Exception as exc:
            elapsed = time.perf_counter() - start
            steps.append(f"RealWorld strategy encountered error: {exc}")
            return Solution(
                answer=None,
                strategy_name=self.name,
                confidence=0.05,
                reasoning_steps=steps,
                execution_time=elapsed,
            )

    # ------------------------------------------------------------------ #
    #                       Physics Solvers                              #
    # ------------------------------------------------------------------ #

    def _solve_physics_kinetic_energy(self, text: str, steps: list[str]) -> Any:
        """KE = 0.5 * m * v^2"""
        if "kinetic energy" not in text.lower():
            return None

        # Look for mass (kg) and velocity (m/s)
        m_match = re.search(r'(\d+\.?\d*)\s*(?:kg|kilograms?)', text, re.IGNORECASE)
        v_match = re.search(r'(\d+\.?\d*)\s*(?:m/s|mps|meters per second)', text, re.IGNORECASE)

        if not (m_match and v_match):
            nums = [float(x) for x in re.findall(r'\d+\.?\d*', text)]
            if len(nums) >= 2:
                mass, velocity = nums[0], nums[1]
            else:
                return None
        else:
            mass = float(m_match.group(1))
            velocity = float(v_match.group(1))

        ke = 0.5 * mass * (velocity ** 2)
        steps.append(f"Identified kinetic energy formula: KE = 0.5 × m × v²")
        steps.append(f"Parameters: mass = {mass} kg, velocity = {velocity} m/s")
        steps.append(f"Calculation: 0.5 × {mass} × ({velocity})² = {ke:.2f} Joules")
        return round(ke, 2)

    def _solve_speed_of_light_travel(self, text: str, steps: list[str]) -> Any:
        """t = d / c"""
        text_lower = text.lower()
        if "light" not in text_lower or ("sun" not in text_lower and "travel" not in text_lower and "distance" not in text_lower):
            return None

        # Distance in km or meters
        nums = [float(x) for x in re.findall(r'\d+\.?\d*', text)]
        if len(nums) < 1:
            return None

        distance_km = nums[0]
        speed_km_s = 300000.0  # default speed of light in km/s
        if len(nums) >= 2:
            speed_km_s = nums[1]

        t_seconds = distance_km / speed_km_s
        steps.append(f"Applied electromagnetic radiation travel time: t = d / c")
        steps.append(f"Distance: {distance_km:,} km, Speed of light: {speed_km_s:,} km/s")
        steps.append(f"Travel time: {t_seconds:.2f} seconds ({t_seconds/60:.2f} minutes)")
        return round(t_seconds, 2)

    def _solve_planetary_weight(self, text: str, steps: list[str]) -> Any:
        """W = m * g"""
        text_lower = text.lower()
        if "weight" not in text_lower or ("gravity" not in text_lower and "moon" not in text_lower and "mars" not in text_lower):
            return None

        nums = [float(x) for x in re.findall(r'\d+\.?\d*', text)]
        if not nums:
            return None

        mass = nums[0]
        # Detect target planet gravity
        g = 9.80665
        planet = "Earth"
        if "moon" in text_lower:
            g = PHYSICAL_CONSTANTS["gravity_moon"]["value"]
            planet = "Moon"
        elif "mars" in text_lower:
            g = PHYSICAL_CONSTANTS["gravity_mars"]["value"]
            planet = "Mars"
        elif "jupiter" in text_lower:
            g = PHYSICAL_CONSTANTS["gravity_jupiter"]["value"]
            planet = "Jupiter"

        if len(nums) >= 2 and ("m/s" in text_lower or "gravity is" in text_lower):
            g = nums[1]

        weight = mass * g
        steps.append(f"Calculated gravitational force on {planet}: W = m × g")
        steps.append(f"Mass = {mass} kg, Surface Gravity = {g} m/s²")
        steps.append(f"Resulting weight = {weight:.2f} Newtons")
        return round(weight, 2)

    def _solve_electrical_power(self, text: str, steps: list[str]) -> Any:
        """P = V * I"""
        text_lower = text.lower()
        if "power" not in text_lower and "watts" not in text_lower:
            return None
        if "volt" not in text_lower or ("amp" not in text_lower and "current" not in text_lower):
            return None

        v_match = re.search(r'(\d+\.?\d*)\s*(?:v|volts?)', text, re.IGNORECASE)
        i_match = re.search(r'(\d+\.?\d*)\s*(?:a|amps?|amperes?)', text, re.IGNORECASE)

        if v_match and i_match:
            v, i = float(v_match.group(1)), float(i_match.group(1))
        else:
            nums = [float(x) for x in re.findall(r'\d+\.?\d*', text)]
            if len(nums) < 2:
                return None
            v, i = nums[0], nums[1]

        p = v * i
        steps.append(f"Applied electrical power equation: P = V × I")
        steps.append(f"Voltage: {v} V, Current: {i} A")
        steps.append(f"Calculated Power: {p:.2f} Watts")
        return round(p, 2)

    # ------------------------------------------------------------------ #
    #                      Financial Solvers                             #
    # ------------------------------------------------------------------ #

    def _solve_compound_interest(self, text: str, steps: list[str]) -> Any:
        """A = P * (1 + r)^t"""
        text_lower = text.lower()
        if "compound interest" not in text_lower and "compounded" not in text_lower:
            return None

        nums = [float(x) for x in re.findall(r'\d+\.?\d*', text)]
        if len(nums) < 3:
            return None

        # Typically: principal ($10,000), rate (7%), years (5)
        p = max(nums)
        remaining = [x for x in nums if x != p]
        r_raw = min(remaining)
        t_years = max(remaining)

        rate = r_raw / 100.0 if r_raw > 1.0 else r_raw
        amount = p * ((1.0 + rate) ** t_years)
        interest = amount - p

        steps.append(f"Applied compound interest formula: A = P(1 + r)^t")
        steps.append(f"Principal: ${p:,.2f}, Annual Rate: {rate*100:.2f}%, Duration: {t_years} years")
        steps.append(f"Total Future Value: ${amount:,.2f} (Earned Interest: ${interest:,.2f})")
        return round(amount, 2)

    def _solve_mortgage_payment(self, text: str, steps: list[str]) -> Any:
        """Monthly payment: M = P [r(1+r)^n] / [(1+r)^n - 1]"""
        text_lower = text.lower()
        if "mortgage" not in text_lower and "loan payment" not in text_lower:
            return None

        nums = [float(x) for x in re.findall(r'\d+\.?\d*', text)]
        if len(nums) < 3:
            return None

        p = max(nums)
        other = [x for x in nums if x != p]
        r_annual = min(other)
        years = max(other)

        r_monthly = (r_annual / 100.0) / 12.0
        n_months = int(years * 12)

        m = p * (r_monthly * ((1 + r_monthly) ** n_months)) / (((1 + r_monthly) ** n_months) - 1)
        steps.append(f"Applied standard amortization formula for monthly payment")
        steps.append(f"Loan Principal: ${p:,.2f}, Annual Interest: {r_annual}%, Term: {years} years ({n_months} payments)")
        steps.append(f"Monthly Mortgage Payment: ${m:,.2f}")
        return round(m, 2)

    def _solve_roi(self, text: str, steps: list[str]) -> Any:
        """ROI = ((Gain - Cost) / Cost) * 100"""
        text_lower = text.lower()
        if "roi" not in text_lower and "return on investment" not in text_lower:
            return None

        nums = [float(x) for x in re.findall(r'\d+\.?\d*', text)]
        if len(nums) < 2:
            return None

        cost, revenue = min(nums), max(nums)
        roi = ((revenue - cost) / cost) * 100.0
        steps.append(f"Applied Return on Investment (ROI) metric: ((Revenue - Cost) / Cost) × 100")
        steps.append(f"Cost basis: ${cost:,.2f}, Gross Return: ${revenue:,.2f}")
        steps.append(f"Calculated ROI: {roi:.2f}%")
        return round(roi, 2)

    def _solve_rule_of_72(self, text: str, steps: list[str]) -> Any:
        """Years to double = 72 / rate"""
        text_lower = text.lower()
        if "rule of 72" not in text_lower and ("double" not in text_lower or "interest" not in text_lower):
            return None

        nums = [float(x) for x in re.findall(r'\d+\.?\d*', text)]
        rates = [x for x in nums if x < 50 and x != 72]
        if not rates:
            return None

        rate = rates[0]
        years = 72.0 / rate
        steps.append(f"Applied financial Rule of 72: Doubling Time = 72 / rate")
        steps.append(f"Interest rate: {rate}%, Estimated doubling period: {years:.2f} years")
        return round(years, 2)

    # ------------------------------------------------------------------ #
    #                       Health & Units                               #
    # ------------------------------------------------------------------ #

    def _solve_bmi(self, text: str, steps: list[str]) -> Any:
        """BMI = weight / (height^2)"""
        text_lower = text.lower()
        if "bmi" not in text_lower and "body mass index" not in text_lower:
            return None

        nums = [float(x) for x in re.findall(r'\d+\.?\d*', text)]
        if len(nums) < 2:
            return None

        weight = max(nums)
        height = min(nums)
        if height > 3.0:  # If in cm, convert to meters
            height /= 100.0

        bmi = weight / (height ** 2)

        category = "Normal weight"
        for cat in BMI_CATEGORIES:
            if cat["min"] <= bmi <= cat["max"]:
                category = cat["category"]
                break

        steps.append(f"Applied WHO Body Mass Index metric: BMI = weight / height²")
        steps.append(f"Weight: {weight} kg, Height: {height} m")
        steps.append(f"Calculated BMI: {bmi:.2f} ({category})")
        return round(bmi, 2)

    def _solve_temperature_conversion(self, text: str, steps: list[str]) -> Any:
        """Fahrenheit <-> Celsius"""
        text_lower = text.lower()
        if "fahrenheit" not in text_lower and "celsius" not in text_lower:
            return None

        nums = [float(x) for x in re.findall(r'-?\d+\.?\d*', text)]
        if not nums:
            return None

        val = nums[0]
        if "fahrenheit to celsius" in text_lower or "f to c" in text_lower:
            c = (val - 32.0) * 5.0 / 9.0
            steps.append(f"Converted {val}°F to Celsius: ({val} - 32) × 5/9 = {c:.2f}°C")
            return round(c, 2)
        elif "celsius to fahrenheit" in text_lower or "c to f" in text_lower:
            f = (val * 9.0 / 5.0) + 32.0
            steps.append(f"Converted {val}°C to Fahrenheit: ({val} × 9/5) + 32 = {f:.2f}°F")
            return round(f, 2)
        return None

    def _solve_ideal_gas(self, text: str, steps: list[str]) -> Any:
        """P = (n * R * T) / V"""
        text_lower = text.lower()
        if "ideal gas" not in text_lower and "pascals" not in text_lower and "moles" not in text_lower:
            return None

        nums = [float(x) for x in re.findall(r'\d+\.?\d*', text)]
        if len(nums) < 3:
            return None

        # moles (e.g. 2), volume (e.g. 0.05), temp (e.g. 300)
        n = nums[0]
        v = nums[1]
        t = nums[2]
        r = 8.314  # J/mol·K

        p = (n * r * t) / v
        steps.append(f"Applied Ideal Gas Law: P = (n × R × T) / V")
        steps.append(f"n = {n} moles, V = {v} m³, T = {t} K, R = {r} J/(mol·K)")
        steps.append(f"Calculated Pressure: {p:,.2f} Pascals")
        return round(p, 2)

    def _solve_geographical_distance(self, text: str, steps: list[str]) -> Any:
        """Look up verified global flight distances."""
        text_upper = text.upper()
        for (c1, c2), dist in REAL_WORLD_FLIGHT_DISTANCES.items():
            if c1 in text_upper and c2 in text_upper:
                steps.append(f"Retrieved real-world geospatial flight corridor: {c1} ↔ {c2}")
                steps.append(f"Great-circle airway distance: {dist:,} km")
                return dist
        return None
