"""
Real-world datasets, scientific constants, financial formulas,
geographic logistics networks, and benchmark problem sets for ARIA.
"""

from __future__ import annotations

import math
from typing import Any

# ===================================================================== #
#                   1. Physical & Scientific Constants                  #
# ===================================================================== #

PHYSICAL_CONSTANTS: dict[str, dict[str, Any]] = {
    "speed_of_light": {
        "symbol": "c",
        "value": 299792458.0,
        "unit": "m/s",
        "name": "Speed of Light in vacuum",
    },
    "gravitational_constant": {
        "symbol": "G",
        "value": 6.67430e-11,
        "unit": "N·m²/kg²",
        "name": "Newtonian constant of gravitation",
    },
    "planck_constant": {
        "symbol": "h",
        "value": 6.62607015e-34,
        "unit": "J·s",
        "name": "Planck constant",
    },
    "avogadro_constant": {
        "symbol": "N_A",
        "value": 6.02214076e23,
        "unit": "mol⁻¹",
        "name": "Avogadro constant",
    },
    "boltzmann_constant": {
        "symbol": "k_B",
        "value": 1.380649e-23,
        "unit": "J/K",
        "name": "Boltzmann constant",
    },
    "elementary_charge": {
        "symbol": "e",
        "value": 1.602176634e-19,
        "unit": "C",
        "name": "Elementary electric charge",
    },
    "standard_gravity_earth": {
        "symbol": "g_earth",
        "value": 9.80665,
        "unit": "m/s²",
        "name": "Standard Earth gravitational acceleration",
    },
    "gravity_moon": {
        "symbol": "g_moon",
        "value": 1.62,
        "unit": "m/s²",
        "name": "Moon surface gravity",
    },
    "gravity_mars": {
        "symbol": "g_mars",
        "value": 3.72,
        "unit": "m/s²",
        "name": "Mars surface gravity",
    },
    "gravity_jupiter": {
        "symbol": "g_jupiter",
        "value": 24.79,
        "unit": "m/s²",
        "name": "Jupiter surface gravity",
    },
    "mass_earth": {
        "symbol": "M_earth",
        "value": 5.9722e24,
        "unit": "kg",
        "name": "Mass of Planet Earth",
    },
    "mass_moon": {
        "symbol": "M_moon",
        "value": 7.342e22,
        "unit": "kg",
        "name": "Mass of Moon",
    },
    "mass_sun": {
        "symbol": "M_sun",
        "value": 1.9885e30,
        "unit": "kg",
        "name": "Mass of the Sun",
    },
    "distance_earth_sun": {
        "symbol": "1_AU",
        "value": 149597870700.0,
        "unit": "m",
        "name": "Astronomical Unit (Distance Earth to Sun)",
    },
    "distance_earth_moon": {
        "symbol": "d_moon",
        "value": 384400000.0,
        "unit": "m",
        "name": "Average distance Earth to Moon",
    },
}

# ===================================================================== #
#                   2. Real-World Financial Models                     #
# ===================================================================== #

FINANCIAL_DATA = {
    # Historical US Consumer Price Index (Annual Average Inflation %)
    "us_inflation_rates": [
        {"year": 2018, "rate": 2.44},
        {"year": 2019, "rate": 1.81},
        {"year": 2020, "rate": 1.23},
        {"year": 2021, "rate": 4.70},
        {"year": 2022, "rate": 8.00},
        {"year": 2023, "rate": 4.12},
        {"year": 2024, "rate": 3.10},
    ],
    # Reference FX conversion rates (USD base)
    "currency_rates": {
        "USD": 1.00,
        "EUR": 0.92,
        "GBP": 0.79,
        "JPY": 152.4,
        "INR": 83.9,
        "CAD": 1.36,
        "AUD": 1.51,
        "CHF": 0.88,
    },
}

# ===================================================================== #
#                   3. Global Geographic Network                        #
# ===================================================================== #

# Real-world flight & shipping distances (in kilometers)
REAL_WORLD_FLIGHT_DISTANCES = {
    ("LONDON", "DUBAI"): 5500,
    ("DUBAI", "TOKYO"): 7930,
    ("LONDON", "NEW YORK"): 5570,
    ("NEW YORK", "SAN FRANCISCO"): 4140,
    ("SAN FRANCISCO", "TOKYO"): 8280,
    ("LONDON", "PARIS"): 344,
    ("PARIS", "BERLIN"): 878,
    ("BERLIN", "DUBAI"): 4620,
    ("DUBAI", "SINGAPORE"): 5850,
    ("SINGAPORE", "TOKYO"): 5320,
    ("SINGAPORE", "SYDNEY"): 6300,
    ("DUBAI", "MUMBAI"): 1930,
    ("MUMBAI", "SINGAPORE"): 3910,
    ("NEW YORK", "PARIS"): 5830,
}

CITY_COORDINATES = {
    "NEW YORK": {"lat": 40.7128, "lon": -74.0060},
    "LONDON": {"lat": 51.5074, "lon": -0.1278},
    "PARIS": {"lat": 48.8566, "lon": 2.3522},
    "BERLIN": {"lat": 52.5200, "lon": 13.4050},
    "DUBAI": {"lat": 25.2048, "lon": 55.2708},
    "MUMBAI": {"lat": 19.0760, "lon": 72.8777},
    "SINGAPORE": {"lat": 1.3521, "lon": 103.8198},
    "TOKYO": {"lat": 35.6762, "lon": 139.6503},
    "SYDNEY": {"lat": -33.8688, "lon": 151.2093},
    "SAN FRANCISCO": {"lat": 37.7749, "lon": -122.4194},
}

# ===================================================================== #
#                   4. Health & Medical Benchmarks                     #
# ===================================================================== #

BMI_CATEGORIES = [
    {"category": "Underweight", "min": 0.0, "max": 18.5},
    {"category": "Normal weight", "min": 18.5, "max": 24.9},
    {"category": "Overweight", "min": 25.0, "max": 29.9},
    {"category": "Obesity Class I", "min": 30.0, "max": 34.9},
    {"category": "Obesity Class II", "min": 35.0, "max": 39.9},
    {"category": "Obesity Class III", "min": 40.0, "max": 100.0},
]

# ===================================================================== #
#             5. Curated Real-World Problem Training Dataset            #
# ===================================================================== #

REAL_WORLD_BENCHMARKS = [
    # --- Physics & Space ---
    {
        "id": "PHYS-001",
        "domain": "Physics",
        "problem": "What is the kinetic energy of a 1200 kg car travelling at 25 m/s?",
        "ground_truth": 375000.0,  # 0.5 * 1200 * 25^2 = 375,000 J
        "type": "MATHEMATICAL",
        "formula": "KE = 0.5 * m * v^2",
    },
    {
        "id": "PHYS-002",
        "domain": "Astrophysics",
        "problem": "How long in seconds does light take to travel from the Sun to Earth given distance 149600000 km and light speed 300000 km/s?",
        "ground_truth": 498.67,  # 149600000 / 300000 = ~498.67 s (~8.31 min)
        "type": "MATHEMATICAL",
        "formula": "t = d / c",
    },
    {
        "id": "PHYS-003",
        "domain": "Physics",
        "problem": "Calculate the weight in Newtons of an 80 kg astronaut on the Moon where gravity is 1.62 m/s^2",
        "ground_truth": 129.6,  # 80 * 1.62 = 129.6 N
        "type": "MATHEMATICAL",
        "formula": "W = m * g",
    },
    {
        "id": "PHYS-004",
        "domain": "Electrical Engineering",
        "problem": "Calculate the power in Watts consumed by a circuit with 120 Volts and a current of 2.5 Amperes",
        "ground_truth": 300.0,  # P = V * I = 120 * 2.5 = 300 W
        "type": "MATHEMATICAL",
        "formula": "P = V * I",
    },

    # --- Finance & Economics ---
    {
        "id": "FIN-001",
        "domain": "Personal Finance",
        "problem": "Calculate compound interest total amount on $10000 invested at 7% annual interest compounded annually for 5 years",
        "ground_truth": 14025.52,  # 10000 * (1.07)^5 = 14025.5173
        "type": "MATHEMATICAL",
        "formula": "A = P * (1 + r)^t",
    },
    {
        "id": "FIN-002",
        "domain": "Real Estate / Mortgage",
        "problem": "Calculate the monthly mortgage payment on a $300000 loan with 6% annual interest over a 30 year term",
        "ground_truth": 1798.65,  # M = P [ r(1+r)^n ] / [ (1+r)^n - 1 ]
        "type": "MATHEMATICAL",
        "formula": "M = P * [r(1+r)^n] / [(1+r)^n - 1]",
    },
    {
        "id": "FIN-003",
        "domain": "Corporate Finance",
        "problem": "Calculate the Return on Investment (ROI) percentage if an investment cost $50000 and generated $65000 total revenue",
        "ground_truth": 30.0,  # ((65000 - 50000) / 50000) * 100 = 30%
        "type": "MATHEMATICAL",
        "formula": "ROI = ((Gain - Cost) / Cost) * 100",
    },
    {
        "id": "FIN-004",
        "domain": "Economics",
        "problem": "Using the Rule of 72, how many years will it take for an investment to double at 8% annual return?",
        "ground_truth": 9.0,  # 72 / 8 = 9 years
        "type": "MATHEMATICAL",
        "formula": "Years = 72 / rate",
    },

    # --- Logistics & Route Optimization ---
    {
        "id": "LOG-001",
        "domain": "Global Logistics",
        "problem": "Find the shortest route distance from LONDON to TOKYO given: LONDON-DUBAI=5500, DUBAI-TOKYO=7930, LONDON-PARIS=344, PARIS-SINGAPORE=10730, SINGAPORE-TOKYO=5320",
        "ground_truth": 13430.0,  # London -> Dubai (5500) + Dubai -> Tokyo (7930) = 13,430 km
        "type": "OPTIMIZATION",
        "formula": "Dijkstra Graph Optimization",
    },
    {
        "id": "LOG-002",
        "domain": "Supply Chain Knapsack",
        "problem": "Cargo knapsack capacity 100 kg. Available pallets (value, weight): (60, 20), (100, 50), (120, 30). What is max value?",
        "ground_truth": 280.0,  # All 3 fit: 20+50+30 = 100kg, value = 60+100+120 = 280
        "type": "OPTIMIZATION",
        "formula": "0/1 Dynamic Programming Knapsack",
    },

    # --- Health & Medicine ---
    {
        "id": "MED-001",
        "domain": "Medicine / Nutrition",
        "problem": "Calculate the Body Mass Index (BMI) for a person weighing 75 kg with height 1.75 meters",
        "ground_truth": 24.49,  # 75 / (1.75^2) = 24.4897
        "type": "MATHEMATICAL",
        "formula": "BMI = weight / (height^2)",
    },
    {
        "id": "MED-002",
        "domain": "Pharmacology",
        "problem": "Calculate pediatric dosage if adult dose is 500 mg and child weight is 20 kg using Clark's rule (child_weight / 70 * adult_dose)",
        "ground_truth": 142.86,  # 20 / 70 * 500 = 142.857
        "type": "MATHEMATICAL",
        "formula": "Child Dose = (Weight_kg / 70) * Adult_Dose",
    },

    # --- Chemistry & Environmental Science ---
    {
        "id": "CHEM-001",
        "domain": "Chemistry",
        "problem": "Calculate the pressure in Pascals of 2 moles of ideal gas in a 0.05 m^3 container at temperature 300 Kelvin (R = 8.314 J/mol·K)",
        "ground_truth": 99768.0,  # P = nRT / V = 2 * 8.314 * 300 / 0.05 = 99,768 Pa
        "type": "MATHEMATICAL",
        "formula": "P = (n * R * T) / V",
    },
    {
        "id": "CHEM-002",
        "domain": "Unit Conversion",
        "problem": "Convert 98.6 degrees Fahrenheit to Celsius using formula (F - 32) * 5 / 9",
        "ground_truth": 37.0,  # (98.6 - 32) * 5 / 9 = 37.0 C
        "type": "MATHEMATICAL",
        "formula": "C = (F - 32) * 5 / 9",
    },

    # --- Logic & Real-World Diagnostic Syllogisms ---
    {
        "id": "LOGIC-001",
        "domain": "Medical Diagnosis",
        "problem": "If all patients with condition Alpha have elevated biomarker Beta, and patient John has elevated biomarker Beta, does this prove John has Alpha? If all humans are mammals and all mammals are vertebrates, what are humans?",
        "ground_truth": "humans are vertebrates",
        "type": "LOGICAL",
        "formula": "Transitive categorical syllogism",
    },
    {
        "id": "LOGIC-002",
        "domain": "Corporate Governance",
        "problem": "If all executive board members are shareholders, and Maria is an executive board member, what is Maria?",
        "ground_truth": "maria is an shareholder",
        "type": "LOGICAL",
        "formula": "Deductive universal instantiation",
    },

    # --- Real World Time-Series & Sequence Prediction ---
    {
        "id": "SEQ-001",
        "domain": "Macroeconomics",
        "problem": "Predict the next annual US inflation rate given sequence: 1.8, 2.4, 3.0, 3.6, 4.2",
        "ground_truth": 4.8,  # Arithmetic progression with diff = +0.6
        "type": "SEQUENCE",
        "formula": "Arithmetic sequence prediction",
    },
    {
        "id": "SEQ-002",
        "domain": "Technology Hardware",
        "problem": "Predict the next transistor count doubling step in thousands: 100, 200, 400, 800",
        "ground_truth": 1600.0,  # Geometric progression r = 2
        "type": "SEQUENCE",
        "formula": "Geometric sequence prediction (Moore's law)",
    },
]
