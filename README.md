**#Information Dynamics: A Unified Predictive Framework for Interstellar Objects**

https://img.shields.io/badge/License-MIT-yellow.svg
https://img.shields.io/badge/python-3.10+-blue.svg

**This code repository is the companion to the paper:**

**Information Dynamics: A Unified Predictive Framework for Interstellar Objects — With Immediate Testable Predictions for 3I/ATLAS**
Kai Huang*, Hongkui Liu†
Submitted to arXiv on Jan 19, 2026

Abstract: This work introduces the Information Dynamics framework, which unifies the anomalous behaviors of interstellar objects (ISOs) — 1I/‘Oumuamua, 2I/Borisov, and 3I/ATLAS — within a single, continuous spectrum defined by an information purity parameter, p. By adapting the Complex Ginzburg-Landau equation, the model explains key observational puzzles: strong non-gravitational acceleration without a coma (1I), standard cometary activity (2I), and, most critically, the coexistence of intense macroscopic jets with a lack of sub-micron dust scattering in 3I/ATLAS. We propose this “dust deficit” is not a compositional anomaly but a signature of a mixed dynamical state (p ≈ 0.17) where energy channels into macroscopic order. We conclude with three immediate, testable predictions for 3I/ATLAS before mid-2026. All code for reproducing the results and figures is provided here.

**Core Questions & Code-Based Explanations**
This repository provides computational answers to the main puzzles outlined in the paper. Each core question is addressed by a specific script or module:

Core Scientific Question	What This Code Does	Key File(s)
Q1: The ISO Spectrum – Can three seemingly different ISOs be described by one model?	Fits the p-spectrum, placing 1I, 2I, and 3I on a continuum from thermally-driven to information-driven states.	examples/fit_iso_spectrum.py
Q2: 3I’s Missing Micro-Dust – Why are there jets but no Rayleigh-scattering dust?	Simulates how energy allocation shifts from microscopic dust production to macroscopic structure as p increases.	examples/explain_dust_deficit.py
Q3: 3I’s Jet Symmetry – Why 120° triple symmetry?	Demonstrates how the N=3 projection in the model spontaneously generates stable 120° symmetric patterns.	examples/simulate_3i_jets.py
Q4: Testable Predictions – What should we look for in 3I before mid-2026?	Generates synthetic observational signatures (anti-tail coherence, synchronized wobble) based on the model’s parameters for 3I.	examples/generate_predictions.py

**Getting Started in 5 Minutes**
You do not need a complex development environment to run the core simulations.

1. Quick Installation
Ensure you have Python 3.10+. Then install the minimal dependencies:

bash
pip install numpy scipy matplotlib
2. Run the Key Examples
Clone the repo and run the central demonstration:

bash
git clone https://github.com/hkaiopen/InformationDynamics.git
cd InformationDynamics
python examples/fit_iso_spectrum.py  # Reproduces the core p-spectrum (Fig. 1)
python examples/explain_dust_deficit.py # Visualizes the energy allocation argument
These scripts will generate the figures that form the backbone of the paper’s argument.

**Repository Structure**
InformationDynamics/
├── README.md                      # This file
├── requirements.txt               # Minimal Python dependencies
├── src/                           # Core model implementation
│   ├── model.py                   # Solver for the CGLE (Eq. 1)
│   └── projector.py               # Transforms internal state to observables (Eq. 2)
└── examples/                      # Self-contained, runnable scripts
    ├── fit_iso_spectrum.py        # **Core**: Fits data, calculates p, makes Fig. 1
    ├── explain_dust_deficit.py    # **Core**: Shows energy reallocation logic
    ├── simulate_3i_jets.py        # Generates N=3 symmetric jet patterns
    └── generate_predictions.py    # Creates plots for the 3 testable predictions
For a Wider Audience: Beyond Astrophysics
The Information Dynamics framework is, at its heart, a tool for studying pattern formation in nonlinear, dissipative systems far from equilibrium. While applied here to interstellar objects, the core model.py implementing the Complex Ginzburg-Landau equation is agnostic to scale.

Researchers in other fields (e.g., fluid dynamics, active matter, nonlinear optics) may find the provided code a useful starting point for simulating how simple rules (γ, ω, ϵ) can give rise to complex, ordered structures (N-fold symmetries). We welcome explorations and adaptations.

**Data & Reproducibility**
1I & 2I Data: The scripts automatically fetch publicly available orbital and photometric data for 1I/‘Oumuamua and 2I/Borisov via astroquery from NASA JPL Horizons and other archives.

3I/ATLAS Data: Parameters for 3I are based on fits to preliminary acceleration estimates and morphological reports from recent literature (e.g., Hubble analyses). As definitive astrometric and photometric data become public, the fitting scripts can be easily updated.

**License & Citation**
This software is licensed under the MIT License. If you use this code in your research, please cite our accompanying arXiv paper (details to be added upon acceptance).

**Contact**
For questions regarding the scientific framework, please refer to the paper. For technical issues with the code, please open an Issue on this GitHub repository.
