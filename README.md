# Information Dynamics: A Unified Predictive Framework for Interstellar Objects

## Core Scientific Questions & Computational Answers

This repository provides computational answers to the main puzzles outlined in the paper. Each core question is addressed by a specific script or module that demonstrates how the Information Dynamics framework explains ISO anomalies through numerical simulation and visualization.

### Core Scientific Questions

| Question | What This Code Does | Paper's Explanation (in a nutshell) |
|----------|-------------------|--------------------------------------|
| **Q1: The ISO Spectrum** – Can three seemingly different ISOs be described by one model? | Fits the p-spectrum, placing 1I, 2I, and 3I on a continuum from thermally-driven to information-driven states.  [`examples/fit_iso_spectrum.py`](examples/fit_iso_spectrum.py) | Yes, all three ISOs lie on a continuous spectrum defined by information purity p = ε/(γ+ε): 1I (p=0.83) is information-driven, 2I (p=0.09) is thermally-driven, and 3I (p=0.17) is in a mixed state. |
| **Q2: 3I's Missing Micro-Dust** – Why are there jets but no Rayleigh-scattering dust? | Simulates how energy allocation shifts from microscopic dust production to macroscopic structure as p increases.  [`examples/explain_dust_deficit.py`](examples/explain_dust_deficit.py) | At p=0.17 (mixed state), energy is channeled into building macroscopic jet structures rather than fragmenting into sub-micron dust. The "dust deficit" is a predictable signature of this dynamical state. |
| **Q3: 3I's Jet Symmetry** – Why 120° triple symmetry? | Demonstrates how the N=3 projection in the model spontaneously generates stable 120° symmetric patterns.  [`examples/simulate_3i_jets.py`](examples/simulate_3i_jets.py) | The N=3 symmetry parameter in the observable projection naturally creates stable 120° patterns when the nonlinear term (ε) is significant. This is an emergent property of the mixed dynamical state. |
| **Q4: Testable Predictions** – What should we look for in 3I before mid-2026? | Generates synthetic observational signatures (anti-tail coherence, synchronized wobble) based on the model's parameters for 3I.  [`examples/generate_predictions.py`](examples/generate_predictions.py) | Three concrete predictions: 1) Coherent anti-tail, 2) Synchronized jet wobble with period ~6.7 days, 3) Discrete activity jumps. Verification of any two would strongly support the model. |

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/hkaiopen/InformationDynamics.git
cd InformationDynamics

# Install dependencies
pip install numpy pandas matplotlib scipy
```

### Run Core Examples

```bash
# Q1: Fit the ISO spectrum and generate Figure 1
python examples/fit_iso_spectrum.py

# Q2: Explain 3I's dust deficit through energy reallocation
python examples/explain_dust_deficit.py

# Q3: Simulate 3I's N=3 symmetric jet patterns
python examples/simulate_3i_jets.py

# Q4: Generate visualizations for the three testable predictions
python examples/generate_predictions.py
```

## Code Structure

```
InformationDynamics/
├── information_dynamics_model.py      # Core CGLE implementation
├── iso_parameter_fitting.py           # Parameter analysis & Table 1
└── examples/                          # Self-contained scripts
    ├── fit_iso_spectrum.py            # Q1: ISO spectrum fitting
    ├── explain_dust_deficit.py        # Q2: Dust deficit explanation
    ├── simulate_3i_jets.py            # Q3: Jet symmetry simulation
    ├── generate_predictions.py        # Q4: Testable predictions
    └── predict_jan22_alignment.py     # P1: Predictions for Sun-Earth-3I alignment
```

## Core Model Implementation

The Information Dynamics model is based on a simplified Complex Ginzburg-Landau Equation (CGLE):

```python
# From information_dynamics_model.py
class InformationDynamicsModel:
    def __init__(self, gamma, omega, epsilon, F_thermal=0.0):
        self.gamma = gamma      # Linear dissipation rate
        self.omega = omega      # Characteristic frequency
        self.epsilon = epsilon  # Nonlinear self-interaction
        self.F_thermal = F_thermal  # Thermal forcing
        
    def information_purity(self):
        """p = ε/(γ + ε) - quantifies dominance of self-organization"""
        return self.epsilon / (self.gamma + self.epsilon)
```

## Parameter Space

| Object | γ (10⁻⁵ s⁻¹) | ω (10⁻⁴ rad/s) | ε (10⁻⁴) | p |
|--------|--------------|----------------|----------|----|
| 1I/'Oumuamua | 0.12 ± 0.05 | 2.18 ± 0.10 | 0.059 ± 0.005 | 0.83 |
| 2I/Borisov | 21 ± 3 | 1.75 ± 0.15 | 0.21 ± 0.03 | 0.09 |
| 3I/ATLAS | 3.8 ± 0.5 | 1.08 ± 0.02 | 0.078 ± 0.007 | 0.17 |

## Three Testable Predictions for 3I/ATLAS

1. **Anti-tail Coherence**: Narrow, coherent structure resistant to solar wind dispersion
2. **Synchronized Wobble**: Jets wobble in phase with period ~6.73 days (2π/ω)
3. **Discrete Activity Jumps**: Step-like changes uncorrelated with heliocentric distance

**Verification of any two predictions before mid-2026 would support the model.**
