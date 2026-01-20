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
| **Q5: Predictions for Sun-Earth-3I Alignment** – What unique observational signatures occur during specific geometric alignments? | Simulates changes in forward-scattering brightness, polarization, and tail/anti-tail visibility for alignment events like Jan 22, 2026. [`examples/predict_jan22_alignment.py`](examples/predict_jan22_alignment.py) | The model predicts that during close Sun-Earth-3I alignment: 1) Forward-scattering enhances brightness, 2) Polarization signals may shift due to ordered jet material vs. dust, 3) Anti-tail visibility is maximized, providing a critical test of jet coherence. |

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

# Q5: Generate predictions for the Sun-Earth-3I alignment event
python examples/predict_jan22_alignment.py
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
    └── predict_jan22_alignment.py     # Q5: Predictions for Sun-Earth-3I alignment
```

## Core Model Implementation

The Information Dynamics model is based on a simplified Complex Ginzburg-Landau Equation (CGLE):

```python
# From information_dynamics_model.py
class InformationDynamicsModel:
    def __init__(self, gamma, omega, epsilon, F_thermal=0.0):
        self.gamma = gamma      # Linear dissipation rate (s⁻¹)
        self.omega = omega      # Characteristic frequency (rad/s)
        self.epsilon = epsilon  # Nonlinear self-interaction strength (s⁻¹)
        self.F_thermal = F_thermal  # Thermal forcing
        
    def information_purity(self):
        """p = ε/(γ + ε) - quantifies dominance of self-organization"""
        return self.epsilon / (self.gamma + self.epsilon + 1e-12)
```

**Verification Note**: The model implementation has been cross-checked with the paper. The core equation `∂Ψ/∂t = -γΨ + iωΨ + ε|Ψ|²Ψ + F_thermal + η(t)` and the calculation of information purity `p` are correctly implemented.

## Parameter Space

The best-fit parameters for the three known interstellar objects are as follows. These values have been verified against Table 1 in the paper and the numerical calculations in `iso_parameter_fitting.py`.

| Object | γ (10⁻⁵ s⁻¹) | ω (10⁻⁴ rad/s) | ε (10⁻⁴) | p |
|--------|--------------|----------------|----------|----|
| 1I/'Oumuamua | 0.12 ± 0.05 | 2.18 ± 0.10 | 0.059 ± 0.005 | 0.83 |
| 2I/Borisov | 21 ± 3 | 1.75 ± 0.15 | 0.21 ± 0.03 | 0.09 |
| 3I/ATLAS | 3.8 ± 0.5 | 1.08 ± 0.02 | 0.078 ± 0.007 | 0.17 |

**Verification Note**: The parameter values and the derived information purity `p` have been cross-validated. The calculations `p = ε/(γ+ε)` using the tabulated γ and ε values yield 0.83, 0.09, and 0.17 respectively, matching the stated `p` values and the phenomenological classifications (Information-Driven, Thermally-Driven, Mixed State).

## Three Testable Predictions for 3I/ATLAS

1.  **Anti-tail Coherence**: Narrow, coherent structure resistant to solar wind dispersion due to ε-driven macroscopic order.
2.  **Synchronized Wobble**: Jets wobble in phase while preserving 120° separation, with a period of ~6.73 days (T = 2π/ω).
3.  **Discrete Activity Jumps**: Step-like changes in brightness and non-gravitational acceleration uncorrelated with heliocentric distance, caused by noise-induced state transitions.

**Verification of any two predictions before mid-2026 would strongly support the model.**

## Testable Predictions for Sun-Earth-3I Alignment

The model makes specific predictions for periods when the Sun, Earth, and 3I/ATLAS are in near-perfect alignment (e.g., around **Jan 22, 2026**). These geometric conditions provide unique observational tests:

1.  **Forward-Scattering Brightness Surge**: When Earth is close to the Sun-3I line, forward-scattering by ordered jet material (not dust) is predicted to cause a sharp, short-term increase in total brightness. This contrasts with the smoother surge expected from typical cometary dust.
2.  **Polarization Signature Shift**: The polarization phase curve is expected to show a distinct feature, as the aligned, ε-driven jet material scatters light differently than an isotropic cloud of microscopic dust.
3.  **Optimal Anti-tail Visibility**: The alignment provides the best geometry to observe the sunward anti-tail predicted by the model. Its persistence and narrow width during this period would be a critical test of the ε-driven coherence against solar wind disruption.

These alignment-specific predictions offer a high-signal opportunity to distinguish the Information Dynamics framework from traditional cometary models before 3I/ATLAS becomes unobservable.
