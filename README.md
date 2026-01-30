# Information Dynamics: A Unified Predictive Framework for Interstellar Objects

This repository accompanies the manuscript **“Information Dynamics: A Unified Predictive Framework for Interstellar Objects — With Immediate Testable Predictions for 3I/ATLAS”** (Huang & Liu, 2026).

It provides the computational core and analysis scripts for the proposed phenomenological model, enabling the reproduction of key results, the simulation of interstellar object (ISO) behaviors, and the verification of testable predictions.

## Core Scientific Framework

The **Information Dynamics** framework explains the diverse behaviors of interstellar objects through the competition between traditional thermal driving (`F_thermal`) and internal, coherent self-organization (nonlinear term `ε|Ψ|²Ψ`). The key unifying parameter is the **information purity `p`**:

`p = ε / (γ + ε)`

This scalar quantifies the dominance of self-organizing, information-driven processes (high `p`) over dissipative, thermally-driven processes (low `p`), placing all known ISOs on a continuous spectrum.

## Core Scientific Questions & Computational Answers

The code in this repository directly addresses the main puzzles outlined in the paper:

| Question | What This Code Does | Paper's Explanation |
| :--- | :--- | :--- |
| **Q1: The ISO Spectrum** – Can three seemingly different ISOs be described by one model? | **Fits the `p`-spectrum**, placing 1I, 2I, and 3I on a continuum. [`examples/fit_iso_spectrum.py`](examples/fit_iso_spectrum.py) | Yes. 1I/’Oumuamua (`p=0.83`) is **information-driven** (minimal coma), 2I/Borisov (`p=0.09`) is **thermally-driven** (classic comet), and 3I/ATLAS (`p=0.17`) is in a **mixed state** (structured jets, dust deficit). |
| **Q2: 3I's Missing Micro-Dust** – Why are there jets but no Rayleigh-scattering dust? | **Simulates energy allocation**, showing how energy shifts from dust production to macroscopic order as `p` increases. [`examples/explain_dust_deficit.py`](examples/explain_dust_deficit.py) | At `p=0.17`, energy is channeled into building macroscopic jet structures rather than fragmenting into sub-micron dust. The “dust deficit” is a signature of this mixed dynamical state. |
| **Q3: 3I's Jet Symmetry** – Why 120° triple symmetry? | **Demonstrates pattern formation** from the CGLE, showing stable 120° symmetric patterns emerge naturally. [`examples/simulate_3i_jets.py`](examples/simulate_3i_jets.py) | The `N=3` symmetry parameter in the observable projection (`C_k`) naturally creates stable 120° patterns when the nonlinear term (`ε`) is significant—an emergent property of the mixed state. |
| **Q4: Testable Predictions** – What should we look for in 3I before mid-2026? | **Generates synthetic observables** (anti-tail, wobble) based on the fitted model parameters for 3I. [`examples/generate_predictions.py`](examples/generate_predictions.py) | Predicts: 1) Coherent anti-tail, 2) Synchronized jet wobble, 3) Discrete activity jumps. **Verification of any two would strongly support the model.** |
| **Q5: Predictions for the Jan 22, 2026 Alignment** – What unique signatures occur during the Sun-Earth-3I alignment? | **Simulates alignment-specific effects**: enhanced forward-scattering, polarization shifts, and optimal anti-tail visibility. [`examples/predict_jan22_alignment.py`](examples/predict_jan22_alignment.py) | The near-perfect alignment (`α ≈ 0.69°`) provides a **critical, time-sensitive test** of jet coherence and material properties predicted by the model. |

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
# Q1: Fit the ISO spectrum and generate the p-continuum plot
python examples/fit_iso_spectrum.py

# Q2: Visualize the energy reallocation explaining 3I's dust deficit
python examples/explain_dust_deficit.py

# Q3: Simulate the emergence of 3I's 120° symmetric jet patterns
python examples/simulate_3i_jets.py

# Q4: Generate visualizations for the three primary testable predictions
python examples/generate_predictions.py

# Q5: Generate predictions for the crucial Sun-Earth-3I alignment event (Jan 22, 2026)
python examples/predict_jan22_alignment.py
```

## Code Structure

```
InformationDynamics/
├── information_dynamics_model.py      # Core CGLE numerical implementation
├── iso_parameter_fitting.py           # Parameter fitting & Table 1 generation
└── examples/                          # Self-contained demonstration scripts
    ├── fit_iso_spectrum.py            # Q1: ISO spectrum fitting
    ├── explain_dust_deficit.py        # Q2: Dust deficit explanation
    ├── simulate_3i_jets.py            # Q3: Jet symmetry simulation
    ├── generate_predictions.py        # Q4: Testable predictions visualization
    └── predict_jan22_alignment.py     # Q5: Alignment-specific predictions
```

## Core Model Implementation

The model is based on a simplified Complex Ginzburg-Landau Equation (CGLE), as defined in the paper:

```python
# Core implementation from information_dynamics_model.py
class InformationDynamicsModel:
    def __init__(self, gamma, omega, epsilon, F_thermal=0.0):
        self.gamma = gamma      # Linear dissipation rate (s⁻¹)
        self.omega = omega      # Characteristic frequency (rad/s)
        self.epsilon = epsilon  # Nonlinear self-interaction strength (s⁻¹)
        self.F_thermal = F_thermal  # Traditional thermal forcing

    def information_purity(self):
        """Calculate p = ε/(γ + ε). Quantifies dominance of self-organization."""
        return self.epsilon / (self.gamma + self.epsilon + 1e-12)
```

**Verification Note**: The implementation of the core equation `∂Ψ/∂t = (γ - iω)Ψ + ε|Ψ|²Ψ + F_thermal + η(t)` and the calculation of `p` have been cross-checked against the manuscript.

## Fitted Parameter Space (Table 1)

The best-fit parameters for the three known interstellar objects, as presented in the paper, are below. These values are verified in `iso_parameter_fitting.py`.

| Object | γ (10⁻⁵ s⁻¹) | ω (10⁻⁴ rad/s) | ε (10⁻⁴) | **p** | State Classification |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1I/’Oumuamua** | 0.12 ± 0.05 | 2.18 ± 0.10 | 0.059 ± 0.005 | **0.83** | Information-Driven |
| **2I/Borisov** | 21 ± 3 | 1.75 ± 0.15 | 0.21 ± 0.03 | **0.09** | Thermally-Driven |
| **3I/ATLAS** | 3.8 ± 0.5 | 1.08 ± 0.02 | 0.078 ± 0.007 | **0.17** | Mixed State |

**Verification Note**: The derived `p` values (`p = ε/(γ+ε)`) using the tabulated γ and ε yield 0.83, 0.09, and 0.17, confirming the phenomenological classification.

## Urgent, Testable Predictions for 3I/ATLAS

The model makes concrete, time-sensitive predictions for 3I/ATLAS before it becomes unobservable in mid-2026. The **Sun-Earth-3I alignment around January 22, 2026 (α ≈ 0.69°)** provides a unique verification window.

### Primary Predictions (Leveraging Jan 22, 2026 Alignment):
1.  **Coherent Anti-Tail**: A narrow, sunward-facing anti-tail structure that persists for weeks without rapid dissipation, maintained by the self-organizing term (`ε`). *Initial Jan 2026 Hubble data show enhanced anti-tail visibility.*
2.  **Synchronized Jet Wobble**: If present, jets will wobble **in phase** while preserving 120° separation. Predicted wobble period: **P_wobble ≈ 16.2 hours** (based on ω = 1.08×10⁻⁴ rad/s). *Jan 2026 TESS photometry shows rotation periods of ~15-17 hours.*
3.  **Discrete Activity Jumps**: Step-like changes in total brightness/non-gravitational acceleration, uncorrelated with heliocentric distance, caused by noise-induced state transitions (`η(t)`).

### Secondary Prediction:
4.  **High-Frequency Gravitational Response**: Close planetary encounters would temporarily increase `F_thermal` (outgassing) without disrupting the underlying coherent jet structure, a signature distinct from traditional cometary response.

**Verification of any two of Predictions 1-3 before mid-2026 would constitute strong evidence for the model.**

## Updated Observational Context (January 2026)

Recent observations (Hubble, TESS) have refined the geometric constraints for 3I/ATLAS, which are used as inputs for alignment predictions:
*   **Apparent Magnitude**: V ≈ 16.7
*   **Heliocentric Distance**: ≈ 3.33 AU
*   **Geocentric Distance**: ≈ 2.35 AU
*   **Opposition Surge Amplitude**: Estimated 0.3 - 0.5 mag

These updates refine geometric predictions but **do not alter the core fitted model parameters** (γ, ω, ε, p) presented in Table 1.

## Data and Code Availability

*   **All code** for fitting, analysis, and figure generation is publicly available in this repository.
*   **Underlying raw observations** are available through the [Mikulski Archive for Space Telescopes (MAST)](https://archive.stsci.edu/).
*   **Orbital elements and ephemerides** were retrieved from NASA's [JPL Horizons system](https://ssd.jpl.nasa.gov/horizons.cgi).

## Citation

If you use this code or the associated framework in your research, please cite the accompanying paper:
> Huang, K., & Liu, H. (2026). Information Dynamics: A Unified Predictive Framework for Interstellar Objects — With Immediate Testable Predictions for 3I/ATLAS. *[JaPL]*.

## License

This project is licensed under the MIT License.

---

**This framework not only unifies existing ISO anomalies but generates immediate, testable predictions. We strongly encourage observational campaigns to target the unique Sun-Earth-3I alignment window in early 2026.**
