"""
infer_dynamical_age.py

Information Dynamics Extension: Dynamical Age Inference from Isotope Ratios

Uses Galactic chemical evolution (GCE) models to infer the formation epoch
of interstellar objects from isotopic signatures.

Key features:
- 12C/13C decreases with cosmic time (AGB & SNe enrichment).
- D/H decreases with cosmic time (stellar astration).
- For cold environments (T < 30 K), apply an additional fractionation
  enhancement to D/H (modeled as an empirical factor).
"""

import numpy as np
from scipy.optimize import brentq
from typing import Dict


class GalacticChemicalEvolution:
    """
    Simplified GCE model for isotope evolution.

    Time convention:
    - t = 0: Big Bang (13.8 Gyr ago)
    - t = 13.8: present day
    - "Age" = 13.8 - t (lookback time from present)

    12C/13C: decreases with time as 13C is produced by AGB stars and SNe.
    Early universe: high 12C/13C (little 13C enrichment)
    Present: low 12C/13C (Solar System ~89)

    D/H: decreases with time due to stellar astration (no net production).
    Primordial: ~2.5e-5 (BBN)
    Present ISM: ~1.5e-5 (depleted)

    Note: 3I/ATLAS D/H = 9.8e-3 is EXTREMELY high. This is not a GCE effect
    but extreme isotopic fractionation in a cold, unprocessed environment.
    We model it with an additional "cold fractionation enhancement" factor.
    """

    def __init__(self, t_univ_Gyr: float = 13.8):
        self.t_univ = t_univ_Gyr

        # 12C/13C evolution parameters
        self.C13_early = 200.0   # at early times (z ~ 4-5)
        self.C13_late = 89.0     # present-day Solar System
        self.C13_tau = 4.0       # enrichment timescale (Gyr)

        # D/H evolution parameters
        self.DH_primordial = 2.5e-5   # BBN value
        self.DH_present = 1.5e-5      # present ISM
        self.DH_tau = 8.0             # depletion timescale (Gyr)

        # Cold fractionation enhancement for cometary ices
        # In cold environments (T < 30 K), D/H can be enhanced by
        # gas-grain chemistry and ion-molecule reactions
        self.DH_cold_enhancement = 400.0  # factor for T < 30 K

    def C13_vs_t(self, t_Gyr: float) -> float:
        """12C/13C as function of cosmic time t (0 = Big Bang)."""
        return self.C13_late + (self.C13_early - self.C13_late) * np.exp(-t_Gyr / self.C13_tau)

    def DH_vs_t(self, t_Gyr: float, T_form: float = 30.0) -> float:
        """
        D/H as function of cosmic time, with cold fractionation.

        For cold environments (T < 50 K), apply additional fractionation.
        """
        # Base GCE depletion
        DH_base = self.DH_present + (self.DH_primordial - self.DH_present) * np.exp(-t_Gyr / self.DH_tau)

        # Cold fractionation enhancement
        if T_form < 50.0:
            # Enhancement decreases with time (more processed at late times)
            enhancement = self.DH_cold_enhancement * np.exp(-t_Gyr / 5.0)
            DH_base *= (1.0 + enhancement)

        return DH_base

    def infer_age_from_C13(self, C13: float) -> Dict:
        """Infer formation epoch from 12C/13C."""
        def equation(t):
            return self.C13_vs_t(t) - C13

        try:
            # Search in [0.1, 13.5] Gyr
            t_form = brentq(equation, 0.1, 13.5)
            age_Gyr = self.t_univ - t_form
            z = self._t_to_z(t_form)
            return {
                't_cosmic_Gyr': float(t_form),
                'age_Gyr': float(age_Gyr),
                'redshift': float(z),
                'method': '12C/13C',
                'valid': True
            }
        except ValueError:
            return {
                't_cosmic_Gyr': np.nan, 'age_Gyr': np.nan,
                'redshift': np.nan, 'method': '12C/13C', 'valid': False
            }

    def infer_age_from_DH(self, DH: float, T_form: float = 30.0) -> Dict:
        """Infer formation epoch from D/H."""
        def equation(t):
            return self.DH_vs_t(t, T_form) - DH

        try:
            # FIXED: lower bound 0.1 -> 0.01 because root is near ~0.08 Gyr
            t_form = brentq(equation, 0.01, 13.5)
            age_Gyr = self.t_univ - t_form
            z = self._t_to_z(t_form)
            return {
                't_cosmic_Gyr': float(t_form),
                'age_Gyr': float(age_Gyr),
                'redshift': float(z),
                'method': 'D/H',
                'valid': True
            }
        except ValueError:
            return {
                't_cosmic_Gyr': np.nan, 'age_Gyr': np.nan,
                'redshift': np.nan, 'method': 'D/H', 'valid': False
            }

    def _t_to_z(self, t_Gyr: float) -> float:
        """Approximate t-z relation (flat LambdaCDM, H0=70)."""
        if t_Gyr > 12:
            return 0.1
        elif t_Gyr < 0.5:
            return 10.0
        else:
            # Simplified: z ~ exp((12 - t)/2.2) - 1
            return float(np.exp((12.0 - t_Gyr) / 2.2) - 1.0)


class DynamicalAgeInference:
    """Combines chemical age with dynamical wandering time."""

    def __init__(self):
        self.gce = GalacticChemicalEvolution()
        self.v_inf = 60.0e3  # m/s

    def wandering_time(self, d_travel_pc: float = 5000.0) -> float:
        """Estimate ISM wandering time."""
        d_m = d_travel_pc * 3.086e16
        t_s = d_m / self.v_inf
        return t_s / (1e9 * 365.25 * 86400)

    def infer_full_age(self, C13: float, DH: float,
                       T_form: float = 30.0) -> Dict:
        """Full age inference."""
        age_C = self.gce.infer_age_from_C13(C13)
        age_D = self.gce.infer_age_from_DH(DH, T_form)

        # Weighted average (C13 more reliable for old ages)
        ages = []
        weights = []
        if age_C['valid']:
            ages.append(age_C['age_Gyr'])
            weights.append(2.0)
        if age_D['valid']:
            ages.append(age_D['age_Gyr'])
            weights.append(1.0)

        if ages:
            age_chem = np.average(ages, weights=weights)
        else:
            age_chem = np.nan

        t_wander = self.wandering_time()
        t_formation = age_chem - t_wander if not np.isnan(age_chem) else np.nan

        return {
            'age_chemical_Gyr': float(age_chem),
            'age_carbon_Gyr': float(age_C['age_Gyr']) if age_C['valid'] else np.nan,
            'age_hydrogen_Gyr': float(age_D['age_Gyr']) if age_D['valid'] else np.nan,
            't_wandering_Gyr': float(t_wander),
            't_formation_Gyr': float(t_formation),
            'consistency_with_Nature': 9.0 < age_chem < 13.0
        }


def main():
    print("=" * 70)
    print("DYNAMICAL AGE INFERENCE: 3I/ATLAS")
    print("=" * 70)

    dai = DynamicalAgeInference()

    # Inputs (preliminary from literature)
    C13 = 147.0          # assumed 12C/13C from ALMA (needs verification)
    DH = 9.8e-3          # observed D/H for 3I/ATLAS
    T_form = 30.0        # formation temperature (K)

    result = dai.infer_full_age(C13, DH, T_form)

    # Individual constraints
    age_C = dai.gce.infer_age_from_C13(C13)
    age_D = dai.gce.infer_age_from_DH(DH, T_form)

    print(f"\n12C/13C = {C13:.0f}:")
    print(f"  Cosmic time t = {age_C['t_cosmic_Gyr']:.1f} Gyr after Big Bang")
    print(f"  → Formation age = {age_C['age_Gyr']:.1f} Gyr ago")
    print(f"  → Redshift z ≈ {age_C['redshift']:.1f}")

    print(f"\nD/H = {DH:.2e} (with cold fractionation at T={T_form} K):")
    print(f"  Cosmic time t = {age_D['t_cosmic_Gyr']:.1f} Gyr after Big Bang")
    print(f"  → Formation age = {age_D['age_Gyr']:.1f} Gyr ago")
    print(f"  → Redshift z ≈ {age_D['redshift']:.1f}")

    print(f"\nCombined inference:")
    print(f"  Chemical age: {result['age_chemical_Gyr']:.1f} Gyr")
    print(f"  Wandering time: {result['t_wandering_Gyr']:.1f} Gyr")
    print(f"  Parent system formation: {result['t_formation_Gyr']:.1f} Gyr ago")

    print(f"\nComparison with Nature 2026 (10-12 Gyr):")
    print(f"  Consistent: {result['consistency_with_Nature']}")


if __name__ == "__main__":
    main()