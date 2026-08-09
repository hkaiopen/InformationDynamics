"""
isotope_information_fidelity.py

Information Dynamics Extension: Chemical Information Fidelity from Isotope Ratios

Maps extreme isotope anomalies to a chemical information fidelity metric,
and compares it with the dynamical p = 0.17 from the CGLE model.

"""

import numpy as np
from typing import Dict, Optional


class IsotopeInformationFidelity:
    """
    Computes chemical information fidelity from isotopic deviations.

    Two complementary metrics:
    1. Chemical Exoticity Index (E_chem): how far from Solar System baseline
    2. Chemical Information Fidelity (F_chem): primordial memory retention

    Solar System bodies cluster at (f_lock ~ 0-2.5, Y_C ~ 0).
    3I/ATLAS is at (f_lock = 6.48, Y_C = 0.50), far outside the cluster.
    """

    # Solar System baseline and scatter (from Huang & Liu 2026b)
    SOLAR_C13 = 89.0
    SS_DH_RANGE = (1.5e-5, 5.3e-4)   # Sun to comets
    SS_YC_RANGE = (-0.06, 0.05)      # Solar System carbon baseline band
    SS_FLOCK_RANGE = (0.3, 2.5)      # Solar System f_lock range

    def __init__(self):
        self.f_earth = np.log(1.5e-4 / 1.5e-5)  # ~2.30

    def compute_f_lock(self, DH: float) -> float:
        """Hydrogen information coordinate: ln(DH / 1.5e-5)."""
        return np.log(DH / 1.5e-5)

    def compute_Y_C(self, C12_C13: float) -> float:
        """Carbon baseline deviation: ln(12C/13C / 89)."""
        return np.log(C12_C13 / self.SOLAR_C13)

    def compute_Y_N(self, N14_N15: float) -> float:
        """Nitrogen baseline deviation: ln(14N/15N / 272)."""
        return np.log(N14_N15 / 272.0)

    def compute_exoticity(self, DH: float, C12_C13: float,
                          N14_N15: Optional[float] = None) -> Dict:
        """
        Compute Chemical Exoticity Index E_chem.

        E_chem measures how many "Solar System sigmas" the object deviates.
        E_chem = 1: marginally outside SS
        E_chem = 10: strongly exotic
        """
        f_lock = self.compute_f_lock(DH)
        Y_C = self.compute_Y_C(C12_C13)

        # Deviation in f_lock (normalized by SS range)
        ss_flock_width = self.SS_FLOCK_RANGE[1] - self.SS_FLOCK_RANGE[0]
        delta_f = max(0, (f_lock - self.SS_FLOCK_RANGE[1]) / ss_flock_width)

        # Deviation in Y_C (normalized by SS scatter)
        ss_yc_width = self.SS_YC_RANGE[1] - self.SS_YC_RANGE[0]
        delta_y = abs(Y_C) / ss_yc_width

        # Combined exoticity (quadrature sum)
        E_chem = np.sqrt(delta_f**2 + delta_y**2)

        # Chemical Information Fidelity
        # High exoticity = high primordial memory = high F_chem
        # Using sigmoid with characteristic scale E0 = 5
        E0 = 5.0
        F_chem = 1.0 / (1.0 + np.exp(-(E_chem - E0) / 2.0))

        # Alternative: direct mapping to p-scale
        # Map E_chem ~ [0, 20] to p_chem ~ [0, 1] via tanh
        p_chem = np.tanh(E_chem / 10.0)

        return {
            'E_chem': float(E_chem),
            'F_chem': float(F_chem),
            'p_chem': float(p_chem),
            'f_lock': float(f_lock),
            'Y_C': float(Y_C),
            'Y_N': float(self.compute_Y_N(N14_N15)) if N14_N15 else None,
            'delta_f': float(delta_f),
            'delta_y': float(delta_y)
        }

    def compare_with_dynamical_p(self, result: Dict, p_dyn: float = 0.17) -> Dict:
        """
        Compare chemical metrics with dynamical p.

        Physical interpretation:
        - If F_chem >> p_dyn: chemical memory is well-preserved despite
          low dynamical order (common for ancient ISOs)
        - If F_chem ~ p_dyn: chemical and dynamical states are coupled
        - If F_chem << p_dyn: dynamical order exceeds chemical memory
        """
        F_chem = result['F_chem']
        p_chem = result['p_chem']

        ratio = F_chem / p_dyn if p_dyn > 0 else np.inf

        if ratio > 3.0:
            interpretation = (
                "Chemical memory strongly exceeds dynamical order. "
                "The object retains primordial chemistry despite "
                "thermally-driven surface activity."
            )
        elif ratio > 0.5:
            interpretation = (
                "Chemical and dynamical states are moderately coupled."
            )
        else:
            interpretation = (
                "Dynamical order exceeds chemical memory. "
                "Surface activity has erased primordial signatures."
            )

        return {
            'p_dyn': p_dyn,
            'F_chem': F_chem,
            'p_chem': p_chem,
            'F_chem_over_p_dyn': float(ratio),
            'interpretation': interpretation,
            'consistency': 0.3 < ratio < 3.0
        }

    def plot_diagram(self, DH: float, C12_C13: float,
                     N14_N15: Optional[float] = None,
                     save_path: str = 'isotope_fidelity.png'):
        """Create f_lock - Y_C diagram."""
        import matplotlib.pyplot as plt

        result = self.compute_exoticity(DH, C12_C13, N14_N15)
        comp = self.compare_with_dynamical_p(result)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Solar System baseline band
        ax.axhspan(-0.06, 0.05, alpha=0.15, color='blue', label='SS carbon baseline')
        ax.axhline(0, color='blue', linestyle='--', alpha=0.3)

        # Solar System bodies
        bodies = {
            'Sun': (0.34, 0.049), 'Jupiter': (0.41, 0.010),
            'Earth': (2.34, 0.009), 'Moon': (2.30, 0.019),
            'Mars': (1.65, 0.012), 'Venus': (2.52, -0.008),
            '67P': (1.26, -0.057)
        }
        for name, (f, y) in bodies.items():
            ax.scatter(f, y, s=80, color='blue', alpha=0.6, zorder=3)
            ax.annotate(name, (f, y), textcoords="offset points", 
                       xytext=(4, 4), fontsize=8)

        # 3I/ATLAS
        ax.scatter(result['f_lock'], result['Y_C'], s=400, color='red',
                  marker='*', zorder=5, label=f"3I/ATLAS (E={result['E_chem']:.1f})")

        # Exoticity contours
        f_grid = np.linspace(-1, 8, 200)
        y_grid = np.linspace(-0.3, 0.8, 200)
        F, Y = np.meshgrid(f_grid, y_grid)
        ss_fw = self.SS_FLOCK_RANGE[1] - self.SS_FLOCK_RANGE[0]
        ss_yw = self.SS_YC_RANGE[1] - self.SS_YC_RANGE[0]
        df = np.maximum(0, (F - self.SS_FLOCK_RANGE[1]) / ss_fw)
        dy = np.abs(Y) / ss_yw
        E = np.sqrt(df**2 + dy**2)
        ax.contour(F, Y, E, levels=[1, 3, 5, 10], colors='gray',
                  linestyles='--', alpha=0.5)
        ax.clabel(ax.contour(F, Y, E, levels=[1, 3, 5, 10], colors='gray'),
                 inline=True, fontsize=8, fmt='E=%1.0f')

        ax.set_xlabel(r'$f_{\mathrm{lock}} = \ln(D/H / 1.5\times10^{-5})$', fontsize=12)
        ax.set_ylabel(r'$Y_C = \ln(^{12}C/^{13}C / 89)$', fontsize=12)
        ax.set_title(
            f'Isotope Exoticity Diagram\n'
            f"F_chem={comp['F_chem']:.2f}, p_dyn={comp['p_dyn']:.2f}, "
            f"Ratio={comp['F_chem_over_p_dyn']:.1f}",
            fontsize=12, fontweight='bold'
        )
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-1, 8)
        ax.set_ylim(-0.3, 0.8)

        plt.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return save_path


def main():
    print("=" * 70)
    print("ISOTOPE INFORMATION FIDELITY: 3I/ATLAS")
    print("=" * 70)

    iif = IsotopeInformationFidelity()

    # 3I/ATLAS data
    DH = 9.8e-3
    C12_C13 = 147.0
    N14_N15 = 363.0

    result = iif.compute_exoticity(DH, C12_C13, N14_N15)
    comp = iif.compare_with_dynamical_p(result, p_dyn=0.17)

    print(f"\n3I/ATLAS Isotope Data:")
    print(f"  D/H = {DH:.2e}  →  f_lock = {result['f_lock']:.3f}")
    print(f"  12C/13C = {C12_C13:.0f}  →  Y_C = {result['Y_C']:.3f}")
    print(f"  14N/15N = {N14_N15:.0f}  →  Y_N = {result['Y_N']:.3f}")

    print(f"\nChemical Exoticity:")
    print(f"  E_chem = {result['E_chem']:.2f} (Solar System sigma units)")
    print(f"  delta_f = {result['delta_f']:.2f}, delta_y = {result['delta_y']:.2f}")

    print(f"\nChemical Information Metrics:")
    print(f"  F_chem (primordial memory) = {comp['F_chem']:.3f}")
    print(f"  p_chem (scaled exoticity)  = {comp['p_chem']:.3f}")
    print(f"  p_dyn (dynamical)          = {comp['p_dyn']:.3f}")
    print(f"  F_chem / p_dyn             = {comp['F_chem_over_p_dyn']:.2f}")
    print(f"  Consistency (0.3<r<3)      = {comp['consistency']}")

    print(f"\nInterpretation:")
    print(f"  {comp['interpretation']}")

    plot_path = iif.plot_diagram(DH, C12_C13, N14_N15)
    print(f"\nPlot saved to: {plot_path}")


if __name__ == "__main__":
    main()
