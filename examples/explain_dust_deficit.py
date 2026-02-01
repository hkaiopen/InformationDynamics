# examples/explain_dust_deficit.py
"""
Information Dynamics: A Unified Predictive Framework for Interstellar Objects

Purpose: Semi-quantitative diagnostic for the ISO dust-deficit anomaly.
         Integrates conceptual energy-flow diagrams with observationally
         constrained dust size distribution analysis.

Code: https://github.com/hkaiopen/InformationDynamics
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge

# ============================================================================
# PART 1: OBSERVATIONALLY-CONSTRAINED DUST SIZE DISTRIBUTION ANALYSIS
# ============================================================================

class DustScatteringDiagnostic:
    """
    Semi-quantitative diagnostic comparing scattering-effective dust size
    distributions across ISOs, based on published constraints.
    """
    def __init__(self):
        # Observational constraints on size distribution index (q)
        # q = -d log n(a) / d log a
        self.iso_params = {
            '1I': {'name': "1I/'Oumuamua", 'q': 4.0, 'color': '#d62728',
                   'basis': 'Dust coma non-detection upper limits'},
            '2I': {'name': '2I/Borisov', 'q': 3.5, 'color': '#1f77b4',
                   'basis': 'Classical cometary dust (e.g., Cordiner+ 2020)'},
            '3I': {'name': '3I/ATLAS', 'q': 3.8, 'color': '#2ca02c',
                   'basis': 'Weak Rayleigh scattering, morphology (Jewitt+ 2025)'}
        }
        # Common size range (meters)
        self.a_min = 1e-6   # 1 micron
        self.a_max = 1e-3   # 1 mm

    def compute_scattering_spectrum(self, q):
        """
        Compute scattering-effective distribution: σ(a) ∝ a^2 * a^(-q)
        """
        a_vals = np.logspace(np.log10(self.a_min), np.log10(self.a_max), 500)
        # n(a) ~ a^(-q)
        number_dist = a_vals ** (-q)
        # Geometrical cross-section ~ a^2
        weighted_cross_section = (a_vals ** 2) * number_dist
        return a_vals, weighted_cross_section

    def fraction_in_size_bins(self, a_vals, weighted_cross_section):
        """
        Calculate fractional contribution from small/large grains.
        """
        total = np.trapz(weighted_cross_section, a_vals)

        # Small grains: < 10 microns
        small_mask = a_vals < 1e-5
        small_contrib = np.trapz(weighted_cross_section[small_mask],
                                 a_vals[small_mask]) / total

        # Large grains: > 100 microns
        large_mask = a_vals > 1e-4
        large_contrib = np.trapz(weighted_cross_section[large_mask],
                                 a_vals[large_mask]) / total

        return small_contrib, large_contrib, total

    def run_diagnostic(self):
        """
        Execute full comparative analysis and generate diagnostic plot.
        """
        print("\n" + "="*70)
        print("OBSERVATIONAL DUST DIAGNOSTIC: SCATTERING-EFFECTIVE SIZE DISTRIBUTION")
        print("="*70)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Panel A: Conceptual energy allocation (based on p)
        self._plot_energy_allocation(ax1)

        # Panel B: Quantitative scattering spectra
        print("\n{:25} | {:>20} | {:>20}".format(
            "Object", "Small (<10 μm) Contrib.", "Large (>100 μm) Contrib."))
        print("-" * 75)

        for iso_key, params in self.iso_params.items():
            a_vals, wcs = self.compute_scattering_spectrum(params['q'])
            small_frac, large_frac, _ = self.fraction_in_size_bins(a_vals, wcs)

            # Print quantitative results
            print("{:25} | {:20.1%} | {:20.1%}".format(
                params['name'], small_frac, large_frac))

            # Plot spectrum
            ax2.loglog(a_vals * 1e6,  # convert to microns for x-axis
                      wcs / np.max(wcs),  # normalize for clear comparison
                      label=params['name'],
                      color=params['color'],
                      linewidth=2.5)

        # Format diagnostic plot
        ax2.set_xlabel('Particle Radius (μm)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Relative Scattering Contribution\n(Normalized)', fontsize=12)
        ax2.set_title('Scattering-Effective Size Distribution\n(Observationally Constrained)', 
                     fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, which='both', linestyle='--')
        ax2.axvline(x=10, color='grey', linestyle=':', alpha=0.7, label='10 μm')
        ax2.axvline(x=100, color='black', linestyle=':', alpha=0.7, label='100 μm')

        plt.tight_layout()
        return fig

    def _plot_energy_allocation(self, ax):
        """Create conceptual energy flow diagram."""
        ax.set_xlim(-2, 2)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')

        # Title
        ax.text(0, 1.3, 'Conceptual Energy Allocation\n(Information Purity p)', 
                ha='center', fontsize=12, fontweight='bold')

        # 1I (High p)
        self._draw_object_panel(ax, x=-1.0, p=0.83, name="1I/'Oumuamua",
                               order_frac=0.8, dust_frac=0.0, color='#d62728')
        # 2I (Low p)
        self._draw_object_panel(ax, x=0.0, p=0.09, name='2I/Borisov',
                               order_frac=0.1, dust_frac=0.7, color='#1f77b4')
        # 3I (Mid p)
        self._draw_object_panel(ax, x=1.0, p=0.17, name='3I/ATLAS',
                               order_frac=0.3, dust_frac=0.3, color='#2ca02c')

    def _draw_object_panel(self, ax, x, p, name, order_frac, dust_frac, color):
        """Draw a single object's energy allocation panel."""
        # Core circle
        core = Circle((x, 0), 0.4, facecolor=color, alpha=0.2, edgecolor=color, linewidth=2)
        ax.add_patch(core)

        # Order energy wedge (for macroscopic structures)
        if order_frac > 0:
            order_wedge = Wedge((x, 0), 0.35, 90, 90 + 360 * order_frac,
                               facecolor='#d62728', alpha=0.6, edgecolor='darkred')
            ax.add_patch(order_wedge)

        # Dust energy wedge
        if dust_frac > 0:
            dust_wedge = Wedge((x, 0), 0.35, 90 - 360 * dust_frac, 90,
                              facecolor='#8c564b', alpha=0.6, edgecolor='#5c3c2b')
            ax.add_patch(dust_wedge)

        # Labels
        ax.text(x, 0.6, f'{name}\np={p}', ha='center', fontsize=10, fontweight='bold')
        ax.text(x, 0, f'Order:\n{order_frac:.0%}', ha='center', va='center', fontsize=8)
        ax.text(x, -0.5, f'Dust:\n{dust_frac:.0%}', ha='center', fontsize=8)

# ============================================================================
# PART 2: ENERGY BUDGET QUANTIFICATION FOR 3I/ATLAS
# ============================================================================

def quantify_3I_energy_budget():
    """
    Detailed energy budget for 3I/ATLAS based on information purity p=0.17.
    """
    p = 0.17
    # Simplified model: order_fraction = p^1.5
    order_fraction = p ** 1.5  # ~0.07
    thermal_fraction = 1 - order_fraction  # ~0.93

    # Dust production efficiency (exponentially suppressed with p)
    dust_efficiency = np.exp(-5 * p)  # ~0.43
    dust_production = thermal_fraction * dust_efficiency  # ~0.40

    # Other thermal processes
    other_thermal = thermal_fraction * (1 - dust_efficiency)  # ~0.53

    print("\n" + "="*70)
    print("QUANTITATIVE ENERGY BUDGET: 3I/ATLAS (p = 0.17)")
    print("="*70)
    print("\n{:<30} {:>10} {:>10}".format("Component", "Fraction", "%"))
    print("-" * 50)
    print("{:<30} {:10.3f} {:10.1f}%".format(
        "Order (Jet Maintenance)", order_fraction, order_fraction*100))
    print("{:<30} {:10.3f} {:10.1f}%".format(
        "Thermal → Dust", dust_production, dust_production*100))
    print("{:<30} {:10.3f} {:10.1f}%".format(
        "Thermal → Other", other_thermal, other_thermal*100))
    print("{:<30} {:10.3f} {:10.1f}%".format(
        "TOTAL THERMAL", thermal_fraction, thermal_fraction*100))

    print("\n" + "="*70)
    print("KEY INTERPRETATION:")
    print("="*70)
    print("""
    1. Only ~7% of energy maintains macroscopic jet coherence.
    2. Only ~40% of total energy efficiently produces dust.
    3. This energy allocation shift explains:
       - Strong large-scale jets (order energy)
       - Suppressed sub-micron dust scattering (reduced efficient dust production)
    4. The 'dust deficit' is a dynamical, not compositional, signature.
    """)

    # Create pie chart
    fig, ax = plt.subplots(figsize=(7, 7))
    labels = ['Order (Jets)', 'Dust Production', 'Other Thermal']
    sizes = [order_fraction, dust_production, other_thermal]
    colors = ['#d62728', '#8c564b', '#1f77b4']

    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                     autopct='%1.1f%%', startangle=90,
                                     wedgeprops=dict(edgecolor='w', linewidth=1.5))
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    ax.set_title('3I/ATLAS: Model Energy Budget (p = 0.17)',
                fontsize=14, fontweight='bold', pad=20)
    return fig

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Execute full diagnostic pipeline."""
    print(__doc__)

    # 1. Run comparative dust diagnostic
    diagnostic = DustScatteringDiagnostic()
    fig_diag = diagnostic.run_diagnostic()

    # 2. Quantify 3I energy budget
    fig_budget = quantify_3I_energy_budget()

    # 3. Save outputs
    fig_diag.savefig('dust_deficit_diagnostic_v5.png', dpi=300, bbox_inches='tight')
    fig_budget.savefig('3I_energy_budget_v5.png', dpi=300, bbox_inches='tight')

    print("\n" + "="*70)
    print("OUTPUT SUMMARY:")
    print("="*70)
    print("✅ 1. 'dust_deficit_diagnostic_v5.png' - Comparative diagnostic plot")
    print("✅ 2. '3I_energy_budget_v5.png' - 3I/ATLAS energy budget breakdown")
    print("\n📋 Key conclusion printed above.")
    print("="*70)

    plt.show()

if __name__ == "__main__":
    main()
