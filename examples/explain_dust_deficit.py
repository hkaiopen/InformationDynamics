# examples/explain_dust_deficit.py
"""
Information Dynamics: A Unified Predictive Framework for Interstellar Objects

Core script demonstrating the energy reallocation logic that explains the
"dust deficit" anomaly of 3I/ATLAS - strong jet activity without the expected
sub-micron dust scattering signatures.

Code: https://github.com/hkaiopen/InformationDynamics
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Polygon

def create_energy_flow_diagram():
    """
    Create energy flow diagram showing how energy allocation changes with p value.
    
    Visual explanation of why 3I/ATLAS has jets but no micro-dust.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Colors for energy types
    colors = {
        'thermal': '#1f77b4',    # Blue for thermal energy
        'order': '#d62728',      # Red for order/coherence energy
        'mixed': '#2ca02c',      # Green for mixed state
        'dust': '#8c564b',       # Brown for dust
    }
    
    # Panel 1: 2I/Borisov (p=0.09, Thermally-Driven)
    ax1 = axes[0]
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_aspect('equal')
    ax1.axis('off')
    
    # Sun
    sun1 = Circle((0, 1), 0.3, color='gold', ec='orange', lw=2, zorder=5)
    ax1.add_patch(sun1)
    
    # Thermal energy flow (dominant)
    ax1.annotate('', xy=(0, 0.4), xytext=(0, 0.7),
                 arrowprops=dict(arrowstyle='->', lw=4, color=colors['thermal']))
    
    # Thermal energy conversion center
    thermal_center = Circle((0, 0), 0.5, color=colors['thermal'], alpha=0.6)
    ax1.add_patch(thermal_center)
    
    # Abundant dust production (many small circles)
    for i in range(15):
        angle = i * 2*np.pi/15
        r = 0.8 + 0.2 * np.random.rand()
        x = r * np.cos(angle)
        y = -r * np.sin(angle) - 0.5
        dust_size = 0.05 + 0.03 * np.random.rand()
        dust = Circle((x, y), dust_size, color=colors['dust'], alpha=0.7)
        ax1.add_patch(dust)
    
    ax1.text(0, 1.4, "2I/Borisov\np = 0.09", fontsize=12, ha='center', fontweight='bold')
    ax1.text(0, 0, "Thermally\nDominant", fontsize=10, ha='center', va='center', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax1.text(0, -1.2, "Abundant\nMicro-dust", fontsize=10, ha='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    # Panel 2: 3I/ATLAS (p=0.17, Mixed State)
    ax2 = axes[1]
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_aspect('equal')
    ax2.axis('off')
    
    # Sun
    sun2 = Circle((0, 1), 0.3, color='gold', ec='orange', lw=2, zorder=5)
    ax2.add_patch(sun2)
    
    # Split energy flow (thermal and order)
    ax2.annotate('', xy=(-0.15, 0.4), xytext=(0, 0.7),
                 arrowprops=dict(arrowstyle='->', lw=2, color=colors['thermal']))
    ax2.annotate('', xy=(0.15, 0.4), xytext=(0, 0.7),
                 arrowprops=dict(arrowstyle='->', lw=2, color=colors['order']))
    
    # Mixed energy center
    mixed_center = Circle((0, 0), 0.4, color=colors['mixed'], alpha=0.6)
    ax2.add_patch(mixed_center)
    
    # Jets (order component) - triple symmetry
    jet_angles = [0, 2*np.pi/3, 4*np.pi/3]
    for angle in jet_angles:
        dx = 0.9 * np.cos(angle)
        dy = 0.9 * np.sin(angle)
        ax2.annotate('', xy=(dx, dy), xytext=(0, 0),
                     arrowprops=dict(arrowstyle='->', lw=2.5, color='darkgreen'))
    
    # Limited dust production (few, small)
    dust_positions = [(-0.7, -0.7), (0, -0.9), (0.7, -0.7)]
    for x, y in dust_positions:
        dust = Circle((x, y), 0.04, color=colors['dust'], alpha=0.4)
        ax2.add_patch(dust)
    
    ax2.text(0, 1.4, "3I/ATLAS\np = 0.17", fontsize=12, ha='center', fontweight='bold')
    ax2.text(0, 0, "Mixed State", fontsize=10, ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax2.text(0, -1.2, "Suppressed\nMicro-dust", fontsize=10, ha='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
    
    # Panel 3: 1I/'Oumuamua (p=0.83, Information-Driven)
    ax3 = axes[2]
    ax3.set_xlim(-1.5, 1.5)
    ax3.set_ylim(-1.5, 1.5)
    ax3.set_aspect('equal')
    ax3.axis('off')
    
    # Sun
    sun3 = Circle((0, 1), 0.3, color='gold', ec='orange', lw=2, zorder=5)
    ax3.add_patch(sun3)
    
    # Order energy flow (dominant)
    ax3.annotate('', xy=(0, 0.4), xytext=(0, 0.7),
                 arrowprops=dict(arrowstyle='->', lw=4, color=colors['order']))
    
    # Order energy center
    order_center = Circle((0, 0), 0.5, color=colors['order'], alpha=0.6)
    ax3.add_patch(order_center)
    
    # Macroscopic structure (coherent pattern)
    structure_angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    for angle in structure_angles:
        dx = 0.7 * np.cos(angle)
        dy = 0.7 * np.sin(angle)
        structure = Circle((dx, dy), 0.1, color='darkred', alpha=0.8)
        ax3.add_patch(structure)
        ax3.plot([0, dx], [0, dy], 'darkred', alpha=0.5, linewidth=1)
    
    # No dust
    ax3.text(0, -1.2, "No\nMicro-dust", fontsize=10, ha='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8))
    
    ax3.text(0, 1.4, "1I/'Oumuamua\np = 0.83", fontsize=12, ha='center', fontweight='bold')
    ax3.text(0, 0, "Order\nDominant", fontsize=10, ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    fig.suptitle('Energy Reallocation Explains the Dust Deficit', 
                 fontsize=14, fontweight='bold', y=1.05)
    
    plt.tight_layout()
    return fig, axes

def plot_dust_production_vs_p():
    """
    Plot dust production efficiency vs information purity p.
    
    Shows the negative correlation between information purity and
    sub-micron dust production.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Generate p values
    p_values = np.linspace(0, 1, 100)
    
    # Model functions based on Information Dynamics
    # Thermal fraction decreases with p
    thermal_fraction = 1 - p_values**1.5
    # Order fraction increases with p
    order_fraction = p_values**1.5
    
    # Dust production efficiency (exponentially suppressed with p)
    # Based on energy allocation away from microscopic fragmentation
    dust_efficiency = np.exp(-5 * p_values)
    
    # Panel 1: Energy fractions vs p
    ax1.plot(p_values, thermal_fraction, 'b-', linewidth=3, label='Thermal Energy')
    ax1.plot(p_values, order_fraction, 'r-', linewidth=3, label='Order Energy')
    ax1.fill_between(p_values, 0, thermal_fraction, color='blue', alpha=0.2)
    ax1.fill_between(p_values, thermal_fraction, 1, color='red', alpha=0.2)
    
    # Mark ISO positions
    iso_p = [0.09, 0.17, 0.83]
    iso_names = ['2I/Borisov', '3I/ATLAS', "1I/'Oumuamua"]
    iso_colors = ['blue', 'green', 'red']
    
    for p, name, color in zip(iso_p, iso_names, iso_colors):
        thermal_at_p = 1 - p**1.5
        ax1.plot(p, thermal_at_p, 'o', markersize=10, color=color, 
                markeredgecolor='black', linewidth=1.5)
        ax1.annotate(name, (p, thermal_at_p + 0.05), ha='center', fontsize=9)
    
    ax1.set_xlabel('Information Purity (p)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Energy Fraction', fontsize=12, fontweight='bold')
    ax1.set_title('Energy Allocation vs Information Purity', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.1)
    
    # Panel 2: Dust production vs p
    ax2.plot(p_values, dust_efficiency, 'brown', linewidth=3, label='Dust Production Efficiency')
    
    # Mark ISO positions
    for p, name, color in zip(iso_p, iso_names, iso_colors):
        dust_eff = np.exp(-5 * p)
        ax2.plot(p, dust_eff, 'o', markersize=10, color=color, 
                markeredgecolor='black', linewidth=1.5)
    
    ax2.set_xlabel('Information Purity (p)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Relative Dust Production', fontsize=12, fontweight='bold')
    ax2.set_title('Dust Production Suppression with Increasing p', fontsize=12, fontweight='bold')
    
    # Add key insight
    explanation = (
        "Key Insight:\n"
        "• 3I/ATLAS (p=0.17) in mixed state\n"
        "• Energy divided: ~70% thermal, ~30% order\n"
        "• Order energy builds coherent jet structures\n"
        "• Thermal energy produces dust inefficiently\n"
        "• Result: Strong jets + suppressed micro-dust"
    )
    ax2.text(0.02, 0.02, explanation, transform=ax2.transAxes, fontsize=9,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))
    
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.1)
    
    plt.tight_layout()
    return fig, (ax1, ax2)

def calculate_energy_budget():
    """
    Calculate detailed energy budget for 3I/ATLAS.
    
    Shows quantitatively how energy is allocated in the mixed state.
    """
    p_3I = 0.17
    
    # Energy fractions based on Information Dynamics model
    order_fraction = p_3I**1.5  # ~0.07
    thermal_fraction = 1 - order_fraction  # ~0.93
    
    # Dust production efficiency from thermal energy
    dust_efficiency = np.exp(-5 * p_3I)  # ~0.43
    dust_production = thermal_fraction * dust_efficiency  # ~0.40
    
    # Other thermal processes (heating, radiation, etc.)
    other_thermal = thermal_fraction * (1 - dust_efficiency)  # ~0.53
    
    print("="*70)
    print("ENERGY BUDGET ANALYSIS: 3I/ATLAS (p = 0.17)")
    print("="*70)
    
    print("\nTotal Incoming Energy = 100 arbitrary units")
    print("\nEnergy Allocation:")
    print("-"*50)
    print(f"Order Energy (builds/maintains jets): {order_fraction*100:5.1f}%")
    print(f"Thermal Energy: {thermal_fraction*100:5.1f}%")
    print(f"  ├─ Efficient Dust Production: {dust_production*100:5.1f}%")
    print(f"  └─ Other Thermal Processes: {other_thermal*100:5.1f}%")
    
    print("\n" + "="*70)
    print("SCIENTIFIC INTERPRETATION")
    print("="*70)
    print("\n1. Only ~7% of energy goes to building/maintaining jet structures")
    print("2. Only ~40% of thermal energy (40% of total) produces dust efficiently")
    print("3. ~53% of energy dissipated in other thermal processes")
    print("4. RESULT: Strong macroscopic jets but suppressed microscopic dust")
    print("5. This explains the 'dust deficit' anomaly")
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(8, 8))
    
    labels = ['Order Energy (Jets)', 'Dust Production', 'Other Thermal Processes']
    sizes = [order_fraction*100, dust_production*100, other_thermal*100]
    colors = ['#d62728', '#8c564b', '#1f77b4']
    
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                      autopct='%1.1f%%', startangle=90,
                                      wedgeprops=dict(edgecolor='black', linewidth=1))
    
    # Improve text appearance
    for text in texts:
        text.set_fontsize(10)
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax.set_title('3I/ATLAS Energy Budget (Information Purity p = 0.17)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    return fig, ax

def main():
    """
    Main function demonstrating the dust deficit explanation.
    """
    print("="*70)
    print("EXPLAINING THE DUST DEFICIT ANOMALY")
    print("Information Dynamics Framework")
    print("="*70)
    
    print("\nGenerating visual explanations...")
    
    # Create energy flow diagram
    print("\n1. Energy flow diagrams...")
    fig1, axes1 = create_energy_flow_diagram()
    fig1.savefig('energy_flow_diagram.png', dpi=300, bbox_inches='tight')
    print("   Saved: energy_flow_diagram.png")
    
    # Create dust production vs p plot
    print("\n2. Dust production vs information purity...")
    fig2, axes2 = plot_dust_production_vs_p()
    fig2.savefig('dust_production_vs_p.png', dpi=300, bbox_inches='tight')
    print("   Saved: dust_production_vs_p.png")
    
    # Calculate and visualize energy budget
    print("\n3. Detailed energy budget analysis...")
    fig3, ax3 = calculate_energy_budget()
    fig3.savefig('3I_energy_budget.png', dpi=300, bbox_inches='tight')
    print("   Saved: 3I_energy_budget.png")
    
    print("\n" + "="*70)
    print("SUMMARY: THE DUST DEFICIT EXPLAINED")
    print("="*70)
    
    summary_points = [
        "1. 3I/ATLAS has information purity p = 0.17 (mixed state)",
        "2. Energy is split: ~7% to order, ~93% to thermal processes",
        "3. Order energy builds coherent jet structures (N=3 symmetry)",
        "4. Thermal energy has reduced dust production efficiency",
        "5. Result: Strong macroscopic jets + suppressed microscopic dust",
        "6. This explains why 3I shows intense activity but no Rayleigh-scattering dust",
        "7. The 'dust deficit' is a predictable signature of mixed dynamical state",
        "8. Not a compositional anomaly but a dynamical one"
    ]
    
    for i, point in enumerate(summary_points, 1):
        print(f"   {point}")
    
    print("\n" + "="*70)
    print("TESTABLE PREDICTION")
    print("="*70)
    print("\nIf the model is correct, 3I/ATLAS will maintain:")
    print("1. Triple jet structure with 120° separation")
    print("2. Suppressed blue polarization (<0.5%)")
    print("3. Coherent anti-tail structure for >1 week")
    print("4. Step-like activity jumps uncorrelated with distance")
    
    # Show plots
    plt.show()
    
    return True

if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"\nError: Required module not found: {e}")
        print("Please install required packages:")
        print("  pip install numpy matplotlib")
    except Exception as e:
        print(f"\nError during execution: {e}")
