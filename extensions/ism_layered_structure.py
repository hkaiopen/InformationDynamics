"""
ism_layered_structure_plot.py

Generates Figure: ISM Aging Layered Structure (Alteration vs Fidelity)
"""
import numpy as np
import matplotlib.pyplot as plt
from interstellar_medium_aging import ISMAgingModel

def plot_ism_layered_structure(save_path='ism_layered_structure.png'):
    ism = ISMAgingModel(radius_km=1.3, wandering_time_Gyr=10.0)
    layers_data = ism.layered_structure(n_layers=20)
    layers = layers_data['layers']

    depths = [l['depth_m'] for l in layers]
    alterations = [l['alteration'] for l in layers]
    fidelities = [l['fidelity'] for l in layers]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot alteration (left axis)
    color = 'tab:red'
    ax1.set_xlabel('Depth from Surface (m)')
    ax1.set_ylabel('Alteration Fraction', color=color)
    ax1.plot(depths, alterations, color=color, linewidth=2, label='Alteration')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xlim(0, 1.2)  # Zoom into the first 1.2 meters
    ax1.grid(True, alpha=0.3)

    # Plot fidelity (right axis)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Information Fidelity', color=color)
    ax2.plot(depths, fidelities, color=color, linewidth=2, linestyle='--', label='Fidelity')
    ax2.tick_params(axis='y', labelcolor=color)

    # Highlight the zone
    ax1.axvline(x=ism.alteration_depth(), color='orange', linestyle=':', 
                label=f'Alteration Depth ({ism.alteration_depth():.2f} m)')
    ax1.text(ism.alteration_depth()+0.05, 0.5, 
             f'Surface Apex: {ism.alteration_depth():.2f} m\nVolume Preserved: {ism.preservation_fraction()*100:.2f}%',
             fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    ax1.legend(loc='center left')
    ax2.legend(loc='center right')
    plt.title('ISM Aging: Surface Alteration Gradient', fontsize=14, fontweight='bold')

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"ISM layered structure plot saved to: {save_path}")

if __name__ == "__main__":
    plot_ism_layered_structure()