"""
thermal_history_plot.py

Generates Figure: Thermal History Evolution (T, F, epsilon, p)
"""
import numpy as np
import matplotlib.pyplot as plt
from thermal_history_evolution import ThermalHistoryModel

def plot_thermal_history(history, save_path='thermal_history.png'):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Thermal History Evolution of 3I/ATLAS', fontsize=16, fontweight='bold')

    # 1. Temperature (K)
    axs[0,0].plot(history['t_years'], history['T_K'], 'r-', linewidth=2)
    axs[0,0].set_title('Temperature Evolution')
    axs[0,0].set_ylabel('Temperature (K)')
    axs[0,0].grid(True, alpha=0.3)

    # 2. F_thermal
    axs[0,1].plot(history['t_years'], history['F_thermal'], 'b-', linewidth=2)
    axs[0,1].set_title('Thermal Driving (F_thermal)')
    axs[0,1].set_ylabel('F_thermal')
    axs[0,1].grid(True, alpha=0.3)

    # 3. Epsilon
    axs[1,0].plot(history['t_years'], history['epsilon'], 'g-', linewidth=2)
    axs[1,0].set_title('Nonlinear Self-Interaction (epsilon)')
    axs[1,0].set_ylabel(r'$\varepsilon$ (s$^{-1}$)')
    axs[1,0].set_yscale('log')
    axs[1,0].grid(True, alpha=0.3)

    # 4. Information Purity p
    axs[1,1].plot(history['t_years'], history['p'], 'purple', linewidth=2)
    axs[1,1].axhspan(0.10, 0.25, alpha=0.2, color='orange', label='Mixed State Range')
    axs[1,1].axhline(y=0.17, color='red', linestyle='--', label='Initial p = 0.17')
    axs[1,1].set_title('Information Purity (p)')
    axs[1,1].set_xlabel('Time (Myr)')
    axs[1,1].set_ylabel('p')
    axs[1,1].legend()
    axs[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Thermal history plot saved to: {save_path}")

if __name__ == "__main__":
    thm = ThermalHistoryModel()
    history = thm.full_history(t_form_Gyr=10.0)
    plot_thermal_history(history)