"""
thermal_history.py
"""
import numpy as np
import matplotlib.pyplot as plt
from thermal_history_evolution import ThermalHistoryModel

def plot_log_thermal_history(history, save_path='thermal_history_log.png'):
    t_myr = history['t_years'] * 1e6
    t_myr = np.clip(t_myr, 1e-3, None)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Thermal History Evolution of 3I/ATLAS (Logarithmic Time)', fontsize=16, fontweight='bold')

    axs[0,0].semilogx(t_myr, history['T_K'], 'r-', linewidth=2)
    axs[0,0].set_title('Temperature Evolution')
    axs[0,0].set_ylabel('Temperature (K)')
    axs[0,0].grid(True, alpha=0.3, which='both')

    axs[0,1].semilogx(t_myr, history['F_thermal'], 'b-', linewidth=2)
    axs[0,1].set_title('Thermal Driving (F_thermal)')
    axs[0,1].set_ylabel('F_thermal')
    axs[0,1].grid(True, alpha=0.3, which='both')

    axs[1,0].semilogx(t_myr, history['epsilon'], 'g-', linewidth=2)
    axs[1,0].set_title('Nonlinear Self-Interaction (epsilon)')
    axs[1,0].set_ylabel(r'\(\varepsilon\) (s\(^{-1}\))')
    axs[1,0].set_yscale('log')
    axs[1,0].grid(True, alpha=0.3, which='both')

    axs[1,1].semilogx(t_myr, history['p'], 'purple', linewidth=2)
    axs[1,1].axhspan(0.10, 0.25, alpha=0.2, color='orange', label='Mixed State Range')
    axs[1,1].axhline(y=0.17, color='red', linestyle='--', label='Initial p = 0.17')
    peri_idx = np.argmin(history['r_AU'])
    axs[1,1].scatter(t_myr[peri_idx], history['p'][peri_idx], color='blue', s=50, zorder=5)
    axs[1,1].annotate(f'Perihelion p = {history["p"][peri_idx]:.3f}',
                      (t_myr[peri_idx], history['p'][peri_idx]),
                      xytext=(10, -15), textcoords='offset points', fontsize=9)
    axs[1,1].set_title('Information Purity (p)')
    axs[1,1].set_xlabel('Time (Myr)')
    axs[1,1].set_ylabel('p')
    axs[1,1].legend()
    axs[1,1].grid(True, alpha=0.3, which='both')
    axs[1,1].set_xlim(1e-3, 1e4)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Log thermal history plot saved to: {save_path}")

if __name__ == "__main__":
    thm = ThermalHistoryModel()
    history = thm.full_history(t_form_Gyr=10.0)
    plot_log_thermal_history(history)
