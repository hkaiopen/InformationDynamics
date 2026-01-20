#!/usr/bin/env python3
"""
predict_jan22_alignment.py

Script to generate specific, testable predictions for 3I/ATLAS during its
special alignment on January 22, 2026, when it will be in opposition
(Sun-Earth-3I alignment).

Based on the Information Dynamics framework with calibrated parameters:
p=0.17, gamma=3.8e-5 s-1, epsilon=0.078e-4, omega=1.08e-4 rad/s, N=3.

Author: Kai Huang, Hongkui Liu
Date: January 20, 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')  # For headless environments

# ============================================================================
# MODEL PARAMETERS FOR 3I/ATLAS (from paper Table 1)
# ============================================================================

# Calibrated parameters from Information Dynamics model (Table 1)
gamma = 3.8e-5          # Linear dissipation rate [s-1]
gamma_err = 0.5e-5      # Uncertainty
epsilon = 0.078e-4      # Nonlinear self-interaction strength [s-1]
epsilon_err = 0.007e-4  # Uncertainty
omega = 1.08e-4         # Characteristic frequency [rad/s]
omega_err = 0.02e-4     # Uncertainty
N = 3                   # Symmetry parameter (triple jets)
p = 0.17                # Information purity

# Derived parameters
T_period = 2 * np.pi / omega  # Characteristic period [s]
T_period_days = T_period / (24 * 3600)  # Convert to days

# Observational geometry for Jan 22, 2026
alignment_date = datetime(2026, 1, 22)
heliocentric_distance = 1.8  # AU (estimated from paper)
geocentric_distance = 0.8    # AU (estimated from paper)
phase_angle = 0.0            # Opposition phase angle [deg]

# ============================================================================
# QUANTITATIVE PREDICTIONS WITH MONTE CARLO UNCERTAINTY
# ============================================================================

def calculate_recoil_effect(n_sim=1000):
    """
    Calculate predicted non-gravitational acceleration with Monte Carlo uncertainty.
    """
    # Base non-gravitational acceleration (from thermal + nonlinear terms)
    A_ng_base = 1.2e-7  # cm/s^2 (based on preliminary estimates)
    
    # Monte Carlo simulation to estimate uncertainty
    A_ng_samples = []
    for _ in range(n_sim):
        # Sample parameters with uncertainty
        gamma_sim = np.random.normal(gamma, gamma_err)
        epsilon_sim = np.random.normal(epsilon, epsilon_err)
        
        # Calculate information purity for this sample
        p_sim = epsilon_sim / (gamma_sim + epsilon_sim)
        
        # Enhancement factor during opposition
        enhancement_factor = 1.5 + 0.3 * p_sim
        
        # Calculate acceleration for this sample
        A_ng_sample = A_ng_base * enhancement_factor
        A_ng_samples.append(A_ng_sample)
    
    # Calculate statistics
    A_ng_mean = np.mean(A_ng_samples)
    A_ng_std = np.std(A_ng_samples)
    A_ng_median = np.median(A_ng_samples)
    
    # Direction: primarily radial (sunward/anti-sunward)
    # During opposition, we expect anti-sunward acceleration
    direction = "anti-sunward"
    
    return {
        "A_ng_base": A_ng_base,
        "A_ng_opposition": A_ng_mean,
        "A_ng_median": A_ng_median,
        "A_ng_std": A_ng_std,
        "A_ng_2sigma": 2.0 * A_ng_std,  # 95% confidence interval
        "enhancement_factor": f"{1.5 + 0.3 * p:.2f} +/- {0.3 * (epsilon_err/(gamma+epsilon)):.2f}",
        "direction": direction,
        "uncertainty_notes": f"Monte Carlo simulation with n={n_sim} samples",
    }

def calculate_polarization():
    """
    Calculate predicted polarization properties.
    """
    # Updated formula: polarization degree depends on (1-p)
    polarization_degree = 0.1 + 0.3 * (1 - p)  # [%] Total polarization
    
    # Blue polarization especially suppressed
    polarization_blue = polarization_degree * (1 - p)  # Additional suppression in blue
    
    # Red polarization (less suppressed)
    polarization_red = polarization_degree * 0.9
    
    # Polarization angle modulation
    modulation_amplitude = 0.15  # +/-0.15% modulation
    
    return {
        "polarization_degree": polarization_degree,
        "polarization_blue": polarization_blue,
        "polarization_red": polarization_red,
        "modulation_amplitude": modulation_amplitude,
        "phase_angle_dependence": "unusually flat (0-30 deg)",
        "uncertainty": "+/-0.1% (instrumental + model)",
    }

def calculate_jet_properties():
    """
    Calculate predicted jet properties during opposition.
    """
    # Jet stability parameter from nonlinear term
    coherence_time = 5 / p  # days (epsilon-driven stability)
    
    # Conservative estimate for observational planning
    coherence_time_conservative = coherence_time / 4
    
    # Expected jet rotation period (related to omega)
    rotation_period = T_period_days  # ~6.7 days
    
    # Jet intensity modulation (due to N=3 symmetry)
    modulation_depth = 0.4  # 40% peak-to-trough
    
    # Predicted wobble characteristics
    predicted_wobble = "Synchronized across all three jets, amplitude <5 deg"
    
    return {
        "coherence_time_days": coherence_time_conservative,
        "rotation_period_days": rotation_period,
        "modulation_depth": modulation_depth,
        "jet_separation_angle": 120,  # degrees
        "predicted_wobble": predicted_wobble,
    }

def simulate_activity_jumps(t_start, t_end, dt=3600, n_jumps=3):
    """
    Simulate discrete activity jumps (Prediction 3) with noise η(t).
    """
    # Convert to timestamps
    t = np.arange(t_start.timestamp(), t_end.timestamp(), dt)
    
    # Base activity: oscillation with characteristic frequency ω
    activity_base = np.sin(omega * (t - t[0])) * np.exp(-gamma * (t - t[0]))
    
    # Add noise-induced jumps
    activity = activity_base.copy()
    
    total_seconds = (t_end - t_start).total_seconds()
    
    # Determine jump times (randomly distributed，避免在时间窗口的开始和结束附近)
    jump_times = np.random.uniform(t_start.timestamp() + 0.2 * total_seconds,
                                  t_end.timestamp() - 0.2 * total_seconds,
                                  n_jumps)
    
    jump_sizes = []
    for jt in jump_times:
        idx = np.argmin(np.abs(t - jt))
        # Jump size: larger at lower p (more noise-sensitive)
        jump_size = np.random.normal(0.2, 0.05) * (1 - p)
        activity[idx:] += jump_size
        jump_sizes.append(jump_size)
    
    # Normalize for plotting
    activity_normalized = activity / np.max(np.abs(activity))
    
    # Plot time-series
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Convert timestamps back to datetime for plotting
    times = [datetime.fromtimestamp(ti) for ti in t]
    
    ax.plot(times, activity_normalized, 'b-', linewidth=1.5, label='Modeled activity')
    ax.set_xlabel('Time (UTC)', fontsize=12)
    ax.set_ylabel('Normalized Activity Level', fontsize=12)
    ax.set_title(f'Simulated Activity with Step-like Jumps\n{alignment_date.strftime("%b %d, %Y")} ± 3 days', 
                fontsize=14, fontweight='bold')
    
    # Mark jump times
    for i, jt in enumerate(jump_times):
        jump_time = datetime.fromtimestamp(jt)
        ax.axvline(jump_time, color='r', linestyle='--', alpha=0.7)
        ax.text(jump_time, 0.95, f'Jump {i+1}', rotation=90, 
               verticalalignment='top', fontsize=10)
    
    # Mark opposition time
    ax.axvline(alignment_date, color='orange', linestyle='-', linewidth=2, label='Opposition')
    ax.axvspan(alignment_date - timedelta(hours=12), 
               alignment_date + timedelta(hours=12), 
               alpha=0.1, color='orange', label='Opposition window (±12h)')
    
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    fig.savefig('activity_jumps_simulation.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        "plot_file": 'activity_jumps_simulation.png',
        "n_jumps": n_jumps,
        "jump_times": [datetime.fromtimestamp(jt) for jt in jump_times],
        "jump_sizes": jump_sizes,
        "prediction_note": f"Expected {n_jumps} step-like changes in brightness/acceleration",
    }

def generate_geometry_plot(save_path="alignment_geometry.png"):
    """
    Create schematic of the Sun-Earth-3I alignment on Jan 22, 2026.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel 1: Top-down view of orbital plane
    angles = np.linspace(0, 2*np.pi, 100)
    
    # Earth orbit (simplified)
    ax1.plot(np.cos(angles), np.sin(angles), 'b--', alpha=0.3, label="Earth orbit")
    
    # Positions
    ax1.plot(0, 0, 'yo', markersize=20, label="Sun")
    ax1.plot(1, 0, 'bo', markersize=12, label="Earth")
    ax1.plot(heliocentric_distance, 0, 'go', markersize=10, label="3I/ATLAS")
    
    # Connection lines
    ax1.plot([0, 1], [0, 0], 'k-', alpha=0.5, linewidth=1)
    ax1.plot([0, heliocentric_distance], [0, 0], 'g-', alpha=0.5, linewidth=1)
    
    # Jets (N=3 symmetry)
    jet_length = 0.15 * heliocentric_distance
    jet_angles = np.linspace(0, 2*np.pi, 4)[:3]
    for angle in jet_angles:
        dx = jet_length * np.cos(angle)
        dy = jet_length * np.sin(angle)
        ax1.arrow(heliocentric_distance, 0, dx, dy, 
                 head_width=0.03, head_length=0.05, 
                 fc='orange', ec='orange', alpha=0.8, linewidth=1.5)
    
    ax1.set_aspect('equal')
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_xlabel("X [AU]", fontsize=11)
    ax1.set_ylabel("Y [AU]", fontsize=11)
    ax1.set_title("Orbital Geometry (Top View)", fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Observational geometry from Earth
    ax2.plot([-1, 1], [0, 0], 'k-', alpha=0.5, linewidth=1, label="Scattering plane")
    ax2.plot(0, 0, 'bo', markersize=12, label="Earth observer")
    
    # Sun direction
    ax2.arrow(-0.8, 0, 0.6, 0, head_width=0.05, head_length=0.08,
             fc='yellow', ec='orange', label="Sun direction", linewidth=1.5)
    
    # 3I/ATLAS with jets
    object_distance = 0.5  # Normalized for plotting
    ax2.plot(object_distance, 0, 'go', markersize=10, label="3I/ATLAS")
    
    # Jets in the plane of the sky (N=3 symmetry)
    jet_angles_sky = [-30, 0, 30]  # degrees
    for angle in jet_angles_sky:
        rad = np.radians(angle)
        dx = 0.25 * np.cos(rad)
        dy = 0.25 * np.sin(rad)
        ax2.arrow(object_distance, 0, dx, dy, head_width=0.03, head_length=0.05,
                 fc='orange', ec='orange', alpha=0.8, linewidth=1.5)
    
    ax2.set_aspect('equal')
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-0.6, 0.6)
    ax2.set_xlabel("Sky plane X", fontsize=11)
    ax2.set_ylabel("Sky plane Y", fontsize=11)
    ax2.set_title(f"Earth View: Opposition (α={phase_angle}°)", 
                 fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Add informational text
    info_text = f"""Geometry Summary:
• Heliocentric distance: {heliocentric_distance} AU
• Geocentric distance: {geocentric_distance} AU
• Phase angle: {phase_angle}°
• Alignment: Sun-3I-Earth = 0°
• Model parameters from Huang & Liu (2026)"""
    
    fig.text(0.02, 0.02, info_text, fontsize=9, 
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.2))
    
    plt.suptitle(f"3I/ATLAS Opposition Geometry: {alignment_date.strftime('%Y-%m-%d')}", 
                fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Geometry plot saved to: {save_path}")
    return save_path

def main():
    """Generate and display all predictions."""
    print("=" * 70)
    print("INFORMATION DYNAMICS: SPECIFIC PREDICTIONS FOR 3I/ATLAS")
    print(f"Special Alignment Event: {alignment_date.strftime('%Y-%m-%d')}")
    print("=" * 70)
    
    print("\n1. MODEL PARAMETERS (from Table 1 with uncertainties):")
    print(f"   Information purity (p): {p}")
    print(f"   Linear dissipation (γ): {gamma:.2e} +/- {gamma_err:.1e} s-1")
    print(f"   Nonlinear strength (ε): {epsilon:.2e} +/- {epsilon_err:.1e}")
    print(f"   Characteristic frequency (ω): {omega:.2e} +/- {omega_err:.1e} rad/s")
    print(f"   Symmetry parameter (N): {N} (triple jets)")
    print(f"   Characteristic period: {T_period_days:.1f} days")
    
    print("\n2. OBSERVATIONAL GEOMETRY:")
    print(f"   Heliocentric distance: {heliocentric_distance} AU")
    print(f"   Geocentric distance: {geocentric_distance} AU")
    print(f"   Phase angle: {phase_angle} deg (opposition)")
    print(f"   Sun-3I-Earth angle: 0 deg (perfect alignment)")
    
    print("\n3. QUANTITATIVE PREDICTIONS:")
    
    # Recoil effect with Monte Carlo uncertainty
    recoil = calculate_recoil_effect(n_sim=1000)
    print(f"\n   A. Non-gravitational acceleration (recoil effect):")
    print(f"      Base value: {recoil['A_ng_base']:.2e} cm/s^2")
    print(f"      Opposition enhanced: {recoil['A_ng_opposition']:.2e} cm/s^2")
    print(f"      Median (MC): {recoil['A_ng_median']:.2e} cm/s^2")
    print(f"      Uncertainty (1σ): +/- {recoil['A_ng_std']:.2e} cm/s^2")
    print(f"      95% CI (2σ): +/- {recoil['A_ng_2sigma']:.2e} cm/s^2")
    print(f"      Enhancement factor: {recoil['enhancement_factor']}x")
    print(f"      Direction: {recoil['direction']}")
    
    # Polarization
    pol = calculate_polarization()
    print(f"\n   B. Polarization properties:")
    print(f"      Total polarization: {pol['polarization_degree']:.3%}")
    print(f"      Blue (0.45µm): {pol['polarization_blue']:.3%} (suppressed)")
    print(f"      Red (0.65µm): {pol['polarization_red']:.3%}")
    print(f"      Modulation amplitude: +/-{pol['modulation_amplitude']:.3%}")
    print(f"      Phase angle dependence: {pol['phase_angle_dependence']}")
    
    # Jet properties
    jets = calculate_jet_properties()
    print(f"\n   C. Jet structure and dynamics:")
    print(f"      Coherence time: >{jets['coherence_time_days']:.1f} days")
    print(f"      Rotation period: {jets['rotation_period_days']:.1f} days")
    print(f"      Intensity modulation: {jets['modulation_depth']:.1%}")
    print(f"      Jet separation: {jets['jet_separation_angle']} deg")
    print(f"      Predicted wobble: {jets['predicted_wobble']}")
    
    # Activity jumps simulation
    print(f"\n   D. Activity jumps (Prediction 3 simulation):")
    t_start = alignment_date - timedelta(days=3)
    t_end = alignment_date + timedelta(days=3)
    jumps = simulate_activity_jumps(t_start, t_end, n_jumps=3)
    print(f"      Expected jumps: {jumps['n_jumps']}")
    for i, (jump_time, jump_size) in enumerate(zip(jumps['jump_times'], jumps['jump_sizes'])):
        print(f"      Jump {i+1}: {jump_time.strftime('%Y-%m-%d %H:%M UTC')}, size: {jump_size:.3f}")
    print(f"      Plot saved: {jumps['plot_file']}")
    
    print("\n4. MODEL ASSUMPTIONS:")
    print("   1. The CGLE framework captures ISO dynamics")
    print("   2. Parameters stable over observational timescale")
    print("   3. N=3 jet symmetry persists through opposition")
    print("   4. Dust deficit continues (no Rayleigh scatterers)")
    
    print("\n5. FALSIFIABILITY TESTS:")
    print("   1. Strong blue polarization (>1%) indicating sub-micron dust")
    print("   2. Isotropic coma without jet structure")
    print("   3. Non-gravitational acceleration <0.5e-7 cm/s^2")
    print("   4. No activity jumps observed")
    
    # Generate plots
    geometry_plot = generate_geometry_plot()
    
    print("\n" + "=" * 70)
    print("RECOMMENDED OBSERVATIONS (Jan 19-25, 2026):")
    print("1. Polarimetry: B, V, R bands to test dust deficit")
    print("2. Astrometry: High-cadence (<6h) to measure acceleration")
    print("3. Imaging: Monitor jet structure (daily)")
    print("4. Photometry: High-cadence to detect activity jumps")
    print("=" * 70)
    
    print(f"\nPredictions generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Reference: Huang & Liu (2026), Information Dynamics framework")
    print(f"Code: https://github.com/hkaiopen/InformationDynamics")
    
    return {
        "recoil": recoil,
        "polarization": pol,
        "jets": jets,
        "jumps": jumps,
        "geometry_plot": geometry_plot,
    }

if __name__ == "__main__":
    results = main()
    
    # Save numerical predictions
    with open("predictions_jan22_2026.txt", "w", encoding="utf-8") as f:
        f.write(f"3I/ATLAS Opposition Predictions - {alignment_date.strftime('%Y-%m-%d')}\n")
        f.write("="*60 + "\n\n")
        f.write("MODEL PARAMETERS:\n")
        f.write(f"  Information purity (p): {p}\n")
        f.write(f"  Gamma (γ): {gamma:.2e} +/- {gamma_err:.1e} s-1\n")
        f.write(f"  Epsilon (ε): {epsilon:.2e} +/- {epsilon_err:.1e}\n")
        f.write(f"  Omega (ω): {omega:.2e} +/- {omega_err:.1e} rad/s\n")
        f.write(f"  Symmetry (N): {N}\n\n")
        
        f.write("QUANTITATIVE PREDICTIONS:\n")
        f.write(f"  1. Non-gravitational acceleration: {results['recoil']['A_ng_opposition']:.2e} cm/s^2\n")
        f.write(f"     (95% CI: +/- {results['recoil']['A_ng_2sigma']:.2e} cm/s^2)\n")
        f.write(f"  2. Polarization degree: {results['polarization']['polarization_degree']:.3%}\n")
        f.write(f"  3. Polarization (blue): {results['polarization']['polarization_blue']:.3%}\n")
        f.write(f"  4. Jet coherence time: {results['jets']['coherence_time_days']:.1f} days\n")
        f.write(f"  5. Expected activity jumps: {results['jumps']['n_jumps']}\n\n")
        
        f.write("GEOMETRY:\n")
        f.write(f"  Heliocentric distance: {heliocentric_distance} AU\n")
        f.write(f"  Geocentric distance: {geocentric_distance} AU\n")
        f.write(f"  Phase angle: {phase_angle} deg\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Model: Information Dynamics (Huang & Liu, 2026)\n")
    
    print(f"\nNumerical predictions saved to: predictions_jan22_2026.txt")
