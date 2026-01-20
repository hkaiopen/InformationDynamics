"""
generate_predictions.py

Information Dynamics: A Unified Predictive Framework for Interstellar Objects
Three Testable Predictions for 3I/ATLAS

This script generates visualizations and detailed descriptions of the three
testable predictions for the interstellar object 3I/ATLAS based on the
Information Dynamics framework.

Author: Kai Huang, Hongkui Liu
Date: January 20, 2026
Paper: https://github.com/hkaiopen/InformationDynamics
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from matplotlib.patches import Rectangle, Polygon, Circle
import pandas as pd

# ============================================================================
# CONSTANTS AND PARAMETERS (from Table 1 in the paper)
# ============================================================================

# Parameters for 3I/ATLAS (best-fit values)
PARAMS_3I = {
    'γ': 3.8e-5,          # linear dissipation rate (s⁻¹)
    'ω': 1.08e-4,         # characteristic frequency (rad/s)
    'ε': 0.078e-4,        # nonlinear self-interaction strength (s⁻¹)
    'F_thermal': 0.12,    # thermal outgassing forcing (dimensionless)
    'N': 3,               # symmetry parameter (triple jets)
    'p': 0.17             # information purity
}

# Derived quantities
OMEGA_PERIOD_DAYS = (2 * np.pi) / PARAMS_3I['ω'] / (24 * 3600)  # ~6.73 days

# Observational timeline
TIMELINE = {
    'current': datetime(2026, 1, 20),
    'perihelion': datetime(2025, 6, 15),
    'unobservable': datetime(2026, 7, 1)
}

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_observation_timeline():
    """Create timeline showing critical observation window for 3I/ATLAS."""
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Key dates
    dates = [TIMELINE['perihelion'], TIMELINE['current'], TIMELINE['unobservable']]
    events = ['Perihelion\n(Jun 2025)', 'Current\n(Jan 2026)', 'Unobservable\n(Jul 2026)']
    
    for date, event in zip(dates, events):
        ax.plot(date, 0, 'o', markersize=12, 
                color='red' if date == TIMELINE['current'] else 'blue')
        ax.text(date, 0.15, event, ha='center', fontsize=10, fontweight='bold')
    
    # Prediction testing windows
    prediction_windows = [
        (datetime(2025, 8, 1), datetime(2026, 2, 1), 'Prediction 1\nAnti-tail'),
        (datetime(2025, 9, 1), datetime(2026, 3, 1), 'Prediction 2\nWobble'),
        (datetime(2025, 10, 1), datetime(2026, 4, 1), 'Prediction 3\nJumps')
    ]
    
    colors = ['#1f77b4', '#2ca02c', '#d62728']
    for i, (start, end, label) in enumerate(prediction_windows):
        width = (end - start).days
        rect = Rectangle((start, -0.2), timedelta(days=width), 0.4,
                        facecolor=colors[i], alpha=0.3, edgecolor=colors[i], linewidth=2)
        ax.add_patch(rect)
        
        mid = start + (end - start) / 2
        ax.text(mid, -0.3, label, ha='center', fontsize=9, fontweight='bold')
    
    # Critical window (current to May 2026)
    ax.axvspan(TIMELINE['current'], datetime(2026, 5, 1), alpha=0.1, color='gold',
               label='Critical Testing Window')
    
    ax.set_xlim(datetime(2025, 5, 1), datetime(2026, 8, 1))
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_title('Observation Timeline for 3I/ATLAS\nCritical: Unobservable by mid-2026', 
                 fontsize=14, fontweight='bold', pad=20)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend(loc='upper left')
    
    # Urgency note
    urgency = (
        "URGENT: 3I/ATLAS becomes unobservable by mid-2026\n"
        "Observations must be completed before July 2026\n"
        "Verification of any 2 predictions supports the model"
    )
    ax.text(0.02, 0.95, urgency, transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.9),
            verticalalignment='top')
    
    plt.tight_layout()
    return fig


def visualize_prediction_1():
    """Visualize Prediction 1: Anti-tail coherence."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel 1: Conceptual diagram
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-1, 3)
    ax1.set_aspect('equal')
    ax1.axis('off')
    
    # Sun
    sun = Circle((0, 2.5), 0.3, color='gold', ec='orange', lw=2)
    ax1.add_patch(sun)
    
    # Comet
    comet = Circle((0, 0), 0.15, color='gray', alpha=0.8)
    ax1.add_patch(comet)
    ax1.text(0, 0, '3I', fontsize=10, ha='center', va='center', 
             fontweight='bold', color='white')
    
    # Anti-tail (narrow, coherent)
    tail_points = [(-0.1, 0), (-0.1, 1.5), (0.1, 1.5), (0.1, 0)]
    tail = Polygon(tail_points, closed=True, color='blue', alpha=0.6)
    ax1.add_patch(tail)
    
    # Coherence "field lines"
    for y in [0.3, 0.6, 0.9, 1.2]:
        ax1.plot([-0.08, 0.08], [y, y], 'b-', linewidth=2, alpha=0.7)
    
    # Solar wind (being resisted)
    for x in [-1.0, -0.5, 0, 0.5, 1.0]:
        ax1.annotate('', xy=(x, 0.8), xytext=(x, 2.2),
                     arrowprops=dict(arrowstyle='->', lw=1, color='orange', alpha=0.5))
    
    ax1.text(0, 2.8, 'Sun', ha='center', fontsize=11, fontweight='bold')
    ax1.text(0, 0.8, 'Coherent\nAnti-tail', ha='center', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    ax1.set_title('Prediction 1: Anti-tail Coherence\nε-driven structure resists dispersion', 
                  fontsize=12, fontweight='bold')
    
    # Panel 2: Quantitative prediction
    ax2.set_title('Predicted Tail Width vs Time', fontsize=12, fontweight='bold')
    
    time_days = np.linspace(0, 60, 100)
    
    # Typical comet (disperses)
    typical_width = 0.5 + 0.025 * time_days
    
    # 3I/ATLAS prediction (maintains coherence)
    predicted_width = 0.15 * np.ones_like(time_days)
    predicted_width += 0.03 * np.sin(2 * np.pi * time_days / 14)  # small oscillations
    
    ax2.plot(time_days, typical_width, 'r-', linewidth=3, 
             label='Typical Comet (disperses)')
    ax2.plot(time_days, predicted_width, 'b-', linewidth=3,
             label='3I/ATLAS Prediction (coherent)')
    
    ax2.fill_between(time_days, predicted_width-0.02, predicted_width+0.02, 
                     color='blue', alpha=0.2)
    
    ax2.set_xlabel('Time (days)', fontsize=11)
    ax2.set_ylabel('Tail Width (arbitrary units)', fontsize=11)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Observation windows
    ax2.axvspan(7, 21, alpha=0.2, color='green', label='HST Cycle 1')
    ax2.axvspan(35, 49, alpha=0.2, color='orange', label='HST Cycle 2')
    ax2.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    return fig


def visualize_prediction_2():
    """Visualize Prediction 2: Synchronized wobble."""
    period = OMEGA_PERIOD_DAYS
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel 1: Wobble geometry
    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-1.2, 1.2)
    ax1.set_aspect('equal')
    
    # Nucleus
    nucleus = Circle((0, 0), 0.1, color='gray', alpha=0.8)
    ax1.add_patch(nucleus)
    
    # Jet positions at two phases
    base_angles = np.array([0, 2*np.pi/3, 4*np.pi/3])
    
    # Phase 1
    wobble1 = 0.15
    angles1 = base_angles + wobble1
    
    # Phase 2 (half period later)
    wobble2 = -0.15
    angles2 = base_angles + wobble2
    
    colors = ['red', 'green', 'blue']
    
    for i, (angle1, angle2, color) in enumerate(zip(angles1, angles2, colors)):
        # Phase 1 (solid)
        x1 = 0.8 * np.cos(angle1)
        y1 = 0.8 * np.sin(angle1)
        ax1.plot([0, x1], [0, y1], color=color, linewidth=2)
        
        # Phase 2 (dashed)
        x2 = 0.8 * np.cos(angle2)
        y2 = 0.8 * np.sin(angle2)
        ax1.plot([0, x2], [0, y2], color=color, linewidth=2, linestyle='--')
        
        # Wobble arc
        arc_angles = np.linspace(angle1, angle2, 20)
        arc_x = 0.3 * np.cos(arc_angles)
        arc_y = 0.3 * np.sin(arc_angles)
        ax1.plot(arc_x, arc_y, 'k:', alpha=0.5, linewidth=1)
    
    ax1.set_xlabel('X', fontsize=11)
    ax1.set_ylabel('Y', fontsize=11)
    ax1.set_title('Synchronized Wobble\nAll jets move in phase', 
                  fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Angle markers
    for angle in [0, 120, 240]:
        rad = angle * np.pi/180
        x = 0.9 * np.cos(rad)
        y = 0.9 * np.sin(rad)
        ax1.text(x, y, f'{angle}°', ha='center', va='center', fontsize=8,
                 bbox=dict(boxstyle="circle,pad=0.2", facecolor="white", alpha=0.8))
    
    # Panel 2: Predicted wobble period
    ax2.set_title(f'Predicted Wobble Period: {period:.2f} days', 
                  fontsize=12, fontweight='bold')
    
    time_days = np.linspace(0, 100, 500)
    time_seconds = time_days * 86400
    
    # Wobble angle (sinusoidal)
    wobble_amplitude = 0.2  # radians
    wobble_angle = wobble_amplitude * np.sin(PARAMS_3I['ω'] * time_seconds)
    
    ax2.plot(time_days, np.degrees(wobble_angle), 'b-', linewidth=2)
    
    # Mark HST observation cycles
    hst_cycles = [10, 30, 50, 70, 90]
    for cycle in hst_cycles:
        ax2.axvline(x=cycle, color='red', linestyle=':', alpha=0.5)
        if cycle == 30:
            ax2.text(cycle, ax2.get_ylim()[1]*0.8, 'HST\nCycle', 
                     ha='center', fontsize=8,
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    ax2.set_xlabel('Time (days)', fontsize=11)
    ax2.set_ylabel('Wobble Angle (degrees)', fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Detection criteria
    criteria = (
        "Detection Criteria:\n"
        "• Measure jet directions over ≥2 cycles\n"
        "• Check for in-phase motion\n"
        "• Verify 120° separation maintained\n"
        f"• Look for {period:.1f}-day period"
    )
    ax2.text(0.02, 0.02, criteria, transform=ax2.transAxes, fontsize=9,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.9))
    
    plt.tight_layout()
    return fig, period


def visualize_prediction_3():
    """Visualize Prediction 3: Discrete activity jumps."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Generate simulated data
    np.random.seed(42)
    time_days = np.linspace(0, 200, 1000)
    
    # Heliocentric distance
    r_helio = 1.0 + 0.5 * np.abs(time_days - 100) / 100
    
    # Base brightness (1/r² law)
    base_brightness = 1.0 / r_helio**2
    
    # Add typical variations
    typical_variations = 0.1 * np.sin(2 * np.pi * time_days / 25)
    
    # For 3I/ATLAS: Add step-like changes (uncorrelated with distance)
    step_times = [45, 92, 138, 185]  # Not correlated with perihelion (~day 100)
    step_sizes = [0.35, -0.25, 0.45, -0.35]
    
    predicted_brightness = base_brightness + typical_variations
    
    for step_time, step_size in zip(step_times, step_sizes):
        idx = np.argmin(np.abs(time_days - step_time))
        predicted_brightness[idx:] += step_size
    
    # Add measurement noise
    noise = 0.05 * np.random.randn(len(time_days))
    measured = predicted_brightness + noise
    
    # Panel 1: Brightness comparison
    ax1.plot(time_days, base_brightness, 'k:', linewidth=1, alpha=0.5, 
             label='1/r² baseline')
    ax1.plot(time_days, measured, 'b-', linewidth=2, 
             label='3I/ATLAS Prediction (with jumps)')
    
    # Mark jumps
    for step_time, step_size in zip(step_times, step_sizes):
        ax1.axvline(x=step_time, color='red', linestyle=':', alpha=0.7, linewidth=1)
        ax1.text(step_time, ax1.get_ylim()[1]*0.95, f'{step_size:+.2f}', 
                 ha='center', fontsize=8, color='red',
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    ax1.set_ylabel('Relative Brightness', fontsize=11)
    ax1.set_title('Prediction 3: Discrete Activity Jumps\nStep-like changes uncorrelated with distance', 
                  fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Panel 2: Non-gravitational acceleration
    ax2.set_title('Non-gravitational Acceleration with Jumps', 
                  fontsize=12, fontweight='bold')
    
    # Base acceleration
    base_accel = 0.01 / r_helio**2
    
    # Add jumps (different times)
    accel_jump_times = [38, 87, 142, 192]
    accel_jump_sizes = [0.004, -0.003, 0.005, -0.004]
    
    predicted_accel = base_accel.copy()
    
    for jump_time, jump_size in zip(accel_jump_times, accel_jump_sizes):
        idx = np.argmin(np.abs(time_days - jump_time))
        predicted_accel[idx:] += jump_size
    
    # Add noise
    accel_noise = 0.001 * np.random.randn(len(time_days))
    measured_accel = predicted_accel + accel_noise
    
    ax2.plot(time_days, base_accel, 'k:', linewidth=1, alpha=0.5, 
             label='Smooth baseline')
    ax2.plot(time_days, measured_accel, 'g-', linewidth=2, 
             label='Measured (with jumps)')
    
    for jump_time, jump_size in zip(accel_jump_times, accel_jump_sizes):
        ax2.axvline(x=jump_time, color='purple', linestyle=':', alpha=0.7, linewidth=1)
    
    ax2.set_ylabel('Acceleration (arb. units)', fontsize=11)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Panel 3: Correlation analysis (key test)
    ax3.set_title('Lack of Correlation with Distance (Key Test)', 
                  fontsize=12, fontweight='bold')
    
    # Calculate residuals
    brightness_residuals = measured - base_brightness
    accel_residuals = measured_accel - base_accel
    
    # Plot residuals vs distance
    ax3.scatter(r_helio, brightness_residuals, c=time_days, 
                cmap='viridis', s=20, alpha=0.7, 
                label='Brightness residuals')
    
    ax3.scatter(r_helio, accel_residuals, c=time_days,
                cmap='plasma', s=20, alpha=0.7, marker='s',
                label='Acceleration residuals')
    
    ax3.set_xlabel('Heliocentric Distance (AU)', fontsize=11)
    ax3.set_ylabel('Residual (Observed - Expected)', fontsize=11)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    # Add correlation coefficients
    corr_brightness = np.corrcoef(r_helio, brightness_residuals)[0,1]
    corr_accel = np.corrcoef(r_helio, accel_residuals)[0,1]
    
    stats = (
        f"Correlation coefficients:\n"
        f"Brightness: {corr_brightness:.3f}\n"
        f"Acceleration: {corr_accel:.3f}\n"
        f"\nPrediction: |correlation| < 0.1\n"
        f"(Jumps uncorrelated with distance)"
    )
    
    ax3.text(0.02, 0.02, stats, transform=ax3.transAxes, fontsize=9,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.9))
    
    plt.tight_layout()
    return fig


def create_verification_table():
    """Create verification criteria table for predictions."""
    data = {
        'Prediction': [
            '1. Anti-tail Coherence',
            '2. Synchronized Wobble', 
            '3. Discrete Activity Jumps'
        ],
        'Observable': [
            'HST/SPHEREx tail images',
            'Time-resolved jet directions',
            'Photometry + acceleration'
        ],
        'Key Test': [
            'Width < 0.2×typical, coherent > 2 weeks',
            f'In-phase motion, period ~{OMEGA_PERIOD_DAYS:.1f} days',
            'Step changes, |correlation| < 0.1 with distance'
        ],
        'Verification': [
            'Coherent structure in ≥2 epochs',
            'Wobble detected with expected period',
            '≥2 significant jumps observed'
        ],
        'Falsification': [
            'Tail disperses like typical comet',
            'Jets move independently/not at all',
            'Changes smooth & distance-correlated'
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Create visual table
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center',
                     colColours=['#1f77b4']*len(df.columns))
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#1f77b4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title('Verification Criteria for Information Dynamics Framework\n' +
                 '3I/ATLAS becomes unobservable by mid-2026', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig, df


# ============================================================================
# TEXTUAL PREDICTION SUMMARY
# ============================================================================

def print_prediction_summary():
    """Print detailed summary of all predictions."""
    print("=" * 70)
    print("THREE TESTABLE PREDICTIONS FOR 3I/ATLAS")
    print("Information Dynamics Framework")
    print("=" * 70)
    
    print(f"\nModel parameters for 3I/ATLAS:")
    for key, value in PARAMS_3I.items():
        if key in ['γ', 'ε']:
            print(f"  {key} = {value:.2e}")
        else:
            print(f"  {key} = {value}")
    
    print(f"\nDerived quantities:")
    print(f"  Jet wobble period: T = 2π/ω ≈ {OMEGA_PERIOD_DAYS:.2f} days")
    
    print("\n" + "-" * 40)
    print("PREDICTION 1: ANTI-TAIL COHERENCE")
    print("-" * 40)
    print("Description: Any sunward anti-tail will maintain a narrow, coherent structure")
    print("over weeks, resisting solar wind dispersion due to underlying jet stability")
    print("driven by the nonlinear term ε.")
    print("\nKey indicators:")
    print("  • Narrow, collimated anti-tail structure")
    print("  • Minimal broadening over timescales of weeks")
    print("  • Resistance to solar wind disruption")
    print("  • Width < 20% of typical comet tail width")
    
    print("\n" + "-" * 40)
    print("PREDICTION 2: SYNCHRONIZED WOBBLE")
    print("-" * 40)
    print("Description: If jets are confirmed, they will wobble in phase (synchronously)")
    print(f"while preserving mutual 120° separation, with a period tied to ω.")
    print(f"\nPredicted wobble period: T = 2π/ω ≈ {OMEGA_PERIOD_DAYS:.2f} days")
    print("\nKey indicators:")
    print(f"  • Synchronized wobble of all {PARAMS_3I['N']} jets")
    print(f"  • Period of ~{OMEGA_PERIOD_DAYS:.2f} days")
    print("  • Preservation of 120° angular separation between jets")
    print("  • In-phase oscillation (no phase lag between jets)")
    
    print("\n" + "-" * 40)
    print("PREDICTION 3: DISCRETE ACTIVITY JUMPS")
    print("-" * 40)
    print("Description: Total brightness and acceleration may show step-like changes")
    print("uncorrelated with heliocentric distance, stemming from noise-induced")
    print("state transitions (η(t) term in the model).")
    print("\nKey indicators:")
    print("  • Sudden brightness changes (Δmag ~ 0.5-1.0)")
    print("  • Abrupt changes in non-gravitational acceleration")
    print("  • No correlation with heliocentric distance")
    print("  • Timescale between jumps: days to weeks")
    print("  • Correlation coefficient with distance: |r| < 0.1")
    
    print("\n" + "=" * 70)
    print("VALIDATION CRITERIA")
    print("=" * 70)
    print("\nThe paper states:")
    print("\"Verification of any two of these predictions before mid-2026")
    print("would constitute strong evidence for the model.\"")
    
    print("\nFalsification conditions:")
    print("1. 3I/ATLAS displays strong micro-dust scattering (contradicting p=0.17)")
    print("2. None of these predictions are observed despite adequate monitoring")
    print("3. Future ISOs fall outside the (γ, ε) parameter space defined")
    
    print("\n" + "=" * 70)
    print("OBSERVATIONAL TIMELINE")
    print("=" * 70)
    print("\nCritical: 3I/ATLAS becomes unobservable by mid-2026")
    print(f"Current date: {TIMELINE['current'].strftime('%B %d, %Y')}")
    print(f"Unobservable after: {TIMELINE['unobservable'].strftime('%B %Y')}")
    
    print("\nRecommended observational schedule:")
    print("  • Weekly monitoring: High-resolution imaging")
    print("  • Daily photometry: Brightness measurements")
    print("  • Spectroscopy: Regular dust/composition checks")
    print("  • Astrometry: Precise non-gravitational acceleration")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Generate all prediction visualizations and summaries."""
    print("=" * 70)
    print("INFORMATION DYNAMICS: TESTABLE PREDICTIONS FOR 3I/ATLAS")
    print("=" * 70)
    
    # Print textual summary
    print_prediction_summary()
    
    # Generate visualizations
    print("\n\nGenerating visualizations...")
    
    print("\n1. Creating observation timeline...")
    fig1 = create_observation_timeline()
    fig1.savefig('observation_timeline.png', dpi=300, bbox_inches='tight')
    print("   Saved as 'observation_timeline.png'")
    
    print("\n2. Visualizing Prediction 1: Anti-tail Coherence...")
    fig2 = visualize_prediction_1()
    fig2.savefig('prediction_1_anti_tail.png', dpi=300, bbox_inches='tight')
    print("   Saved as 'prediction_1_anti_tail.png'")
    
    print("\n3. Visualizing Prediction 2: Synchronized Wobble...")
    fig3, period = visualize_prediction_2()
    fig3.savefig('prediction_2_wobble.png', dpi=300, bbox_inches='tight')
    print(f"   Saved as 'prediction_2_wobble.png'")
    print(f"   Predicted wobble period: {period:.2f} days")
    
    print("\n4. Visualizing Prediction 3: Discrete Activity Jumps...")
    fig4 = visualize_prediction_3()
    fig4.savefig('prediction_3_jumps.png', dpi=300, bbox_inches='tight')
    print("   Saved as 'prediction_3_jumps.png'")
    
    print("\n5. Creating verification table...")
    fig5, df = create_verification_table()
    fig5.savefig('verification_table.png', dpi=300, bbox_inches='tight')
    print("   Saved as 'verification_table.png'")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
    Three testable predictions for 3I/ATLAS before mid-2026:
    
    1. ANTI-TAIL COHERENCE
       - Narrow, coherent structure resistant to solar wind
       - Indicates ε-driven macroscopic order
    
    2. SYNCHRONIZED WOBBLE
       - Jets wobble in phase with period ~{OMEGA_PERIOD_DAYS:.2f} days
       - Preserves 120° separation (N={PARAMS_3I['N']})
    
    3. DISCRETE ACTIVITY JUMPS
       - Step-like changes in brightness/acceleration
       - Uncorrelated with heliocentric distance
       - Noise-induced state transitions
    
    Model parameter: p = {PARAMS_3I['p']:.2f} (mixed state)
    Validation: Any 2/3 predictions confirm the framework
    """)
    
    print("\n✅ All visualizations generated successfully.")
    print("📄 See paper for full theoretical framework and references.")
    
    # Show plots
    plt.show()
    
    return df


if __name__ == "__main__":
    df = main()
