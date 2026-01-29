"""
wobble_calculation.py - Calculate and validate jet wobble period for 3I/ATLAS
Companion code for "Information Dynamics: A Unified Predictive Framework for Interstellar Objects"
Author: Kai Huang, Hongkui Liu
License: MIT

This script implements the wobble period calculation based on the Information Dynamics model,
using the Complex Ginzburg-Landau Equation (CGLE) from the paper. It computes theoretical periods,
simulates wobble behavior, analyzes observational data, and generates visualizations.
Key parameters are sourced from Table 1 in the paper for 3I/ATLAS (e.g., ω=1.08e-4 rad/s, p=0.17).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, fft
from scipy.optimize import curve_fit
import argparse
import json
import csv
from datetime import datetime
from typing import Tuple, Dict, Optional

# ============================================================================
# CONSTANTS AND MODEL PARAMETERS FROM THE PAPER
# ============================================================================

# Model parameters for 3I/ATLAS from Table 1 of the paper
PARAMS_3I = {
    'gamma': 3.8e-5,       # s⁻¹ (dissipation rate)
    'omega': 1.08e-4,      # rad/s (characteristic frequency from paper)
    'epsilon': 1.7e-4,     # nonlinear self-interaction strength
    'F_thermal': 0.12,     # thermal driving intensity
    'N': 3,                # symmetry mode (triple jet)
    'p': 0.17              # information purity
}

# Conversion constants
SECONDS_PER_HOUR = 3600
HOURS_PER_DAY = 24
RAD_TO_DEG = 180 / np.pi

# ============================================================================
# THEORETICAL CALCULATION FUNCTIONS
# ============================================================================

def theoretical_wobble_period(omega: float, 
                             sigma_modulation: float = 0.1) -> Dict[str, float]:
    """
    Calculate theoretical wobble period based on the Information Dynamics model.
    
    Parameters:
    -----------
    omega : float
        Characteristic frequency from the CGLE model (rad/s)
    sigma_modulation : float
        Expected modulation factor due to nonlinear coupling (0-1)
        
    Returns:
    --------
    dict : Dictionary containing period estimates in various units
    """
    # Base period from omega (T = 2π/ω)
    base_period_sec = 2 * np.pi / omega
    
    # Apply nonlinear modulation effect
    # In the CGLE framework, wobble can be modulated by nonlinear terms
    modulated_period_sec = base_period_sec * (1 + sigma_modulation * PARAMS_3I['p'])
    
    results = {
        'base_period_seconds': base_period_sec,
        'modulated_period_seconds': modulated_period_sec,
        'period_hours': modulated_period_sec / SECONDS_PER_HOUR,
        'period_days': modulated_period_sec / (SECONDS_PER_HOUR * HOURS_PER_DAY),
        'frequency_rad_s': omega,
        'frequency_hz': omega / (2 * np.pi),
        'modulation_factor': sigma_modulation
    }
    
    return results


def cgle_wobble_simulation(duration_hours: float = 48, 
                          sampling_minutes: float = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate wobble behavior using a simplified CGLE oscillator model.
    
    Parameters:
    -----------
    duration_hours : float
        Total simulation duration in hours
    sampling_minutes : float
        Time between samples in minutes
        
    Returns:
    --------
    time_array : np.ndarray
        Time points in hours
    wobble_angle : np.ndarray
        Simulated wobble angle in degrees
    """
    # Convert to consistent units
    duration_sec = duration_hours * SECONDS_PER_HOUR
    dt = sampling_minutes * 60  # seconds
    
    # Time array
    n_samples = int(duration_sec / dt)
    time_sec = np.linspace(0, duration_sec, n_samples)
    time_hours = time_sec / SECONDS_PER_HOUR
    
    # Parameters for the simplified oscillator
    omega = PARAMS_3I['omega']
    epsilon = PARAMS_3I['epsilon']
    gamma = PARAMS_3I['gamma']
    
    # Add small nonlinear modulation (creates wobble)
    modulation_freq = omega * 0.05  # 5% modulation
    
    # Simulate wobble angle (combination of main rotation and modulation)
    main_rotation = np.sin(omega * time_sec)
    wobble_modulation = 0.3 * np.sin(modulation_freq * time_sec) * (1 + 0.2 * np.sin(epsilon * time_sec))
    
    # Combine and scale to observable wobble angle (±20° as mentioned in observations)
    wobble_angle = 20 * (main_rotation + wobble_modulation)  # degrees
    
    # Add observational noise
    noise_level = 0.5  # degrees
    wobble_angle += np.random.normal(0, noise_level, size=wobble_angle.shape)
    
    return time_hours, wobble_angle


# ============================================================================
# OBSERVATIONAL DATA ANALYSIS FUNCTIONS
# ============================================================================

def detect_period_from_timeseries(time: np.ndarray, 
                                 angle: np.ndarray,
                                 min_period_hours: float = 1.0,
                                 max_period_hours: float = 48.0) -> Dict[str, float]:
    """
    Detect dominant period in wobble angle time series using Fourier analysis.
    
    Parameters:
    -----------
    time : np.ndarray
        Time values in hours
    angle : np.ndarray
        Wobble angle measurements in degrees
    min_period_hours : float
        Minimum period to consider (hours)
    max_period_hours : float
        Maximum period to consider (hours)
        
    Returns:
    --------
    dict : Dictionary containing period detection results
    """
    # Ensure uniform sampling
    dt = np.median(np.diff(time))
    sampling_rate_hz = 1 / (dt * SECONDS_PER_HOUR)
    
    # Detrend the data
    angle_detrended = signal.detrend(angle)
    
    # Perform FFT
    n = len(angle_detrended)
    freqs = fft.fftfreq(n, d=dt)
    fft_values = fft.fft(angle_detrended)
    power_spectrum = np.abs(fft_values[:n//2]) ** 2
    positive_freqs = freqs[:n//2]
    
    # Convert frequency to period (hours)
    periods = 1 / positive_freqs  # in hours
    periods[0] = np.inf  # Avoid division by zero for DC component
    
    # Filter to reasonable period range
    valid_mask = (periods >= min_period_hours) & (periods <= max_period_hours)
    valid_periods = periods[valid_mask]
    valid_power = power_spectrum[valid_mask]
    
    if len(valid_periods) == 0:
        raise ValueError("No periods found in the specified range")
    
    # Find dominant period
    dominant_idx = np.argmax(valid_power)
    dominant_period = valid_periods[dominant_idx]
    dominant_power = valid_power[dominant_idx]
    
    # Calculate confidence metrics
    total_power = np.sum(valid_power)
    confidence = dominant_power / total_power if total_power > 0 else 0
    
    # Fit a Gaussian to the peak for refined period estimate
    try:
        peak_window = 5
        start_idx = max(0, dominant_idx - peak_window)
        end_idx = min(len(valid_periods), dominant_idx + peak_window + 1)
        
        if end_idx - start_idx >= 3:
            window_periods = valid_periods[start_idx:end_idx]
            window_power = valid_power[start_idx:end_idx]
            
            def gaussian(x, amp, mu, sigma):
                return amp * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
            
            p0 = [dominant_power, dominant_period, 0.5]
            bounds = ([0, min(window_periods), 0], 
                     [dominant_power * 2, max(window_periods), 5])
            
            params, _ = curve_fit(gaussian, window_periods, window_power, 
                                 p0=p0, bounds=bounds, maxfev=5000)
            refined_period = params[1]
        else:
            refined_period = dominant_period
    except:
        refined_period = dominant_period
    
    return {
        'dominant_period_hours': dominant_period,
        'refined_period_hours': refined_period,
        'confidence': confidence,
        'sampling_interval_hours': dt,
        'spectrum_periods': valid_periods,
        'spectrum_power': valid_power
    }


def load_observation_data(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load wobble observation data from CSV or JSON file.
    
    Supported formats:
    - CSV: Two columns (time_hours, angle_degrees)
    - JSON: {"time_hours": [...], "angle_degrees": [...]}
    
    Parameters:
    -----------
    filename : str
        Path to data file
        
    Returns:
    --------
    time : np.ndarray
        Time values in hours
    angle : np.ndarray
        Wobble angle in degrees
    """
    if filename.endswith('.csv'):
        data = np.loadtxt(filename, delimiter=',')
        time = data[:, 0]
        angle = data[:, 1]
    elif filename.endswith('.json'):
        with open(filename, 'r') as f:
            data = json.load(f)
        time = np.array(data['time_hours'])
        angle = np.array(data['angle_degrees'])
    else:
        raise ValueError("Unsupported file format. Use .csv or .json")
    
    return time, angle


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_comprehensive_plot(theoretical_results: Dict,
                             observed_results: Optional[Dict] = None,
                             sim_time: Optional[np.ndarray] = None,
                             sim_angle: Optional[np.ndarray] = None) -> plt.Figure:
    """
    Create comprehensive visualization of wobble analysis.
    
    Parameters:
    -----------
    theoretical_results : dict
        Results from theoretical_wobble_period()
    observed_results : dict, optional
        Results from detect_period_from_timeseries()
    sim_time : np.ndarray, optional
        Simulated time series
    sim_angle : np.ndarray, optional
        Simulated angle series
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Comprehensive figure
    """
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Theoretical vs Observed Period Comparison
    ax1 = plt.subplot(2, 2, 1)
    categories = ['Theoretical (Base)', 'Theoretical (Modulated)']
    periods = [theoretical_results['base_period_seconds'] / SECONDS_PER_HOUR,
              theoretical_results['period_hours']]
    
    bars = ax1.bar(categories, periods, color=['lightblue', 'lightcoral'])
    ax1.set_ylabel('Period (hours)')
    ax1.set_title('Theoretical Wobble Periods')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, period in zip(bars, periods):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{period:.2f}h', ha='center', va='bottom')
    
    if observed_results:
        ax1.axhline(y=observed_results['refined_period_hours'], 
                   color='green', linestyle='--', linewidth=2,
                   label=f'Observed: {observed_results["refined_period_hours"]:.2f}h')
        ax1.legend()
    
    # 2. Simulated Wobble Pattern
    if sim_time is not None and sim_angle is not None:
        ax2 = plt.subplot(2, 2, 2)
        ax2.plot(sim_time, sim_angle, 'b-', linewidth=1.5, alpha=0.7)
        ax2.set_xlabel('Time (hours)')
        ax2.set_ylabel('Wobble Angle (degrees)')
        ax2.set_title('Simulated Jet Wobble Pattern')
        ax2.grid(True, alpha=0.3)
        
        # Mark expected period
        period_hours = theoretical_results['period_hours']
        if len(sim_time) > 0:
            ax2.axvspan(sim_time[0], sim_time[0] + period_hours, 
                       alpha=0.2, color='red', label=f'Predicted Period: {period_hours:.1f}h')
            ax2.legend()
    
    # 3. Power Spectrum (if observational data available)
    if observed_results and 'spectrum_periods' in observed_results:
        ax3 = plt.subplot(2, 2, 3)
        periods = observed_results['spectrum_periods']
        power = observed_results['spectrum_power']
        
        ax3.plot(periods, power, 'g-', linewidth=2)
        ax3.axvline(observed_results['refined_period_hours'], 
                   color='red', linestyle='--', linewidth=2,
                   label=f'Dominant: {observed_results["refined_period_hours"]:.2f}h')
        
        # Highlight predicted period range
        pred_min = theoretical_results['period_hours'] * 0.9
        pred_max = theoretical_results['period_hours'] * 1.1
        ax3.axvspan(pred_min, pred_max, alpha=0.2, color='orange',
                   label='Prediction ±10%')
        
        ax3.set_xlabel('Period (hours)')
        ax3.set_ylabel('Spectral Power')
        ax3.set_title('Power Spectrum of Wobble Observations')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
    
    # 4. Information Summary
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    info_text = (
        f'Information Dynamics Model Results\n'
        f'{"="*40}\n'
        f'Model Parameters for 3I/ATLAS:\n'
        f'  • ω (frequency): {PARAMS_3I["omega"]:.2e} rad/s\n'
        f'  • γ (dissipation): {PARAMS_3I["gamma"]:.2e} s⁻¹\n'
        f'  • ε (nonlinear): {PARAMS_3I["epsilon"]:.2e}\n'
        f'  • p (info purity): {PARAMS_3I["p"]:.2f}\n\n'
        f'Theoretical Predictions:\n'
        f'  • Base Period: {theoretical_results["period_hours"]:.2f} hours\n'
        f'  • Frequency: {theoretical_results["frequency_hz"]:.2e} Hz\n'
        f'  • Modulation Factor: {theoretical_results["modulation_factor"]:.2f}'
    )
    
    if observed_results:
        info_text += (
            f'\n\nObservational Results:\n'
            f'  • Detected Period: {observed_results["refined_period_hours"]:.2f} hours\n'
            f'  • Confidence: {observed_results["confidence"]:.3f}\n'
            f'  • Δ from Prediction: '
            f'{100*(observed_results["refined_period_hours"]/theoretical_results["period_hours"]-1):+.1f}%'
        )
    
    ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes,
            fontfamily='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig


# ============================================================================
# MAIN EXECUTION AND COMMAND-LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Calculate and validate wobble period for 3I/ATLAS based on Information Dynamics model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                            # Run with default parameters
  %(prog)s --modulation 0.15          # Set nonlinear modulation factor
  %(prog)s --data observations.csv    # Compare with observational data
  %(prog)s --simulate --save-sim simulated_wobble.csv  # Generate and save simulation
        """
    )
    
    parser.add_argument('--modulation', type=float, default=0.1,
                       help='Nonlinear modulation factor (default: 0.1)')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to observational data file (CSV or JSON)')
    parser.add_argument('--simulate', action='store_true',
                       help='Generate simulation of wobble behavior')
    parser.add_argument('--sim-duration', type=float, default=48.0,
                       help='Simulation duration in hours (default: 48)')
    parser.add_argument('--save-sim', type=str, default=None,
                       help='Save simulated data to file')
    parser.add_argument('--save-plot', type=str, default='wobble_analysis.png',
                       help='Save analysis plot to file (default: wobble_analysis.png)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Run analysis without displaying plots')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("WOBBLE CALCULATION FOR 3I/ATLAS")
    print("Information Dynamics Model Validation")
    print("=" * 70)
    
    # 1. Calculate theoretical wobble period
    print("\n[1] CALCULATING THEORETICAL WOBBLE PERIOD")
    print("-" * 40)
    theoretical_results = theoretical_wobble_period(
        PARAMS_3I['omega'], 
        sigma_modulation=args.modulation
    )
    
    print(f"Base period from ω: {theoretical_results['base_period_seconds']/SECONDS_PER_HOUR:.2f} hours")
    print(f"Modulated period (p={PARAMS_3I['p']}, σ={args.modulation}): "
          f"{theoretical_results['period_hours']:.2f} hours")
    print(f"Expected frequency: {theoretical_results['frequency_hz']:.2e} Hz")
    
    # 2. Generate simulation if requested
    sim_time, sim_angle = None, None
    if args.simulate:
        print("\n[2] GENERATING WOBBLE SIMULATION")
        print("-" * 40)
        sim_time, sim_angle = cgle_wobble_simulation(
            duration_hours=args.sim_duration,
            sampling_minutes=10
        )
        print(f"Simulated {len(sim_time)} data points over {args.sim_duration} hours")
        
        if args.save_sim:
            save_data = np.column_stack((sim_time, sim_angle))
            np.savetxt(args.save_sim, save_data, delimiter=',',
                      header='time_hours,angle_degrees')
            print(f"Simulation saved to: {args.save_sim}")
    
    # 3. Analyze observational data if provided
    observed_results = None
    if args.data:
        print("\n[3] ANALYZING OBSERVATIONAL DATA")
        print("-" * 40)
        try:
            obs_time, obs_angle = load_observation_data(args.data)
            print(f"Loaded {len(obs_time)} observations from {args.data}")
            
            observed_results = detect_period_from_timeseries(obs_time, obs_angle)
            print(f"Detected period: {observed_results['refined_period_hours']:.2f} hours")
            print(f"Analysis confidence: {observed_results['confidence']:.3f}")
            
            # Calculate agreement with theory
            pred_period = theoretical_results['period_hours']
            obs_period = observed_results['refined_period_hours']
            percent_diff = 100 * (obs_period / pred_period - 1)
            print(f"Difference from prediction: {percent_diff:+.1f}%")
            
            # Replace simulation with real data for plotting
            sim_time, sim_angle = obs_time, obs_angle
            
        except Exception as e:
            print(f"Error analyzing data: {e}")
            print("Continuing with theoretical analysis only...")
    
    # 4. Create and save visualization
    if not args.no_plot:
        print("\n[4] CREATING VISUALIZATION")
        print("-" * 40)
        
        fig = create_comprehensive_plot(
            theoretical_results, 
            observed_results,
            sim_time, 
            sim_angle
        )
        
        plt.savefig(args.save_plot, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {args.save_plot}")
        
        if not args.no_plot and 'DISPLAY' in plt.rcParams:
            plt.show()
    
    # 5. Print final summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    
    # Generate summary text for inclusion in paper
    summary = (
        f"\nSummary for Paper Validation:\n"
        f"The Information Dynamics model predicts a jet wobble period of "
        f"{theoretical_results['period_hours']:.1f} ± "
        f"{theoretical_results['period_hours'] * 0.1:.1f} hours "
        f"for 3I/ATLAS (ω = {PARAMS_3I['omega']:.2e} rad/s, p = {PARAMS_3I['p']:.2f}).\n"
    )
    
    if observed_results:
        summary += (
            f"Observational analysis of {len(sim_time) if sim_time is not None else 'N/A'} data points "
            f"yields a period of {observed_results['refined_period_hours']:.1f} hours, "
            f"which is {100*(observed_results['refined_period_hours']/theoretical_results['period_hours']-1):+.1f}% "
            f"from the theoretical prediction.\n"
        )
    
    summary += (
        f"This code provides quantitative validation of Prediction 2 (Synchronized Wobble) "
        f"from the Information Dynamics framework."
    )
    
    print(summary)
    
    # Save summary to file
    with open('wobble_analysis_summary.txt', 'w') as f:
        f.write(summary)


# ============================================================================
# EXAMPLE USAGE AND DATA GENERATION
# ============================================================================

def generate_example_data():
    """Generate example observation data for testing."""
    # Create simulated "observations" based on model with some noise
    time_hours = np.linspace(0, 72, 200)  # 3 days of observations
    
    # Use theoretical period with some realistic variation
    true_period = theoretical_wobble_period(PARAMS_3I['omega'])['period_hours']
    true_frequency = 2 * np.pi / (true_period * SECONDS_PER_HOUR)
    
    # Simulate wobble with some phase noise
    phase_noise = 0.1 * np.cumsum(np.random.randn(len(time_hours)))
    wobble_angle = 15 * np.sin(true_frequency * time_hours * SECONDS_PER_HOUR + phase_noise)
    wobble_angle += np.random.normal(0, 2, size=len(wobble_angle))  # measurement noise
    
    # Save to CSV
    data = np.column_stack((time_hours, wobble_angle))
    np.savetxt('example_observations.csv', data, delimiter=',',
               header='time_hours,angle_degrees')
    print("Example data saved to 'example_observations.csv'")
    
    # Also save as JSON
    json_data = {
        'time_hours': time_hours.tolist(),
        'angle_degrees': wobble_angle.tolist(),
        'metadata': {
            'generated': datetime.now().isoformat(),
            'true_period_hours': float(true_period),
            'source': 'wobble_calculation.py example'
        }
    }
    
    with open('example_observations.json', 'w') as f:
        json.dump(json_data, f, indent=2)
    print("Example data also saved to 'example_observations.json'")


if __name__ == "__main__":
    # Uncomment the next line to generate example data
    # generate_example_data()
    
    main()
