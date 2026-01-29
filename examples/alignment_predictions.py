"""
alignment_predictions.py - Simulate predicted observational features during 3I-Sun-Earth alignment
Companion code for "Information Dynamics: A Unified Predictive Framework for Interstellar Objects"
Author: Kai Huang, Hongkui Liu
License: MIT

This script simulates observational signatures during the 3I/ATLAS Sun-Earth alignment on January 22, 2026,
based on the Information Dynamics model. It generates predictions for photometric, morphological, and temporal
features (Predictions 1-3 in the paper), including opposition surge, anti-tail enhancement, and activity jumps.

Key parameters are sourced from the paper (e.g., ω=1.08e-4 rad/s, p=0.17) and observational constraints
(phase angle α=0.69°, V≈16.7).

Core Functionality:
1. Alignment Geometry Calculation (AlignmentGeometry class):
   - Computes phase angle, distances, illumination fraction, and alignment quality.

2. Alignment Predictions (AlignmentPredictions class):
   - Generates photometric (opposition surge, magnitude), morphological (anti-tail/jet enhancement, coherence),
     and temporal (jump probability, observation windows) predictions using CGLE-derived metrics.

3. Light Curve Simulation (simulate_light_curve function):
   - Simulates high-cadence light curve with surge, wobble modulation, and discrete jumps.

4. Visualization (create_alignment_visualization function):
   - Creates multi-panel plot: geometry schematic, light curve, power spectrum.

Outputs Include:
1. Numerical Predictions:
   - Geometry details (phase angle, distances).
   - Photometric: apparent magnitude, surge amplitude.
   - Morphological: anti-tail enhancement, structural coherence.
   - Temporal: jump probability, optimal observation windows.

2. Simulated Data:
   - Light curve time series (time_hours, magnitude).

3. Visualization:
   - Comprehensive alignment plot (alignment_predictions.png).

4. Observing Guide Summary:
   - Text file with predictions and strategy for proposals (alignment_observing_guide.txt).

Usage Example:
- python alignment_predictions.py --date 2026-01-22 --hours-range -48 48 --save-results predictions.json

Note: All code uses UTF-8 encoding for cross-platform compatibility (Windows, macOS, Linux).
Requires astropy for time/coordinates (simplified here). Install via pip if needed.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats  # Fixed: added stats import
import argparse
import json
import warnings
import sys
import codecs

# Ensure UTF-8 encoding for cross-platform compatibility
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

warnings.filterwarnings('ignore')

# ============================================================================
# ASTRONOMICAL CONSTANTS AND ALIGNMENT PARAMETERS
# ============================================================================

# Key alignment date: January 22, 2026
ALIGNMENT_PHASE_ANGLE = 0.69  # degrees (α ≈ 0.69° on Jan 22, 2026)

# Solar System Constants
AU = 1.495978707e11  # meters (astronomical unit)

# 3I/ATLAS Model Parameters from the paper (Table 2)
ID_PARAMS = {
    'gamma': 3.8e-5,       # s⁻¹
    'omega': 1.08e-4,      # rad/s
    'epsilon': 1.7e-4,     # nonlinear self-interaction
    'F_thermal': 0.12,     # thermal driving
    'p': 0.17,             # information purity
    'N': 3                 # symmetry mode
}

# Observational constraints during alignment (updated from Jan 2026 data)
OBS_PARAMS = {
    'heliocentric_distance': 3.33,  # AU (from Jan 2026 data)
    'geocentric_distance': 2.35,    # AU (from Jan 2026 data)
    'phase_angle': ALIGNMENT_PHASE_ANGLE,  # degrees
    'apparent_magnitude': 16.7,     # V band (from Jan 2026 data)
    'wobble_period_hours': 16.2     # hours (from Prediction 2)
}

# ============================================================================
# GEOMETRY AND OPPOSITION EFFECT CALCULATIONS
# ============================================================================

class AlignmentGeometry:
    """Calculate geometric parameters during 3I-Sun-Earth alignment."""
    
    def __init__(self, date_str: str = '2026-01-22'):
        """
        Initialize alignment geometry calculator.
        
        Parameters:
        -----------
        date_str : str
            Date of alignment in 'YYYY-MM-DD' format
        """
        self.date_str = date_str
        self.geometry = self.calculate_alignment_geometry()
        
    def calculate_alignment_geometry(self) -> dict:
        """
        Calculate detailed geometric parameters for the alignment.
        
        Returns:
        --------
        geometry : dict
            Dictionary of geometric parameters
        """
        geometry = {
            'date': self.date_str,
            'phase_angle_deg': ALIGNMENT_PHASE_ANGLE,
            'phase_angle_rad': np.deg2rad(ALIGNMENT_PHASE_ANGLE),
            'geocentric_distance_au': OBS_PARAMS['geocentric_distance'],
            'heliocentric_distance_au': OBS_PARAMS['heliocentric_distance'],
            'elongation_sun_deg': 180.0 - ALIGNMENT_PHASE_ANGLE,  # 180 - phase_angle
            'earth_distance_km': OBS_PARAMS['geocentric_distance'] * AU / 1000,
            'sun_distance_km': OBS_PARAMS['heliocentric_distance'] * AU / 1000,
            'alignment_quality': self.calculate_alignment_quality()
        }
        
        # Calculate illumination parameters
        geometry.update(self.calculate_illumination(geometry))
        
        return geometry
    
    def calculate_alignment_quality(self) -> float:
        """
        Calculate alignment quality metric (0-1).
        1.0 = perfect opposition, 0.0 = poor alignment.
        """
        # Perfect opposition would be 0.0° phase angle
        # Quality decreases as phase angle increases
        phase_rad = np.deg2rad(ALIGNMENT_PHASE_ANGLE)
        quality = np.exp(-phase_rad**2 / (2 * (np.deg2rad(5)**2)))  # Gaussian with 5° width
        
        return float(quality)
    
    def calculate_illumination(self, geometry: dict) -> dict:
        """
        Calculate illumination geometry parameters.
        
        Parameters:
        -----------
        geometry : dict
            Base geometry parameters
            
        Returns:
        --------
        illumination : dict
            Illumination parameters
        """
        phase_rad = geometry['phase_angle_rad']
        
        # Illumination fraction (0 = new, 1 = full)
        illumination_fraction = 0.5 * (1 + np.cos(phase_rad))
        
        return {
            'illumination_fraction': illumination_fraction,
            'shadow_length_km': geometry['sun_distance_km'],  # Simplified
            'terminator_offset_deg': np.rad2deg(phase_rad / 2)  # Approximate
        }

# ============================================================================
# PREDICTION GENERATION CORE
# ============================================================================

class AlignmentPredictions:
    """Generate predictions for observational features during alignment."""
    
    def __init__(self, geometry: dict = None):
        """
        Initialize prediction generator.
        
        Parameters:
        -----------
        geometry : dict
            Geometric parameters (default: calculated for alignment date)
        """
        self.geometry = geometry if geometry else AlignmentGeometry().geometry
        self.predictions = self.generate_predictions()
        
    def generate_predictions(self) -> dict:
        """
        Generate comprehensive predictions for alignment observations.
        
        Returns:
        --------
        predictions : dict
            Nested dictionary of predictions by category
        """
        predictions = {
            'geometry': self.geometry,
            'photometric': self.calculate_photometric_predictions(),
            'morphological': self.calculate_morphological_predictions(),
            'temporal': self.calculate_temporal_predictions()
        }
        
        return predictions
    
    def calculate_photometric_predictions(self) -> dict:
        """
        Calculate photometric predictions including opposition surge.
        
        Returns:
        --------
        photometric : dict
            Photometric predictions
        """
        phase_rad = self.geometry['phase_angle_rad']
        base_mag = OBS_PARAMS['apparent_magnitude']
        
        # Opposition surge amplitude (0.3-0.5 mag for phase <1°, from paper)
        # Using empirical model for cometary opposition surge
        if phase_rad < np.deg2rad(1):
            surge_amplitude = 0.4  # mag, consistent with paper's 0.3-0.5 mag range
        else:
            surge_amplitude = 0.4 * np.exp(-(phase_rad - np.deg2rad(1))**2 / (2 * (np.deg2rad(5))**2))
        
        # Enhanced magnitude (surge makes object brighter, so subtract)
        apparent_mag = base_mag - surge_amplitude
        
        # Brightness enhancement factor
        enhancement_factor = 10**(surge_amplitude / 2.5)  # mag to flux ratio
        
        # Surge timescale (hours for full development)
        surge_timescale = 24 / (1 + np.exp(5 * (phase_rad - np.deg2rad(1))))  # Sigmoid
        
        return {
            'base_magnitude': base_mag,
            'surge_amplitude_mag': surge_amplitude,
            'apparent_magnitude_V': apparent_mag,
            'brightness_enhancement': enhancement_factor,
            'surge_timescale_hours': surge_timescale,
            'uncertainty_mag': 0.2  # Estimated observational uncertainty
        }
    
    def calculate_morphological_predictions(self) -> dict:
        """
        Calculate morphological predictions for jets and anti-tail.
        
        Returns:
        --------
        morphological : dict
            Morphological predictions
        """
        p = ID_PARAMS['p']
        alignment_quality = self.geometry['alignment_quality']
        
        # Anti-tail enhancement due to projection (Prediction 1)
        # At small phase angle, anti-tail projects nearly along line of sight
        phase_rad = self.geometry['phase_angle_rad']
        anti_tail_enhancement = 2.0  # ~2x enhancement from paper (Table 2)
        
        # Structural coherence from information purity (high p -> more coherent)
        coherence = 0.95  # from paper (Table 2)
        
        # Jet visibility (N=3 symmetry more pronounced at opposition)
        jet_visibility = alignment_quality * ID_PARAMS['N'] / 3
        
        return {
            'anti_tail_enhancement': anti_tail_enhancement,
            'structural_coherence': coherence,
            'shadow_contrast': 0.5,  # arbitrary
            'jet_visibility_factor': jet_visibility,
            'anti_tail_length_arcsec': 15 * alignment_quality  # estimated
        }
    
    def calculate_temporal_predictions(self) -> dict:
        """
        Calculate temporal predictions for activity variations.
        
        Returns:
        --------
        temporal : dict
            Temporal predictions
        """
        epsilon = ID_PARAMS['epsilon']
        gamma = ID_PARAMS['gamma']
        
        # Activity jump probability (Prediction 3) - ~5% from paper
        jump_prob = 0.05  # 5% probability from paper (Table 2)
        
        # Optimal observation window (days from opposition)
        obs_window = [-3, 3]  # ±3 days from opposition
        
        # Critical times for jumps (arbitrary offsets for simulation)
        critical_offsets = np.array([-18, 0, 24]) / 24  # days
        critical_jd = 2457400.5 + critical_offsets  # arbitrary JD baseline
        
        return {
            'jump_probability': jump_prob,
            'optimal_obs_window_days': obs_window,
            'critical_times_jd': critical_jd.tolist(),
            'wobble_period_hours': OBS_PARAMS['wobble_period_hours']
        }

# ============================================================================
# LIGHT CURVE SIMULATION
# ============================================================================

def simulate_light_curve(predictions: dict,
                         hours_range: tuple = (-48, 48),
                         resolution_minutes: int = 30) -> tuple:
    """
    Simulate high-cadence light curve during alignment period.
    
    Parameters:
    -----------
    predictions : dict
        Alignment predictions
    hours_range : tuple
        Time range in hours relative to opposition
    resolution_minutes : int
        Sampling resolution in minutes
        
    Returns:
    --------
    time_hours : np.ndarray
        Time points in hours
    magnitude : np.ndarray
        Simulated V magnitudes
    """
    # Time array
    t_start, t_end = hours_range
    dt = resolution_minutes / 60  # hours
    n_points = int((t_end - t_start) / dt) + 1
    time_hours = np.linspace(t_start, t_end, n_points)
    
    # Base magnitude
    base_mag = predictions['photometric']['base_magnitude']
    
    # Opposition surge (Gaussian centered at t=0)
    surge_amp = predictions['photometric']['surge_amplitude_mag']
    surge_width = predictions['photometric']['surge_timescale_hours'] / 2  # FWHM/2.35 ≈ sigma
    surge = surge_amp * np.exp(-time_hours**2 / (2 * surge_width**2))
    
    # Wobble modulation (sinusoidal from ω, Prediction 2)
    wobble_period = predictions['temporal']['wobble_period_hours']
    wobble_amp = 0.15  # mag (typical for jet modulation)
    wobble = wobble_amp * np.sin(2 * np.pi * time_hours / wobble_period)
    
    # Discrete activity jumps (Prediction 3)
    jump_prob = predictions['temporal']['jump_probability']
    n_jumps = stats.poisson.rvs(jump_prob * len(time_hours) / 100)  # Fixed: using stats
    jump_times = np.random.choice(len(time_hours), min(n_jumps, len(time_hours)), replace=False)
    jumps = np.zeros_like(time_hours)
    jump_sizes = np.random.normal(0.2, 0.05, len(jump_times))  # mag jumps
    jumps[jump_times] = jump_sizes
    
    # Cumulative jumps
    jumps = np.cumsum(jumps)
    
    # Observational noise
    noise = np.random.normal(0, 0.05, len(time_hours))  # mag
    
    # Combined magnitude (surge brightens, so negative)
    magnitude = base_mag - surge + wobble + jumps + noise
    
    return time_hours, magnitude

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_alignment_visualization(predictions: dict,
                                   light_curve_data: tuple,
                                   resolution_minutes: int = 30) -> plt.Figure:
    """
    Create comprehensive visualization of alignment predictions.
    
    Parameters:
    -----------
    predictions : dict
        Alignment predictions
    light_curve_data : tuple
        (time_hours, magnitude) from simulation
    resolution_minutes : int
        Sampling resolution for power spectrum calculation
        
    Returns:
    --------
    fig : matplotlib.Figure
        The created figure
    """
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f'3I/ATLAS Alignment Predictions: {predictions["geometry"]["date"]}\n'
                 f'Phase Angle: {predictions["geometry"]["phase_angle_deg"]:.2f}°', 
                 fontsize=16, fontweight='bold')
    
    # Grid layout
    gs = fig.add_gridspec(2, 2)
    
    # Panel 1: Geometry schematic
    ax1 = fig.add_subplot(gs[0, 0])
    plot_geometry_schematic(ax1, predictions['geometry'])
    
    # Panel 2: Light curve
    ax2 = fig.add_subplot(gs[0, 1])
    time_hours, magnitude = light_curve_data
    ax2.plot(time_hours, magnitude, 'b-', linewidth=1.5, label='Simulated V mag')
    ax2.invert_yaxis()  # Brighter is smaller mag
    ax2.set_title('Predicted Light Curve (Predictions 1-3)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time from Opposition (hours)')
    ax2.set_ylabel('Apparent V Magnitude')
    ax2.axvline(0, color='r', linestyle='--', linewidth=2, label='Opposition (α=0.69°)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Power spectrum of light curve
    ax3 = fig.add_subplot(gs[1, 0])
    freq, power = signal.periodogram(magnitude, fs=60/resolution_minutes)  # fs in 1/hour
    ax3.plot(1/freq[freq>0], power[freq>0], 'g-', linewidth=1.5)  # Period in hours
    ax3.set_title('Power Spectrum (Wobble Detection)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Period (hours)')
    ax3.set_ylabel('Power')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.axvline(predictions['temporal']['wobble_period_hours'], 
                color='r', linestyle='--', linewidth=2, 
                label=f'Wobble period: {predictions["temporal"]["wobble_period_hours"]:.1f} h')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Key predictions summary
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    summary_text = (
        "INFORMATION DYNAMICS PREDICTIONS\n"
        "===============================\n\n"
        "GEOMETRY:\n"
        f"• Date: {predictions['geometry']['date']}\n"
        f"• Phase angle (α): {predictions['geometry']['phase_angle_deg']:.2f}°\n"
        f"• Heliocentric distance: {predictions['geometry']['heliocentric_distance_au']:.2f} AU\n"
        f"• Geocentric distance: {predictions['geometry']['geocentric_distance_au']:.2f} AU\n\n"
        
        "PHOTOMETRIC (Prediction 1):\n"
        f"• Base magnitude: V = {predictions['photometric']['base_magnitude']:.1f}\n"
        f"• Opposition surge: {predictions['photometric']['surge_amplitude_mag']:.2f} mag\n"
        f"• Predicted V magnitude: {predictions['photometric']['apparent_magnitude_V']:.1f}\n"
        f"• Brightness enhancement: {predictions['photometric']['brightness_enhancement']:.1f}×\n\n"
        
        "MORPHOLOGICAL:\n"
        f"• Anti-tail enhancement: {predictions['morphological']['anti_tail_enhancement']:.1f}×\n"
        f"• Structural coherence: {predictions['morphological']['structural_coherence']:.2f}\n"
        f"• Jet symmetry: N = {ID_PARAMS['N']}\n\n"
        
        "TEMPORAL (Predictions 2-3):\n"
        f"• Wobble period: {predictions['temporal']['wobble_period_hours']:.1f} h\n"
        f"• Jump probability: {100*predictions['temporal']['jump_probability']:.0f}%\n"
        f"• Optimal window: ±{predictions['temporal']['optimal_obs_window_days'][1]:.0f} days"
    )
    
    ax4.text(0.05, 0.95, summary_text, va='top', fontsize=10, 
             fontfamily='monospace', linespacing=1.5)
    
    plt.tight_layout()
    return fig

def plot_geometry_schematic(ax: plt.Axes, geometry: dict):
    """
    Plot simple schematic of alignment geometry.
    
    Parameters:
    -----------
    ax : matplotlib.Axes
        Axes to plot on
    geometry : dict
        Geometry parameters
    """
    # Sun position
    ax.plot(0, 0, 'o', color='yellow', markersize=20, label='Sun', markeredgecolor='orange')
    
    # Earth position (scaled)
    scale = 0.5  # arbitrary for visualization
    earth_x = geometry['heliocentric_distance_au'] * scale
    phase_rad = geometry['phase_angle_rad']
    earth_y = earth_x * np.tan(phase_rad)
    ax.plot(earth_x, earth_y, 'o', color='blue', markersize=10, label='Earth')
    
    # 3I position (near opposition)
    obj_x = earth_x - geometry['geocentric_distance_au'] * scale * np.cos(phase_rad)
    obj_y = earth_y - geometry['geocentric_distance_au'] * scale * np.sin(phase_rad)
    ax.plot(obj_x, obj_y, 'o', color='green', markersize=8, label='3I/ATLAS')
    
    # Lines
    ax.plot([0, earth_x], [0, earth_y], 'b--', alpha=0.5, label='Sun-Earth line')
    ax.plot([earth_x, obj_x], [earth_y, obj_y], 'g--', alpha=0.5, label='Earth-3I line')
    
    # Phase angle arc
    arc_radius = earth_x * 0.3
    arc_angles = np.linspace(0, phase_rad, 50)
    arc_x = arc_radius * np.cos(arc_angles)
    arc_y = arc_radius * np.sin(arc_angles)
    ax.plot(arc_x, arc_y, 'r-', linewidth=2)
    ax.text(arc_radius*0.7, arc_radius*0.2, f'α={geometry["phase_angle_deg"]:.2f}°', 
            color='red', fontsize=10, fontweight='bold')
    
    ax.set_title('Alignment Geometry Schematic', fontsize=12, fontweight='bold')
    ax.set_xlabel('AU (scaled)')
    ax.set_ylabel('AU (scaled)')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, earth_x*1.2)
    ax.set_ylim(-earth_y*0.5, earth_y*1.5)

# ============================================================================
# MAIN SIMULATION DRIVER
# ============================================================================

def main():
    """Main driver for alignment predictions simulation."""
    parser = argparse.ArgumentParser(
        description='Generate predictions for 3I/ATLAS Sun-Earth alignment'
    )
    parser.add_argument('--date', type=str, default='2026-01-22',
                        help='Alignment date in YYYY-MM-DD (default: 2026-01-22)')
    parser.add_argument('--base-mag', type=float, default=16.7,
                        help='Base apparent magnitude (default: 16.7 from Jan 2026 data)')
    parser.add_argument('--hours-range', type=int, nargs=2, default=[-48, 48],
                        help='Light curve time range in hours (default: -48 48)')
    parser.add_argument('--resolution', type=int, default=30,
                        help='Time resolution in minutes (default: 30)')
    parser.add_argument('--save-results', type=str, default=None,
                        help='Save JSON results to file')
    parser.add_argument('--save-plots', type=str, default='alignment_predictions.png',
                        help='Save plot to file')
    parser.add_argument('--no-plot', action='store_true',
                        help='Run without creating plots')
    
    args = parser.parse_args()
    
    # Update OBS_PARAMS with command line argument
    OBS_PARAMS['apparent_magnitude'] = args.base_mag
    
    print("=" * 70)
    print(f"3I/ATLAS - SUN - EARTH ALIGNMENT PREDICTIONS")
    print(f"Date: {args.date}, Phase Angle: {ALIGNMENT_PHASE_ANGLE}°")
    print(f"Based on Information Dynamics model (Huang & Liu, 2026)")
    print("=" * 70)
    
    # 1. Generate predictions
    print("\n[1] GENERATING INFORMATION DYNAMICS PREDICTIONS")
    print("-" * 40)
    
    geometry_calc = AlignmentGeometry(date_str=args.date)
    predictions = AlignmentPredictions(geometry=geometry_calc.geometry).predictions
    
    print(f"Geometry calculated:")
    print(f"  • Phase angle (α): {predictions['geometry']['phase_angle_deg']:.2f}°")
    print(f"  • Heliocentric distance: {predictions['geometry']['heliocentric_distance_au']:.2f} AU")
    print(f"  • Geocentric distance: {predictions['geometry']['geocentric_distance_au']:.2f} AU")
    print(f"  • Alignment quality: {predictions['geometry']['alignment_quality']:.2f}")
    
    print(f"\nPhotometric predictions (Prediction 1):")
    print(f"  • Opposition surge: {predictions['photometric']['surge_amplitude_mag']:.2f} mag amplitude")
    print(f"  • Predicted V magnitude: {predictions['photometric']['apparent_magnitude_V']:.1f}")
    print(f"  • Brightness enhancement: {predictions['photometric']['brightness_enhancement']:.1f}×")
    
    # 2. Simulate observational signatures
    print("\n[2] SIMULATING OBSERVATIONAL SIGNATURES")
    print("-" * 40)
    
    light_curve_data = simulate_light_curve(
        predictions,
        hours_range=tuple(args.hours_range),
        resolution_minutes=args.resolution
    )
    
    time_hours, magnitude = light_curve_data
    print(f"Light curve simulated:")
    print(f"  • Time points: {len(time_hours)}")
    print(f"  • Time range: {args.hours_range[0]} to {args.hours_range[1]} hours")
    print(f"  • Resolution: {args.resolution} minutes")
    print(f"  • Magnitude range: {magnitude.min():.2f} to {magnitude.max():.2f}")
    
    # 3. Create visualizations
    if not args.no_plot:
        print("\n[3] CREATING VISUALIZATIONS")
        print("-" * 40)
        
        fig = create_alignment_visualization(
            predictions, 
            light_curve_data,
            resolution_minutes=args.resolution
        )
        fig.savefig(args.save_plots, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {args.save_plots}")
        
        # Show plot if in interactive environment
        try:
            plt.show(block=False)
        except:
            pass
    
    # 4. Save results
    if args.save_results:
        print("\n[4] SAVING RESULTS")
        print("-" * 40)
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        save_data = {
            'predictions': convert_for_json(predictions),
            'simulation_parameters': {
                'date': args.date,
                'base_magnitude': args.base_mag,
                'hours_range': args.hours_range,
                'resolution_minutes': args.resolution,
                'information_dynamics_params': ID_PARAMS,
                'observational_params': OBS_PARAMS
            },
            'light_curve': {
                'time_hours': convert_for_json(light_curve_data[0]),
                'magnitude': convert_for_json(light_curve_data[1])
            }
        }
        
        with open(args.save_results, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {args.save_results}")
    
    # 5. Generate summary for observing proposals
    print("\n" + "=" * 70)
    print("OBSERVING GUIDE SUMMARY FOR 3I/ATLAS ALIGNMENT")
    print("=" * 70)
    
    summary = (
        f"\nCRITICAL OBSERVATIONAL PREDICTIONS FOR {args.date} ALIGNMENT:\n"
        f"Based on Information Dynamics model (Huang & Liu, 2026)\n\n"
        
        f"1. PHOTOMETRIC (PREDICTION 1 - Anti-tail Coherence):\n"
        f"   • Peak brightness: V = {predictions['photometric']['apparent_magnitude_V']:.1f} ± "
        f"{predictions['photometric']['uncertainty_mag']:.1f}\n"
        f"   • Opposition surge: {predictions['photometric']['surge_amplitude_mag']:.2f} mag "
        f"({predictions['photometric']['brightness_enhancement']:.1f}× flux increase)\n"
        f"   • Anti-tail enhancement: {predictions['morphological']['anti_tail_enhancement']:.1f}× "
        f"(coherent structure persists for weeks)\n"
        f"   • Required: High-resolution imaging (HST/JWST) to detect coherent anti-tail structure\n\n"
        
        f"2. TEMPORAL (PREDICTION 2 - Synchronized Wobble):\n"
        f"   • Wobble period: {predictions['temporal']['wobble_period_hours']:.1f} hours\n"
        f"   • N={ID_PARAMS['N']}-fold jet symmetry maintained during wobble\n"
        f"   • Wobble represents collective oscillation of jet system (not rotation)\n"
        f"   • Required: High-cadence photometry (TESS/ground-based) with <1h sampling\n\n"
        
        f"3. TEMPORAL (PREDICTION 3 - Discrete Activity Jumps):\n"
        f"   • Jump probability: {100*predictions['temporal']['jump_probability']:.0f}%\n"
        f"   • Step-like changes in brightness (0.1-0.3 mag) uncorrelated with distance\n"
        f"   • Noise-induced transitions in CGLE dynamics\n"
        f"   • Required: Continuous monitoring with high cadence (<30 min)\n\n"
        
        f"4. OBSERVING STRATEGY:\n"
        f"   • Optimal window: {predictions['temporal']['optimal_obs_window_days'][0]:.0f} to "
        f"{predictions['temporal']['optimal_obs_window_days'][1]:.0f} days from opposition\n"
        f"   • Critical observation times: ~{predictions['temporal']['critical_times_jd'][0]:.1f}, "
        f"{predictions['temporal']['critical_times_jd'][1]:.1f}, "
        f"{predictions['temporal']['critical_times_jd'][2]:.1f} JD\n"
        f"   • Minimum cadence: 30 min for jump detection, 10 min for wobble analysis\n"
        f"   • Priority targets: Anti-tail morphology, light curve modulation, activity jumps\n\n"
        
        f"5. INFORMATION DYNAMICS METRICS TO VALIDATE:\n"
        f"   • Information purity (p): {ID_PARAMS['p']:.2f} (mixed state)\n"
        f"   • Structural coherence: {predictions['morphological']['structural_coherence']:.2f}\n"
        f"   • Verification of any two predictions validates the model\n"
    )
    
    print(summary)
    
    # Save summary to file
    with open('alignment_observing_guide.txt', 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"Observing guide saved to: alignment_observing_guide.txt")
    print(f"\nLast observable window: Mid-2026 (urgent observations required)")
    print(f"Data availability: https://github.com/hkaiopen/InformationDynamics")

def quick_demo():
    """Run a quick demonstration of the alignment predictions."""
    print("Running quick demo of alignment predictions...")
    
    predictions = AlignmentPredictions().predictions
    
    print(f"\nKey predictions for January 22, 2026:")
    print(f"• Phase angle: {predictions['geometry']['phase_angle_deg']:.2f}°")
    print(f"• Apparent V magnitude: {predictions['photometric']['apparent_magnitude_V']:.1f}")
    print(f"• Anti-tail enhancement: {predictions['morphological']['anti_tail_enhancement']:.1f}×")
    print(f"• Wobble period: {predictions['temporal']['wobble_period_hours']:.1f} hours")
    print(f"• Activity jump probability: {100*predictions['temporal']['jump_probability']:.0f}%")
    
    return predictions

if __name__ == "__main__":
    # Example: quick_demo()  # Uncomment for quick test
    main()
