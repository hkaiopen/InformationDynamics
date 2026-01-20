# iso_parameter_fitting.py
"""
Parameter Fitting for Interstellar Objects using Information Dynamics Framework

This module implements parameter fitting for the three interstellar objects:
1I/'Oumuamua, 2I/Borisov, and 3I/ATLAS, using the Information Dynamics model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional
import warnings

# Import model from information_dynamics_model.py in the same folder
from information_dynamics_model import InformationDynamicsModel

class ISOParameterSet:
    """Container for interstellar object parameters with uncertainties."""
    
    def __init__(self, name: str, gamma: float, omega: float, epsilon: float,
                 F_thermal: float, symmetry: int, gamma_err: float = 0.0,
                 omega_err: float = 0.0, epsilon_err: float = 0.0,
                 F_thermal_err: float = 0.0):
        """
        Initialize ISO parameter set.
        
        Parameters scaled to match paper units:
        - gamma: in 10^-5 s^-1
        - omega: in 10^-4 rad/s
        - epsilon: in 10^-4
        """
        self.name = name
        self.gamma = float(gamma)  # Store in paper units
        self.omega = float(omega)
        self.epsilon = float(epsilon)
        self.F_thermal = float(F_thermal)
        self.symmetry = int(symmetry)
        
        # Uncertainties
        self.gamma_err = float(gamma_err)
        self.omega_err = float(omega_err)
        self.epsilon_err = float(epsilon_err)
        self.F_thermal_err = float(F_thermal_err)
    
    @property
    def gamma_si(self) -> float:
        """Convert gamma to SI units (s^-1)."""
        return self.gamma * 1e-5
    
    @property
    def omega_si(self) -> float:
        """Convert omega to SI units (rad/s)."""
        return self.omega * 1e-4
    
    @property
    def epsilon_si(self) -> float:
        """Convert epsilon to SI units (s^-1)."""
        return self.epsilon * 1e-4
    
    def to_model(self, noise_amplitude: float = 0.0) -> InformationDynamicsModel:
        """Create InformationDynamicsModel instance from these parameters."""
        return InformationDynamicsModel(
            gamma=self.gamma_si,
            omega=self.omega_si,
            epsilon=self.epsilon_si,
            F_thermal=self.F_thermal,
            noise_amplitude=noise_amplitude
        )

class ISOParameterDatabase:
    """Database of interstellar object parameters."""
    
    def __init__(self):
        self.objects: Dict[str, ISOParameterSet] = {}
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize with known interstellar objects."""
        # 1I/'Oumuamua
        self.objects['1I'] = ISOParameterSet(
            name="1I/'Oumuamua",
            gamma=0.12,    # 10^-5 s^-1
            omega=2.18,    # 10^-4 rad/s
            epsilon=0.059, # 10^-4
            F_thermal=0.0,
            symmetry=1,
            gamma_err=0.05,
            omega_err=0.10,
            epsilon_err=0.005
        )
        
        # 2I/Borisov
        self.objects['2I'] = ISOParameterSet(
            name="2I/Borisov",
            gamma=21.0,
            omega=1.75,
            epsilon=0.21,
            F_thermal=0.85,
            symmetry=1,
            gamma_err=3.0,
            omega_err=0.15,
            epsilon_err=0.03,
            F_thermal_err=0.05
        )
        
        # 3I/ATLAS
        self.objects['3I'] = ISOParameterSet(
            name="3I/ATLAS",
            gamma=3.8,
            omega=1.08,
            epsilon=0.078,
            F_thermal=0.12,
            symmetry=3,
            gamma_err=0.5,
            omega_err=0.02,
            epsilon_err=0.007,
            F_thermal_err=0.05
        )
    
    def get_object(self, identifier: str) -> ISOParameterSet:
        """Get parameter set by identifier."""
        if identifier not in self.objects:
            raise KeyError(f"Unknown object identifier: {identifier}")
        return self.objects[identifier]
    
    def get_all_objects(self) -> List[ISOParameterSet]:
        """Get all objects in the database."""
        return list(self.objects.values())
    
    def create_models(self, noise_amplitude: float = 0.0) -> Dict[str, InformationDynamicsModel]:
        """Create InformationDynamicsModel instances for all objects."""
        models = {}
        for key, params in self.objects.items():
            models[key] = params.to_model(noise_amplitude)
        return models

def fit_observational_data(times: np.ndarray, observations: np.ndarray,
                          initial_guess: Optional[Tuple] = None,
                          bounds: Optional[Tuple] = None) -> Dict:
    """
    Fit observational data to the Information Dynamics model.
    
    Parameters:
        times: Observation times in seconds
        observations: Observed quantities (e.g., acceleration)
        initial_guess: Initial parameter guess [gamma, omega, epsilon, F_thermal]
        bounds: Parameter bounds [(lower), (upper)]
        
    Returns:
        Dictionary with fitting results
    """
    try:
        from scipy.optimize import curve_fit
        has_scipy = True
    except ImportError:
        has_scipy = False
        warnings.warn("scipy not available, using simplified fitting")
    
    def model_function(t, gamma, omega, epsilon, F_thermal):
        """Simplified model for fitting."""
        # Damped oscillation with nonlinear correction
        damping = np.exp(-gamma * t)
        oscillation = np.cos(omega * t)
        nonlinear = epsilon * (1 - damping) * np.sin(omega * t)
        thermal = F_thermal * (1 - damping)
        
        return damping * oscillation + nonlinear + thermal
    
    if initial_guess is None:
        initial_guess = (1e-6, 1e-4, 1e-6, 0.1)
    
    if bounds is None:
        bounds = ([0, 0, 0, 0], [1e-3, 1e-2, 1e-3, 2.0])
    
    if has_scipy:
        try:
            popt, pcov = curve_fit(
                model_function,
                times,
                observations,
                p0=initial_guess,
                bounds=bounds,
                maxfev=10000
            )
            
            perr = np.sqrt(np.diag(pcov))
            
            return {
                'success': True,
                'parameters': {
                    'gamma': popt[0],
                    'omega': popt[1],
                    'epsilon': popt[2],
                    'F_thermal': popt[3]
                },
                'uncertainties': {
                    'gamma': perr[0],
                    'omega': perr[1],
                    'epsilon': perr[2],
                    'F_thermal': perr[3]
                },
                'covariance': pcov
            }
        except Exception as e:
            return {'success': False, 'error': str(e), 'parameters': None}
    else:
        # Simple grid search if scipy not available
        warnings.warn("Using simplified grid search (scipy not available)")
        best_params = list(initial_guess)
        best_error = np.inf
        
        # Simple grid around initial guess
        for gamma in np.linspace(initial_guess[0]*0.5, initial_guess[0]*1.5, 3):
            for omega in np.linspace(initial_guess[1]*0.5, initial_guess[1]*1.5, 3):
                for epsilon in np.linspace(initial_guess[2]*0.5, initial_guess[2]*1.5, 3):
                    for F_thermal in np.linspace(initial_guess[3]*0.5, initial_guess[3]*1.5, 3):
                        pred = model_function(times, gamma, omega, epsilon, F_thermal)
                        error = np.sum((pred - observations) ** 2)
                        
                        if error < best_error:
                            best_error = error
                            best_params = [gamma, omega, epsilon, F_thermal]
        
        return {
            'success': True,
            'parameters': {
                'gamma': best_params[0],
                'omega': best_params[1],
                'epsilon': best_params[2],
                'F_thermal': best_params[3]
            },
            'uncertainties': None,
            'warning': 'Grid search used (install scipy for better fitting)'
        }

def generate_parameter_table(database: ISOParameterDatabase) -> pd.DataFrame:
    """
    Generate parameter table similar to Table 1 in the paper.
    
    Returns:
        Pandas DataFrame with all parameters
    """
    rows = []
    
    for key, params in database.objects.items():
        model = params.to_model()
        
        rows.append({
            'Object': params.name,
            'γ (10⁻⁵ s⁻¹)': f"{params.gamma:.2f} ± {params.gamma_err:.2f}",
            'ω (10⁻⁴ rad/s)': f"{params.omega:.2f} ± {params.omega_err:.2f}",
            'ε (10⁻⁴)': f"{params.epsilon:.3f} ± {params.epsilon_err:.3f}",
            'F_thermal': f"{params.F_thermal:.2f} ± {params.F_thermal_err:.2f}",
            'N': params.symmetry,
            'p': f"{model.information_purity:.2f}",
            'State': model.classify_state()
        })
    
    df = pd.DataFrame(rows)
    df = df[['Object', 'γ (10⁻⁵ s⁻¹)', 'ω (10⁻⁴ rad/s)', 'ε (10⁻⁴)', 
             'F_thermal', 'N', 'p', 'State']]
    
    return df

def plot_parameter_space(database: ISOParameterDatabase, 
                        filename: str = 'iso_parameter_space.png'):
    """
    Create visualization of ISO parameter space.
    
    Parameters:
        database: ISOParameterDatabase instance
        filename: Output filename for the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Color coding for objects
    colors = {'1I': '#d62728', '2I': '#1f77b4', '3I': '#2ca02c'}
    markers = {'1I': 's', '2I': 'o', '3I': '^'}
    
    # Panel 1: Information purity spectrum
    ax1 = axes[0]
    
    for key, params in database.objects.items():
        model = params.to_model()
        p = model.information_purity
        
        ax1.scatter(p, 0, s=300, c=colors[key], marker=markers[key],
                   label=params.name, edgecolor='black', linewidth=1.5, zorder=5)
    
    # Classification regions
    ax1.axvspan(0, 0.3, alpha=0.2, color='blue', label='Thermal')
    ax1.axvspan(0.3, 0.7, alpha=0.2, color='green', label='Mixed')
    ax1.axvspan(0.7, 1.0, alpha=0.2, color='red', label='Information')
    
    ax1.set_xlabel('Information Purity (p)', fontsize=12, fontweight='bold')
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_yticks([])
    ax1.set_title('ISO Classification Spectrum', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Panel 2: Gamma vs Epsilon parameter space
    ax2 = axes[1]
    
    for key, params in database.objects.items():
        ax2.errorbar(params.gamma, params.epsilon,
                    xerr=params.gamma_err,
                    yerr=params.epsilon_err,
                    fmt=markers[key], color=colors[key],
                    label=params.name, capsize=5, markersize=10,
                    linewidth=1.5)
    
    # Add lines of constant information purity
    gamma_range = np.linspace(0.1, 25, 100)
    for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
        epsilon_target = (p / (1 - p)) * gamma_range
        ax2.plot(gamma_range, epsilon_target, 'k:', alpha=0.3, linewidth=1)
        if p == 0.5:
            ax2.text(gamma_range[-1], epsilon_target[-1], f'p={p}', 
                    fontsize=9, va='bottom', ha='right')
    
    ax2.set_xlabel('γ (10⁻⁵ s⁻¹)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('ε (10⁻⁴)', fontsize=12, fontweight='bold')
    ax2.set_title('Parameter Space: γ vs ε', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if filename:
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {filename}")
    
    return fig, axes

def generate_predictions_for_3I(database: ISOParameterDatabase) -> Dict[str, Dict]:
    """
    Generate testable predictions for 3I/ATLAS.
    
    Returns:
        Dictionary containing quantitative predictions
    """
    params_3i = database.get_object('3I')
    model_3i = params_3i.to_model()
    period_info = model_3i.predict_jet_period()
    
    predictions = {
        'anti_tail_coherence': {
            'prediction': 'Sunward anti-tail maintains narrow, coherent structure',
            'timescale': '> 2 weeks',
            'mechanism': 'ε-driven coherence resists solar wind dispersion',
            'observable': 'Tail width and morphology',
            'verification': 'Width < 20% of typical comet tail, persists >14 days'
        },
        'synchronized_wobble': {
            'prediction': 'Jets wobble synchronously with preserved 120° separation',
            'period': f"{period_info['period_days']:.2f} days",
            'frequency': f"{period_info['frequency_hz']:.2e} Hz",
            'mechanism': 'ω-driven natural oscillation',
            'observable': 'Jet direction time series',
            'verification': f"Detect ~{period_info['period_days']:.1f}-day period in jet motion"
        },
        'discrete_activity_jumps': {
            'prediction': 'Step-like changes in brightness and acceleration',
            'correlation': 'Uncorrelated with heliocentric distance (|r| < 0.1)',
            'mechanism': 'Noise-induced state transitions (η(t) term)',
            'observable': 'Light curves and astrometric residuals',
            'verification': '≥2 jumps of Δmag > 0.3, uncorrelated with distance'
        }
    }
    
    return predictions

def main():
    """Main analysis function."""
    print("=" * 70)
    print("INFORMATION DYNAMICS: ISO PARAMETER ANALYSIS")
    print("=" * 70)
    
    # Initialize database
    print("\n1. Loading ISO parameter database...")
    database = ISOParameterDatabase()
    
    # Generate parameter table
    print("\n2. Generating parameter table...")
    table = generate_parameter_table(database)
    
    print("\n" + "-" * 80)
    print("PARAMETER TABLE")
    print("-" * 80)
    print(table.to_string(index=False))
    
    # Create models
    print("\n3. Creating InformationDynamicsModel instances...")
    models = database.create_models()
    
    # Verify information purity calculations
    print("\n4. Calculating information purities:")
    for key, model in models.items():
        params = database.objects[key]
        print(f"   {params.name}: p = {model.information_purity:.2f} ({model.classify_state()})")
    
    # Generate predictions for 3I/ATLAS
    print("\n5. Generating predictions for 3I/ATLAS...")
    predictions = generate_predictions_for_3I(database)
    
    print("\n" + "=" * 70)
    print("TESTABLE PREDICTIONS FOR 3I/ATLAS")
    print("=" * 70)
    
    for pred_key, pred_info in predictions.items():
        print(f"\n{pred_key.replace('_', ' ').title()}:")
        for info_key, info_value in pred_info.items():
            print(f"  {info_key}: {info_value}")
    
    # Plot parameter space
    print("\n6. Creating visualization...")
    fig, axes = plot_parameter_space(database, 'iso_parameter_space.png')
    
    print("\n" + "=" * 70)
    print("FALSIFIABILITY CONDITIONS")
    print("=" * 70)
    print("1. 3I/ATLAS shows strong micro-dust scattering (contradicts p=0.17)")
    print("2. None of the three predictions are observed with adequate monitoring")
    print("3. Future ISOs fall outside established parameter space")
    print("4. Observables show strong correlation with distance (contradicts jump prediction)")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    
    return database, models, table, predictions

if __name__ == "__main__":
    try:
        database, models, table, predictions = main()
        
        # Optional: Save table to CSV
        table.to_csv('iso_parameters.csv', index=False)
        print("Parameter table saved to 'iso_parameters.csv'")
        
        # Show plot if in interactive environment
        import sys
        if 'ipykernel' in sys.modules or 'IPython' in sys.modules:
            plt.show()
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()