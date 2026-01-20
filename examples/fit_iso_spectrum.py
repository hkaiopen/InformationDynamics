# examples/fit_iso_spectrum.py
"""
Information Dynamics: A Unified Predictive Framework for Interstellar Objects

Core script that fits ISO data to the Information Dynamics model,
calculates information purity (p) values, and generates the Information Purity Spectrum figure.

Code: https://github.com/hkaiopen/InformationDynamics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

def create_parameter_table():
    """
    Create parameter table matching the paper.
    
    Returns:
        DataFrame with calibrated parameters for all three ISOs
    """
    data = {
        'Object': ["1I/'Oumuamua", "2I/Borisov", "3I/ATLAS"],
        'γ (10⁻⁵ s⁻¹)': ["0.12 ± 0.05", "21 ± 3", "3.8 ± 0.5"],
        'ω (10⁻⁴ rad/s)': ["2.18 ± 0.10", "1.75 ± 0.15", "1.08 ± 0.02"],
        'ε (10⁻⁴)': ["0.059 ± 0.005", "0.21 ± 0.03", "0.078 ± 0.007"],
        'F_thermal': ["0.0 ± 0.0", "0.85 ± 0.05", "0.12 ± 0.05"],
        'N': [1, 1, 3],
        'p': [0.83, 0.09, 0.17]
    }
    
    return pd.DataFrame(data)

def calculate_p_from_parameters():
    """
    Calculate p values from parameters and verify consistency.
    
    p = ε / (γ + ε)
    """
    # Parameters in base units (converted from table units)
    parameters = {
        "1I/'Oumuamua": {"gamma": 0.12e-5, "epsilon": 0.059e-4},
        "2I/Borisov": {"gamma": 21e-5, "epsilon": 0.21e-4},
        "3I/ATLAS": {"gamma": 3.8e-5, "epsilon": 0.078e-4}
    }
    
    print("Calculating information purity (p) values:")
    print("-" * 50)
    
    results = {}
    for name, params in parameters.items():
        gamma = params["gamma"]
        epsilon = params["epsilon"]
        p = epsilon / (gamma + epsilon)
        
        # Paper values for comparison
        paper_p = {"1I/'Oumuamua": 0.83, "2I/Borisov": 0.09, "3I/ATLAS": 0.17}[name]
        
        status = "✓" if abs(p - paper_p) < 0.01 else "✗"
        print(f"{name}: p = {p:.3f} (paper: {paper_p:.2f}) {status}")
        
        results[name] = {
            "gamma": gamma,
            "epsilon": epsilon,
            "p_calculated": p,
            "p_paper": paper_p,
            "consistent": abs(p - paper_p) < 0.01
        }
    
    return results

def plot_spectrum():
    """
    Generate the Information Purity Spectrum figure.
    
    This creates Figure 1 from the paper.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data
    objects = ["1I/'Oumuamua", "2I/Borisov", "3I/ATLAS"]
    p_values = [0.83, 0.09, 0.17]
    colors = ['red', 'blue', 'green']
    markers = ['s', 'o', '^']
    sizes = [300, 300, 500]  # Larger size for 3I (triple jets)
    
    for i, (obj, p, color, marker, size) in enumerate(zip(objects, p_values, colors, markers, sizes)):
        ax.scatter(p, 0, s=size, c=color, marker=marker, 
                  label=obj, edgecolors='black', linewidth=1.5, alpha=0.8, zorder=5)
        
        # Add error bars (estimated uncertainty)
        p_err = 0.03
        ax.errorbar(p, 0, xerr=p_err, fmt='none', 
                   ecolor=color, capsize=5, alpha=0.7, zorder=4)
    
    # Classification regions
    ax.axvspan(0, 0.3, alpha=0.2, color='blue', label='Thermally-Driven (low p)')
    ax.axvspan(0.3, 0.7, alpha=0.2, color='green', label='Mixed State (intermediate p)')
    ax.axvspan(0.7, 1.0, alpha=0.2, color='red', label='Information-Driven (high p)')
    
    # Labels and formatting
    ax.set_xlabel('Information Purity (p)', fontsize=12, fontweight='bold')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_title('Information Purity Spectrum for Interstellar Objects', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Legend (outside plot)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add explanatory text
    explanatory_text = """Classification based on p = ε/(γ+ε):
• High p (e.g., 1I): Information-driven, coherent macroscopic order
• Low p (e.g., 2I): Thermally-driven, standard cometary behavior
• Intermediate p (e.g., 3I): Mixed state, structured jets with dust deficit"""
    
    ax.text(0.02, 0.98, explanatory_text, 
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
            verticalalignment='top')
    
    plt.tight_layout()
    return fig, ax

def simulate_fitting_process():
    """
    Simulate the parameter fitting process.
    
    Demonstrates how the model parameters are obtained from observational data.
    """
    print("\nSimulating parameter fitting process...")
    print("-" * 50)
    
    # Generate synthetic observational data for 3I/ATLAS
    np.random.seed(42)  # For reproducibility
    n_points = 50
    t = np.linspace(0, 100, n_points)  # Time in days
    
    # True parameters (calibrated values for 3I/ATLAS)
    true_params = {
        'gamma': 3.8e-5,
        'omega': 1.08e-4,
        'epsilon': 0.078e-4,
        'F_thermal': 0.12
    }
    
    gamma, omega, epsilon, F_thermal = true_params.values()
    
    # Generate model curve (simplified CGLE solution)
    # This represents observable activity (e.g., brightness)
    model_curve = (np.exp(-gamma * t) * np.cos(omega * t) +
                   epsilon * (1 - np.exp(-gamma * t)) * np.sin(omega * t) +
                   F_thermal * (1 - np.exp(-gamma * t)))
    
    # Add realistic observational noise
    noise_amplitude = 0.1
    noise = noise_amplitude * np.random.randn(n_points)
    observations = model_curve + noise
    
    # Perform fitting to recover parameters
    def fitting_model(t, gamma, omega, epsilon, F_thermal):
        return (np.exp(-gamma * t) * np.cos(omega * t) +
                epsilon * (1 - np.exp(-gamma * t)) * np.sin(omega * t) +
                F_thermal * (1 - np.exp(-gamma * t)))
    
    try:
        # Initial guess (order of magnitude estimates)
        initial_guess = [1e-5, 1e-4, 1e-5, 0.1]
        
        # Perform the fit
        popt, pcov = curve_fit(fitting_model, t, observations, p0=initial_guess)
        perr = np.sqrt(np.diag(pcov))
        
        print("Fitting results for 3I/ATLAS:")
        print(f"  γ: {popt[0]:.2e} ± {perr[0]:.2e} (true: {gamma:.2e})")
        print(f"  ω: {popt[1]:.2e} ± {perr[1]:.2e} (true: {omega:.2e})")
        print(f"  ε: {popt[2]:.2e} ± {perr[2]:.2e} (true: {epsilon:.2e})")
        print(f"  F_thermal: {popt[3]:.3f} ± {perr[3]:.3f} (true: {F_thermal:.3f})")
        
        # Calculate p from fitted parameters
        p_fitted = popt[2] / (popt[0] + popt[2])
        print(f"\n  Calculated p: {p_fitted:.2f} (paper value: 0.17)")
        
        # Check consistency
        if abs(p_fitted - 0.17) < 0.05:
            print("  ✓ Fitted p value consistent with paper")
        else:
            print("  ⚠ Fitted p value differs from paper value")
        
        return {
            "fitted_params": popt,
            "param_errors": perr,
            "p_fitted": p_fitted,
            "success": True
        }
        
    except Exception as e:
        print(f"Fitting failed: {e}")
        return {"success": False, "error": str(e)}

def generate_summary_statistics():
    """
    Generate summary statistics and insights.
    """
    print("\n" + "="*70)
    print("MODEL INSIGHTS")
    print("="*70)
    
    insights = [
        "1. Information purity p = ε/(γ+ε) unifies all three ISOs",
        "2. 1I/'Oumuamua (p=0.83): High purity → macroscopic order, no dust",
        "3. 2I/Borisov (p=0.09): Low purity → thermal driving, cometary",
        "4. 3I/ATLAS (p=0.17): Mixed state → structured jets with dust deficit",
        "5. The p-spectrum predicts dust production: high p = low dust",
        "6. 3I's N=3 symmetry emerges naturally in mixed state",
    ]
    
    for insight in insights:
        print(f"  {insight}")
    
    print("\nKey predictions for 3I/ATLAS (testable before mid-2026):")
    predictions = [
        "1. Anti-tail coherence: Narrow structure persists for weeks",
        "2. Synchronized wobble: Jets wobble in phase (120° separation)",
        "3. Discrete activity jumps: Step-like changes uncorrelated with distance",
    ]
    
    for pred in predictions:
        print(f"  {pred}")

def main():
    """
    Main function for fitting ISO spectrum.
    """
    print("="*70)
    print("INFORMATION DYNAMICS: ISO SPECTRUM FITTING")
    print("="*70)
    
    # Step 1: Display parameter table
    print("\n1. CALIBRATED PARAMETERS:")
    print("-" * 70)
    df = create_parameter_table()
    print(df.to_string(index=False))
    
    # Step 2: Calculate and verify p values
    print("\n2. INFORMATION PURITY CALCULATIONS:")
    results = calculate_p_from_parameters()
    
    # Step 3: Generate spectrum plot
    print("\n3. GENERATING SPECTRUM PLOT...")
    fig, ax = plot_spectrum()
    fig.savefig('information_purity_spectrum.png', dpi=300, bbox_inches='tight')
    print("   Figure saved as 'information_purity_spectrum.png'")
    
    # Step 4: Demonstrate fitting process
    print("\n4. DEMONSTRATING PARAMETER FITTING:")
    fitting_results = simulate_fitting_process()
    
    # Step 5: Generate insights
    generate_summary_statistics()
    
    # Step 6: Instructions for use
    print("\n" + "="*70)
    print("USAGE INSTRUCTIONS")
    print("="*70)
    print("\nTo use this framework for new ISOs:")
    print("  1. Collect observational data (brightness, position, etc.)")
    print("  2. Use fit_iso_parameters.py to fit the CGLE model")
    print("  3. Calculate p = ε/(γ+ε) to classify the object")
    print("  4. Compare to the spectrum to predict behavior")
    print("\nFor 3I/ATLAS predictions:")
    print("  Run: python examples/predict_jan22_alignment.py")
    
    return df, results, fitting_results

if __name__ == "__main__":
    try:
        df, results, fitting_results = main()
        
        # Save numerical results to file
        with open("spectrum_fitting_results.txt", "w", encoding="utf-8") as f:
            f.write("Information Dynamics: ISO Spectrum Fitting Results\n")
            f.write("="*60 + "\n\n")
            f.write(df.to_string(index=False))
            f.write("\n\nGenerated: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
        
        print(f"\nResults saved to: spectrum_fitting_results.txt")
        
    except Exception as e:
        print(f"\nError during execution: {e}")
        print("Please ensure all dependencies are installed:")
        print("  pip install numpy pandas matplotlib scipy")
