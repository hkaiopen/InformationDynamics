"""
bayesian_corner_plot.py

Generates Figure: Bayesian Posterior Corner Plot
Requires: matplotlib, numpy
"""
import numpy as np
import matplotlib.pyplot as plt
from bayesian_parameter_inference import BayesianIDInference

def plot_bayesian_corner(save_path='bayesian_corner.png'):
    # Run the inference to get the samples
    print("Running MCMC to generate posterior samples (this may take ~1 second)...")
    infer = BayesianIDInference()
    result = infer.run_mcmc(n_steps=20000, burn_in=5000)
    samples = result['samples']  # Shape (15000, 4)

    # Compute derived p
    gammas = 10**samples[:, 0]
    omegas = 10**samples[:, 1]
    epsilons = 10**samples[:, 2]
    Fs = samples[:, 3]
    ps = epsilons / (gammas + epsilons)

    # Parameter names and ranges
    params = {
        'log10(gamma)': (samples[:, 0], r'$\log_{10}(\gamma)$'),
        'log10(omega)': (samples[:, 1], r'$\log_{10}(\omega)$'),
        'log10(epsilon)': (samples[:, 2], r'$\log_{10}(\varepsilon)$'),
        'F_thermal': (Fs, r'$F_{\mathrm{thermal}}$'),
        'p': (ps, r'$p$')
    }
    keys = list(params.keys())
    n = len(keys)

    fig, axes = plt.subplots(n, n, figsize=(12, 12))
    
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                # Diagonal: 1D Histogram
                data, label = params[keys[i]]
                ax.hist(data, bins=30, density=True, color='steelblue', alpha=0.7)
                ax.set_yticks([])
                if i == n-1:
                    ax.set_xlabel(label)
                else:
                    ax.set_xticks([])
            elif i > j:
                # Lower triangle: 2D Scatter/Contour
                data_y, _ = params[keys[i]]
                data_x, _ = params[keys[j]]
                ax.scatter(data_x[::10], data_y[::10], s=1, alpha=0.3, color='gray')
                ax.set_xlim(None)
                ax.set_ylim(None)
                if i == n-1:
                    ax.set_xlabel(params[keys[j]][1])
                else:
                    ax.set_xticks([])
                if j == 0:
                    ax.set_ylabel(params[keys[i]][1])
                else:
                    ax.set_yticks([])
            else:
                # Upper triangle: empty
                ax.axis('off')

    plt.suptitle('Bayesian Parameter Inference: 1D & 2D Posterior Distributions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Bayesian corner plot saved to: {save_path}")

if __name__ == "__main__":
    plot_bayesian_corner()