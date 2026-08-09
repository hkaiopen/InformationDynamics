"""
chemical_composition.py
"""
import numpy as np
import matplotlib.pyplot as plt

def plot_chemical_comparison(save_path='chemical_composition.png'):
    species = ['CO₂/H₂O', 'CO/H₂O', 'CH₄/H₂O', 'HDO/H₂O']
    primordial = [0.500, 0.120, 0.040, 0.008]
    ss_comet = [0.150, 0.100, 0.020, 0.0003]
    predicted = [0.691, 0.118, 0.038, 0.007]
    observed = [0.800, 0.150, 0.050, 0.0095]

    x = np.arange(len(species))
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width*1.5, primordial, width, label='Primordial')
    rects2 = ax.bar(x - width/2, ss_comet, width, label='Solar System Comet')
    rects3 = ax.bar(x + width/2, predicted, width, label='Predicted (p=0.17)')
    rects4 = ax.bar(x + width*1.5, observed, width, label='Observed (JWST)')

    ax.set_ylabel('Molar Ratio')
    ax.set_title('Chemical Composition: Model Prediction vs Observation')
    ax.set_xticks(x)
    ax.set_xticklabels(species)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Chemical composition plot saved to: {save_path}")

if __name__ == "__main__":
    plot_chemical_comparison()