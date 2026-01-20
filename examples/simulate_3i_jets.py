# simulate_3i_jets.py
"""
Information Dynamics: A Unified Predictive Framework for Interstellar Objects
Script to generate N=3 symmetric jet patterns for 3I/ATLAS.

Code: https://github.com/hkaiopen/InformationDynamics
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
import matplotlib.animation as animation

class JetSimulator:
    """
    Simulates N=3 symmetric jets for 3I/ATLAS.
    """
    
    def __init__(self):
        """Initialize parameters for 3I/ATLAS."""
        self.gamma = 3.8e-5      # γ
        self.omega = 1.08e-4     # ω
        self.epsilon = 0.078e-4  # ε
        self.F_thermal = 0.12
        self.N = 3  # Triple symmetry
        
        # Derived quantities
        self.p = self.epsilon / (self.gamma + self.epsilon)
        self.wobble_period = (2 * np.pi / self.omega) / 86400  # days
        
    def calculate_order_parameter(self, time: float, jet_index: int) -> complex:
        """
        Calculate order parameter Ψ for a specific jet at given time.
        
        Ψ_k(t) = A * exp(-γt) * exp(i(ωt + φ_k))
        where φ_k = 2πk/N defines the symmetry.
        """
        A = 1.0 * np.exp(-self.gamma * time)
        base_phase = self.omega * time
        phi_k = 2 * np.pi * jet_index / self.N
        
        # Add nonlinear effects
        nonlinear = self.epsilon * A**2
        
        Psi = A * np.exp(1j * (base_phase + phi_k + nonlinear))
        
        # Add small observational noise
        noise = 0.05 * (np.random.randn() + 1j * np.random.randn())
        return Psi + noise
    
    def simulate_jet_positions(self, time: float) -> np.ndarray:
        """
        Calculate jet positions at given time.
        
        Returns array of (x, y) positions for N jets.
        """
        positions = []
        
        # Base angles (120° separation)
        base_angles = np.array([0, 2*np.pi/3, 4*np.pi/3])
        
        # Add wobble (synchronized)
        wobble_amplitude = 0.2  # radians
        wobble_phase = np.sin(self.omega * time) * wobble_amplitude
        
        # All jets wobble in phase
        current_angles = base_angles + wobble_phase
        
        jet_length = 1.0
        
        for angle in current_angles:
            x = jet_length * np.cos(angle)
            y = jet_length * np.sin(angle)
            positions.append((x, y))
        
        return np.array(positions)
    
    def create_static_visualization(self):
        """Create static visualization of N=3 jet pattern."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Panel 1: Jet geometry
        ax1.set_xlim(-1.5, 1.5)
        ax1.set_ylim(-1.5, 1.5)
        ax1.set_aspect('equal')
        ax1.set_title('N=3 Symmetric Jet Pattern\n3I/ATLAS', 
                     fontsize=12, fontweight='bold')
        
        # Nucleus
        nucleus = Circle((0, 0), 0.2, color='gray', alpha=0.8)
        ax1.add_patch(nucleus)
        ax1.text(0, 0, '3I', fontsize=10, ha='center', va='center', 
                fontweight='bold', color='white')
        
        # Jets at time=0
        positions = self.simulate_jet_positions(0)
        colors = ['red', 'green', 'blue']
        
        for i, (x, y) in enumerate(positions):
            # Jet cone
            jet_angle = np.degrees(np.arctan2(y, x))
            jet = Wedge((0, 0), 1.0, jet_angle-15, jet_angle+15, 
                       width=0.7, color=colors[i], alpha=0.4)
            ax1.add_patch(jet)
            
            # Jet direction line
            ax1.plot([0, x*0.9], [0, y*0.9], color=colors[i], linewidth=2)
            
            # Label
            label_x = 1.2 * np.cos(jet_angle * np.pi/180)
            label_y = 1.2 * np.sin(jet_angle * np.pi/180)
            ax1.text(label_x, label_y, f'Jet {i+1}', 
                    ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Add angle markers
        for angle in [0, 120, 240]:
            rad = angle * np.pi/180
            x = 1.1 * np.cos(rad)
            y = 1.1 * np.sin(rad)
            ax1.text(x, y, f'{angle}°', ha='center', va='center', fontsize=8,
                    bbox=dict(boxstyle="circle,pad=0.2", facecolor="white", alpha=0.8))
        
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_xlabel('X', fontsize=10)
        ax1.set_ylabel('Y', fontsize=10)
        
        # Panel 2: Time evolution
        ax2.set_title('Jet Wobble Simulation', 
                     fontsize=12, fontweight='bold')
        
        # Simulate over time
        time_points = np.linspace(0, 100, 500)
        wobble_angles = []
        
        for t in time_points:
            wobble = np.sin(self.omega * t) * 0.2
            wobble_angles.append(np.degrees(wobble))
        
        ax2.plot(time_points, wobble_angles, 'b-', linewidth=2)
        ax2.set_xlabel('Time (s)', fontsize=10)
        ax2.set_ylabel('Wobble Angle (degrees)', fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        # Add period annotation
        ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax2.text(0.5, 0.95, f'Period: {self.wobble_period:.2f} days', 
                transform=ax2.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        return fig, (ax1, ax2)
    
    def create_animated_wobble(self):
        """Create animation of synchronized jet wobble."""
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.set_title('Synchronized Jet Wobble - 3I/ATLAS', 
                    fontsize=14, fontweight='bold')
        
        # Nucleus
        nucleus = Circle((0, 0), 0.15, color='gray', alpha=0.8)
        ax.add_patch(nucleus)
        
        # Initialize jet lines
        jet_lines = []
        for i in range(3):
            line, = ax.plot([], [], '-', linewidth=3, alpha=0.7)
            jet_lines.append(line)
        
        # Time text
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white"))
        
        def init():
            """Initialize animation."""
            for line in jet_lines:
                line.set_data([], [])
            time_text.set_text('')
            return jet_lines + [time_text]
        
        def animate(frame):
            """Update animation frame."""
            t = frame * 0.1  # Time in seconds
            
            # Get current jet positions
            positions = self.simulate_jet_positions(t)
            
            # Update jet lines
            for i, line in enumerate(jet_lines):
                x, y = positions[i]
                line.set_data([0, x], [0, y])
                line.set_color(['red', 'green', 'blue'][i])
            
            # Update time text
            time_text.set_text(f'Time: {t:.1f} s\nPeriod: {self.wobble_period:.2f} days')
            
            return jet_lines + [time_text]
        
        # Create animation
        ani = animation.FuncAnimation(fig, animate, frames=200, 
                                     init_func=init, blit=True, interval=50)
        
        # Add explanation
        explanation = (
            "Key Predictions:\n"
            "• Jets wobble in phase (synchronously)\n"
            "• 120° separation maintained\n"
            f"• Period: {self.wobble_period:.2f} days\n"
            f"• p = {self.p:.2f} (mixed state)"
        )
        ax.text(-1.4, -1.4, explanation, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))
        
        return fig, ax, ani
    
    def simulate_observable_timeseries(self, duration: float = 200):
        """
        Simulate observable timeseries for 3I/ATLAS jets.
        
        Parameters:
            duration: Simulation duration in seconds
            
        Returns:
            Dictionary with simulation results
        """
        time_points = np.linspace(0, duration, 1000)
        observables = []
        jet_angles = []
        
        for t in time_points:
            # Calculate order parameters
            Psi_vals = [self.calculate_order_parameter(t, i) for i in range(3)]
            
            # Calculate observable (simplified projection)
            observable = np.real(np.sum(Psi_vals)) / 3
            observables.append(observable)
            
            # Record jet angles
            positions = self.simulate_jet_positions(t)
            angles = [np.arctan2(y, x) for x, y in positions]
            jet_angles.append(angles)
        
        # Convert to arrays
        observables = np.array(observables)
        jet_angles = np.array(jet_angles)
        
        # Simulate activity jumps
        activity_with_jumps = observables.copy()
        jump_times = [40, 85, 130, 175]
        jump_sizes = [0.3, -0.2, 0.4, -0.3]
        
        for jump_time, jump_size in zip(jump_times, jump_sizes):
            idx = np.argmin(np.abs(time_points - jump_time))
            activity_with_jumps[idx:] += jump_size
        
        return {
            'times': time_points,
            'observables': observables,
            'activity_with_jumps': activity_with_jumps,
            'jet_angles': jet_angles,
            'jump_times': jump_times,
            'jump_sizes': jump_sizes
        }
    
    def create_timeseries_plot(self):
        """Create plot of observable timeseries."""
        # Simulate data
        data = self.simulate_observable_timeseries()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Panel 1: Jet angles over time
        ax1.set_title('Jet Direction Evolution', fontsize=12, fontweight='bold')
        
        for i in range(3):
            angles_deg = np.degrees(data['jet_angles'][:, i])
            ax1.plot(data['times'], angles_deg, 
                    label=f'Jet {i+1}', linewidth=2, alpha=0.8)
        
        ax1.set_ylabel('Jet Angle (degrees)', fontsize=10)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # Panel 2: Observable with activity jumps
        ax2.set_title('Observable with Activity Jumps', 
                     fontsize=12, fontweight='bold')
        
        ax2.plot(data['times'], data['observables'], 'b-', 
                label='Without jumps', linewidth=1, alpha=0.5)
        ax2.plot(data['times'], data['activity_with_jumps'], 'r-', 
                label='With jumps', linewidth=2)
        
        # Mark jump locations
        for jump_time, jump_size in zip(data['jump_times'], data['jump_sizes']):
            ax2.axvline(x=jump_time, color='gray', linestyle=':', alpha=0.5)
            ax2.text(jump_time, ax2.get_ylim()[1]*0.9, f'{jump_size:+.2f}', 
                    ha='center', fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        
        ax2.set_xlabel('Time (s)', fontsize=10)
        ax2.set_ylabel('Observable Value', fontsize=10)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        return fig, (ax1, ax2), data

def main():
    """
    Main function to run all jet simulations.
    """
    print("="*70)
    print("SIMULATING 3I/ATLAS JETS")
    print("="*70)
    
    # Create simulator
    simulator = JetSimulator()
    
    print(f"\n3I/ATLAS parameters:")
    print(f"  γ = {simulator.gamma:.2e} s⁻¹")
    print(f"  ω = {simulator.omega:.2e} rad/s")
    print(f"  ε = {simulator.epsilon:.2e} s⁻¹")
    print(f"  p = ε/(γ+ε) = {simulator.p:.2f}")
    print(f"  Wobble period = {simulator.wobble_period:.2f} days")
    print(f"  N = {simulator.N} (triple symmetry)")
    
    print("\n1. Creating static visualization...")
    fig1, axes1 = simulator.create_static_visualization()
    fig1.savefig('3i_jets_static.png', dpi=300, bbox_inches='tight')
    print("   Saved as '3i_jets_static.png'")
    
    print("\n2. Creating animated wobble...")
    fig2, ax2, ani2 = simulator.create_animated_wobble()
    try:
        ani2.save('3i_jets_wobble.gif', writer='pillow', fps=20)
        print("   Animation saved as '3i_jets_wobble.gif'")
    except:
        print("   Note: Animation saving requires pillow/ffmpeg")
    
    print("\n3. Simulating timeseries...")
    fig3, axes3, data = simulator.create_timeseries_plot()
    fig3.savefig('3i_timeseries.png', dpi=300, bbox_inches='tight')
    print("   Saved as '3i_timeseries.png'")
    
    print("\n" + "="*70)
    print("KEY PREDICTIONS FOR 3I/ATLAS")
    print("="*70)
    print("\nPrediction 1 (Anti-tail coherence):")
    print("  • ε-driven coherence resists dispersion")
    print("  • Narrow, structured anti-tail expected")
    
    print("\nPrediction 2 (Synchronized wobble):")
    print(f"  • Period: {simulator.wobble_period:.2f} days")
    print("  • All jets wobble in phase")
    print("  • 120° separation maintained")
    
    print("\nPrediction 3 (Activity jumps):")
    print("  • Step-like changes in brightness/acceleration")
    print("  • Uncorrelated with heliocentric distance")
    print("  • Noise-induced state transitions")
    
    plt.show()
    
    return simulator, data

if __name__ == "__main__":
    simulator, data = main()
