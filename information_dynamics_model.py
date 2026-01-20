# information_dynamics_model.py
"""
Information Dynamics: A Unified Predictive Framework for Interstellar Objects

This module implements the core Information Dynamics model based on the Complex Ginzburg-Landau Equation (CGLE)
as described in the paper. It includes the order parameter evolution and the information purity metric.

Reference: Huang & Liu (2026), "Information Dynamics: A Unified Predictive Framework for Interstellar Objects"
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
import warnings

class InformationDynamicsModel:
    """
    Implements the Information Dynamics model for interstellar objects (ISOs).
    
    The model is based on a simplified Complex Ginzburg-Landau Equation (CGLE):
    ∂Ψ/∂t = -γΨ + iωΨ + ε|Ψ|²Ψ + F_thermal + η(t)
    """
    
    def __init__(self, gamma: float, omega: float, epsilon: float, 
                 F_thermal: float = 0.0, noise_amplitude: float = 0.0):
        """
        Initialize the model with physical parameters.
        
        Parameters:
        - gamma (γ): linear dissipation rate in s⁻¹
        - omega (ω): characteristic frequency in rad/s
        - epsilon (ε): nonlinear self-interaction strength in s⁻¹
        - F_thermal: thermal forcing term (dimensionless)
        - noise_amplitude: amplitude of Gaussian noise η(t)
        """
        self.gamma = float(gamma)
        self.omega = float(omega)
        self.epsilon = float(epsilon)
        self.F_thermal = float(F_thermal)
        self.noise_amp = float(noise_amplitude)
        
        # Validate parameters
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate model parameters for physical consistency."""
        if self.gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {self.gamma}")
        if self.epsilon < 0:
            raise ValueError(f"epsilon must be non-negative, got {self.epsilon}")
        if self.noise_amp < 0:
            raise ValueError(f"noise_amplitude must be non-negative, got {self.noise_amp}")
        if self.omega <= 0:
            warnings.warn(f"omega should be positive, got {self.omega}")
    
    @property
    def information_purity(self) -> float:
        """
        Calculate the information purity p = ε / (γ + ε)
        
        The information purity quantifies the dominance of nonlinear self-organization
        over total dissipative loss. Ranges from 0 (purely thermal) to 1 (purely information-driven).
        """
        denominator = self.gamma + self.epsilon
        if denominator == 0:
            return 0.0
        return self.epsilon / denominator
    
    def classify_state(self) -> str:
        """
        Classify the object based on its information purity p.
        
        Classification thresholds:
        - p > 0.7: Information-Driven State (like 1I/'Oumuamua)
        - p < 0.3: Thermally-Driven State (like 2I/Borisov)
        - 0.3 ≤ p ≤ 0.7: Mixed State (like 3I/ATLAS)
        """
        p = self.information_purity
        
        if p > 0.7:
            return "Information-Driven State"
        elif p < 0.3:
            return "Thermally-Driven State"
        else:
            return "Mixed State"
    
    def evolve_order_parameter(self, Psi: complex, dt: float, 
                              add_noise: bool = True) -> complex:
        """
        Evolve the order parameter Ψ using the CGLE for one time step.
        
        Parameters:
            Psi: Current complex order parameter
            dt: Time step in seconds
            add_noise: Whether to add noise term η(t)
            
        Returns:
            Updated order parameter after time dt
        """
        # Validate input
        dt = float(dt)
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        
        # Calculate the deterministic part
        linear_term = -self.gamma * Psi
        frequency_term = 1j * self.omega * Psi
        nonlinear_term = self.epsilon * (np.abs(Psi) ** 2) * Psi
        
        dPsi_dt = linear_term + frequency_term + nonlinear_term + self.F_thermal
        
        # Add Gaussian noise if specified
        if add_noise and self.noise_amp > 0:
            noise_real = np.random.normal(0, self.noise_amp * np.sqrt(dt))
            noise_imag = np.random.normal(0, self.noise_amp * np.sqrt(dt))
            dPsi_dt += (noise_real + 1j * noise_imag)
        
        # Euler integration (simple but sufficient for demonstration)
        Psi_new = Psi + dPsi_dt * dt
        return Psi_new
    
    def calculate_observable(self, Psi_values: np.ndarray, symmetry: int = 3) -> float:
        """
        Calculate observable quantity from order parameter field.
        
        O(r,t) = Re[∑_{k=1}^N C_k Ψ(r_k,t)], with C_k = (1/N) exp(i2πk/N)
        
        Parameters:
            Psi_values: Array of complex order parameters at different spatial points
            symmetry: Integer defining fundamental symmetry (N=1 for featureless, N=3 for triple symmetry)
            
        Returns:
            Observable quantity (e.g., local jet flux)
        """
        if not isinstance(Psi_values, np.ndarray):
            Psi_values = np.array(Psi_values, dtype=complex)
        
        N = int(symmetry)
        if N <= 0:
            raise ValueError(f"Symmetry N must be positive, got {N}")
        
        if len(Psi_values) != N:
            raise ValueError(f"Expected {N} Psi values for N={N} symmetry, got {len(Psi_values)}")
        
        # Calculate coefficients C_k
        k_values = np.arange(1, N + 1)
        C_k = (1.0 / N) * np.exp(1j * 2 * np.pi * k_values / N)
        
        # Calculate the observable
        weighted_sum = np.sum(C_k * Psi_values)
        return float(np.real(weighted_sum))
    
    def simulate_time_series(self, initial_conditions: np.ndarray, 
                            total_time: float, dt: float, 
                            symmetry: int = 3,
                            add_noise: bool = True) -> Dict[str, np.ndarray]:
        """
        Simulate the time evolution of the order parameter field.
        
        Parameters:
            initial_conditions: Array of initial Ψ values
            total_time: Total simulation time in seconds
            dt: Time step in seconds
            symmetry: Number of symmetry components
            add_noise: Whether to include noise term
            
        Returns:
            Dictionary with simulation results
        """
        # Validate inputs
        dt = float(dt)
        total_time = float(total_time)
        
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        if total_time <= 0:
            raise ValueError(f"total_time must be positive, got {total_time}")
        
        n_steps = int(np.ceil(total_time / dt))
        actual_steps = min(n_steps, 100000)  # Safety limit
        
        times = np.zeros(actual_steps)
        observables = np.zeros(actual_steps)
        
        # Initialize state
        Psi_current = np.array(initial_conditions, dtype=complex).copy()
        
        for i in range(actual_steps):
            times[i] = i * dt
            
            # Evolve each component
            for j in range(len(Psi_current)):
                Psi_current[j] = self.evolve_order_parameter(Psi_current[j], dt, add_noise)
            
            # Calculate observable
            observables[i] = self.calculate_observable(Psi_current, symmetry)
            
            # Break if we've reached total_time
            if (i + 1) * dt >= total_time:
                break
        
        # Trim arrays if we broke early
        times = times[:i+1]
        observables = observables[:i+1]
        
        return {
            'times': times,
            'observables': observables,
            'time_seconds': times[-1] if len(times) > 0 else 0
        }
    
    def predict_jet_period(self) -> Dict[str, float]:
        """
        Calculate the predicted natural period of jet wobble.
        
        Returns:
            Dictionary with period in various units
        """
        if self.omega <= 0:
            return {'period_seconds': np.inf, 'period_days': np.inf}
        
        period_seconds = 2 * np.pi / self.omega
        period_days = period_seconds / 86400.0  # Convert to days
        
        return {
            'period_seconds': period_seconds,
            'period_days': period_days,
            'frequency_hz': self.omega / (2 * np.pi)
        }
    
    def simulate_activity_jumps(self, initial_value: float = 1.0,
                               total_time: float = 1e6,
                               dt: float = 3600.0,
                               jump_rate: float = 1e-6,
                               jump_std: float = 0.3) -> Dict[str, np.ndarray]:
        """
        Simulate discrete activity jumps.
        
        Parameters:
            initial_value: Initial activity level
            total_time: Total simulation time in seconds
            dt: Time step in seconds
            jump_rate: Probability of jump per second
            jump_std: Standard deviation of jump size
            
        Returns:
            Dictionary with simulation results
        """
        n_steps = int(np.ceil(total_time / dt))
        times = np.zeros(n_steps)
        activity = np.zeros(n_steps)
        jumps = np.zeros(n_steps, dtype=bool)
        
        current_activity = float(initial_value)
        
        for i in range(n_steps):
            times[i] = i * dt
            
            # Check for jump (Poisson process approximation)
            if np.random.random() < jump_rate * dt:
                jump = jump_std * np.random.randn()
                current_activity += jump
                jumps[i] = True
                current_activity = max(0.1, current_activity)  # Ensure positivity
            
            # Add small continuous variations
            current_activity += 0.01 * np.random.randn() * np.sqrt(dt)
            activity[i] = current_activity
        
        return {
            'times': times,
            'activity': activity,
            'jump_indices': np.where(jumps)[0],
            'n_jumps': np.sum(jumps),
            'jump_rate_observed': np.sum(jumps) / total_time
        }