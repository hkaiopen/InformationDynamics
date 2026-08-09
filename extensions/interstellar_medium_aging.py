"""
interstellar_medium_aging.py

Information Dynamics Extension: Long-Term Interstellar Medium Aging

Simulates cosmic ray, dust bombardment, and UV effects on ISO surfaces
during ~10 Gyr wandering time.
"""

import numpy as np
from typing import Dict, List


class ISMAgingModel:
    """
    Models long-term aging of interstellar objects in the ISM.

    Processes:
    - Cosmic ray spallation (modifies surface isotopes, depth ~1 m)
    - Dust bombardment (mantle accretion/erosion, depth ~0.1-1 m)
    - UV photolysis (very shallow, depth ~1 mm)
    - Internal preservation (cold interior retains primordial info)
    """

    def __init__(self,
                 radius_km: float = 1.3,
                 density_kg_m3: float = 500.0,
                 wandering_time_Gyr: float = 10.0):
        self.R = radius_km * 1000.0
        self.rho = density_kg_m3
        self.t_wander = wandering_time_Gyr * 1e9 * 365.25 * 86400

        # ISM parameters
        self.n_ISM = 1e6
        self.T_ISM = 15.0
        self.CR_flux = 1e4
        self.UV_flux = 1e15

    def alteration_depth(self) -> float:
        """Total depth of surface altered by ISM processes."""
        # Cosmic ray penetration (1 GeV protons)
        d_CR = 1.0  # m (approximate for cometary material)

        # Dust bombardment (sputtering + accretion over 10 Gyr)
        # Typical ISM dust flux ~ 1e-15 kg/m^2/s
        dust_flux = 1e-15
        mass_accum = dust_flux * self.t_wander
        d_dust = mass_accum / self.rho

        # UV photolysis (very shallow)
        d_UV = 1e-3  # m

        # Total alteration = max of processes (they act in parallel)
        return max(d_CR, d_dust, d_UV)

    def preservation_fraction(self) -> float:
        """Volume fraction retaining primordial information."""
        d = self.alteration_depth()
        if d >= self.R:
            return 0.0
        V_total = 4/3 * np.pi * self.R**3
        V_preserve = 4/3 * np.pi * (self.R - d)**3
        return float(V_preserve / V_total)

    def surface_isotope(self,
                        primordial: float = 200.0,
                        ISM_value: float = 89.0) -> Dict:
        """Simulate surface isotope modification."""
        d = self.alteration_depth()
        f_preserve = self.preservation_fraction()

        # Surface approaches ISM composition over time
        tau_mix = 1e9 * 365.25 * 86400  # 1 Gyr mixing timescale
        mixing = 1.0 - np.exp(-self.t_wander / tau_mix)

        surface = ISM_value + (primordial - ISM_value) * (1 - mixing)

        # Bulk average
        bulk = f_preserve * primordial + (1 - f_preserve) * surface

        return {
            'surface': float(surface),
            'interior': float(primordial),
            'bulk': float(bulk),
            'preserve_frac': float(f_preserve),
            'alter_depth_m': float(d)
        }

    def layered_structure(self, n_layers: int = 20) -> Dict:
        """
        Detailed layered structure showing surface gradient.
        FIXED: Use non-uniform grid with fine resolution near surface.
        """
        d_alter = self.alteration_depth()
        
        # Create radii: fine near surface (first 2 m), coarse in interior
        # Surface zone: 20 layers in first 2 * d_alter
        # Interior zone: remaining layers
        n_surf = min(15, n_layers // 2)
        n_int = n_layers - n_surf
        
        # Surface radii (from R - 2*d_alter to R)
        surf_max = min(2.0 * d_alter, self.R)
        r_surf = np.linspace(self.R - surf_max, self.R, n_surf + 1)
        
        # Interior radii (from 0 to R - surf_max)
        if n_int > 0 and self.R > surf_max:
            r_int = np.linspace(0, self.R - surf_max, n_int + 1)
            # Remove duplicate boundary
            radii = np.concatenate([r_int[:-1], r_surf])
        else:
            radii = r_surf
        
        layers = []
        for i in range(len(radii) - 1):
            r_in = radii[i]
            r_out = radii[i + 1]
            r_mid = (r_in + r_out) / 2
            depth_from_surface = self.R - r_mid

            # Alteration fraction: 1 at surface, 0 deep inside
            if d_alter > 0:
                alteration = np.exp(-depth_from_surface / (d_alter / 3.0))
            else:
                alteration = 0.0

            # Information fidelity: inverse of alteration
            fidelity = 1.0 - alteration

            # Temperature (simplified: increases toward center due to
            # radioactive heating, but negligible for small comets)
            T_layer = self.T_ISM

            layers.append({
                'layer': i,
                'r_in_m': float(r_in),
                'r_out_m': float(r_out),
                'depth_m': float(depth_from_surface),
                'alteration': float(alteration),
                'fidelity': float(fidelity),
                'temperature_K': float(T_layer)
            })

        return {'layers': layers, 'n_layers': len(layers), 'alter_depth_m': float(d_alter)}


def main():
    print("=" * 70)
    print("INTERSTELLAR MEDIUM AGING: 3I/ATLAS")
    print("=" * 70)

    ism = ISMAgingModel(radius_km=1.3, wandering_time_Gyr=10.0)

    print(f"\nNucleus: R = {ism.R/1000:.1f} km, rho = {ism.rho:.0f} kg/m³")
    print(f"Wandering time: {ism.t_wander/(1e9*365.25*86400):.1f} Gyr")

    d = ism.alteration_depth()
    print(f"\nSurface alteration depth: {d:.3f} m")
    print(f"Relative to radius: {d/ism.R*100:.4f}%")

    f_pres = ism.preservation_fraction()
    print(f"\nVolume preserving primordial info: {f_pres*100:.2f}%")

    iso = ism.surface_isotope(primordial=200.0, ISM_value=89.0)
    print(f"\nIsotope modification (12C/13C):")
    print(f"  Primordial (interior): {iso['interior']:.0f}")
    print(f"  Altered surface:       {iso['surface']:.0f}")
    print(f"  Bulk average:          {iso['bulk']:.0f}")
    print(f"  (Observed: ~147, between surface and interior)")

    print(f"\nLayered structure (showing surface gradient):")
    layers = ism.layered_structure(n_layers=20)

    # Print surface layers in detail
    print(f"  {'Layer':>6} {'Depth(m)':>10} {'Alteration':>12} {'Fidelity':>10}")
    print(f"  {'-'*42}")
    for layer in layers['layers'][:8]:  # surface layers
        print(f"  {layer['layer']:>6} {layer['depth_m']:>10.1f} "
              f"{layer['alteration']:>12.4f} {layer['fidelity']:>10.4f}")
    print(f"  ...")
    for layer in layers['layers'][-3:]:  # deep layers
        print(f"  {layer['layer']:>6} {layer['depth_m']:>10.1f} "
              f"{layer['alteration']:>12.4f} {layer['fidelity']:>10.4f}")


if __name__ == "__main__":
    main()