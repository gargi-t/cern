"""
Theoretical models for Cherenkov and scintillator physics.
Compare experimental results to theoretical predictions.
"""

import numpy as np
from scipy import constants

class TheoryModels:
    """Theoretical calculations for detector responses."""
    
    def __init__(self):
        # Physical constants
        self.h = constants.h          # Planck's constant [J·s]
        self.c = constants.c          # Speed of light [m/s]
        self.e = constants.e          # Electron charge [C]
        self.alpha = constants.alpha  # Fine structure constant
        
        # Material properties
        self.materials = {
            'lead_glass': {
                'n': 1.65,           # Refractive index
                'radiation_length': 2.5,  # cm
                'density': 4.0,      # g/cm³
            },
            'plastic_scintillator': {
                'n': 1.58,
                'density': 1.03,     # g/cm³
                'light_yield': 10000,  # photons/MeV
                'birks_constant': 0.01,  # mm/MeV
            }
        }
    
    def cherenkov_yield_frank_tamm(self, energy_mev=100.0, material='lead_glass',
                                   length_cm=1.0, wavelength_range=(300, 600)):
        """
        Calculate Cherenkov photon yield using Frank-Tamm formula.
        
        Frank-Tamm: dN/dx = 2πα ∫ (1 - 1/(β²n²)) dλ/λ²
        
        Args:
            energy_mev: Electron energy in MeV
            material: Material name
            length_cm: Radiator length in cm
            wavelength_range: (min, max) in nm
        
        Returns:
            Photons per electron
        """
        material_props = self.materials.get(material, self.materials['lead_glass'])
        n = material_props['n']
        
        # Convert energy to velocity β = v/c
        # E = γ m c², where m = 0.511 MeV for electron
        m_e_mev = 0.511  # Electron rest mass in MeV
        gamma = energy_mev / m_e_mev
        beta = np.sqrt(1 - 1/gamma**2)
        
        # Convert wavelength range to meters
        lambda_min = wavelength_range[0] * 1e-9  # m
        lambda_max = wavelength_range[1] * 1e-9  # m
        
        # Frank-Tamm formula integrated over wavelength
        # dN/dx = 2πα (1 - 1/(β²n²)) ∫ dλ/λ²
        prefactor = 2 * np.pi * self.alpha
        
        # Check Cherenkov condition: β > 1/n
        if beta <= 1/n:
            return 0.0
        
        angular_factor = 1 - 1/(beta**2 * n**2)
        
        # Integral of 1/λ² from λ_min to λ_max
        lambda_integral = (1/lambda_min - 1/lambda_max)
        
        # Photons per cm
        photons_per_cm = prefactor * angular_factor * lambda_integral
        
        # Convert to photons per electron for given length
        photons_per_electron = photons_per_cm * length_cm
        
        return photons_per_electron
    
    def cherenkov_yield_approximate(self, energy_mev=100.0, material='lead_glass'):
        """
        Approximate Cherenkov yield using empirical formula.
        Typical: ~100-200 photons/cm/electron in visible range.
        
        Returns:
            Approximate photons per electron
        """
        material_props = self.materials.get(material, self.materials['lead_glass'])
        
        # Check Cherenkov threshold
        m_e_mev = 0.511
        gamma = energy_mev / m_e_mev
        beta = np.sqrt(1 - 1/gamma**2)
        
        if beta <= 1/material_props['n']:
            return 0.0
        
        # Empirical: ~100 photons/cm for relativistic electrons
        # Adjust for energy (more photons at higher energy)
        photons_per_cm = 100 * (gamma / 200)  # Normalize to 100 MeV
        
        # Typical radiator: 1 cm
        photons_per_electron = photons_per_cm * 1.0
        
        return photons_per_electron
    
    def scintillator_birks_law(self, dEdx, L0, kB):
        """
        Birks' law for scintillator saturation.
        
        L = L₀ * dE/dx / (1 + kB * dE/dx)
        
        Args:
            dEdx: Energy loss per unit length (MeV/cm)
            L0: Light yield at low dE/dx (photons/MeV)
            kB: Birks' constant (cm/MeV)
        
        Returns:
            Light yield (photons/MeV)
        """
        return L0 * dEdx / (1 + kB * dEdx)
    
    def electron_dEdx_in_plastic(self, energy_mev):
        """
        Approximate dE/dx for electrons in plastic scintillator.
        
        Uses Bethe-Bloch simplified for electrons.
        Typical: ~2 MeV/cm for relativistic electrons.
        
        Args:
            energy_mev: Electron energy in MeV
        
        Returns:
            dE/dx in MeV/cm
        """
        # Simplified: dE/dx ~ 2 MeV/cm for relativistic electrons
        # Slightly energy dependent
        if energy_mev < 10:
            # Lower energy, higher dE/dx
            return 3.0 - 0.1 * energy_mev
        else:
            # Relativistic plateau
            return 2.0
    
    def predict_cherenkov_response(self, beam_intensity, pmt_gain=1e6, qe=0.25):
        """
        Predict Cherenkov detector response.
        
        Args:
            beam_intensity: Particles per pulse
            pmt_gain: PMT gain
            qe: Quantum efficiency
        
        Returns:
            Predicted ADC counts
        """
        # Photons per electron (from Frank-Tamm)
        photons_per_electron = self.cherenkov_yield_approximate(100.0)
        
        # Total photons
        total_photons = photons_per_electron * beam_intensity
        
        # Electrons at PMT anode
        electrons = total_photons * qe * pmt_gain
        
        # ADC counts (using same conversion as hardware sim)
        adc_conversion = 5e12  # electrons per ADC count
        adc_counts = electrons / adc_conversion
        
        return {
            'photons_per_electron': photons_per_electron,
            'total_photons': total_photons,
            'electrons': electrons,
            'adc_counts': adc_counts,
            'adc_saturated': adc_counts > 16383,  # 14-bit ADC
        }
    
    def predict_scintillator_response(self, beam_intensity, energy_mev=100.0):
        """
        Predict scintillator response with saturation.
        
        Args:
            beam_intensity: Particles per pulse
            energy_mev: Beam energy in MeV
        
        Returns:
            Predicted response
        """
        material = self.materials['plastic_scintillator']
        L0 = material['light_yield']  # photons/MeV at low intensity
        kB = material['birks_constant'] * 10  # Convert mm/MeV to cm/MeV
        
        # Energy deposited
        deposited_energy = beam_intensity * energy_mev  # MeV
        
        # dE/dx for electrons
        dEdx = self.electron_dEdx_in_plastic(energy_mev)  # MeV/cm
        
        # Birks' law light yield
        light_yield_per_mev = self.scintillator_birks_law(dEdx, L0, kB)
        
        # Total light
        total_light = light_yield_per_mev * deposited_energy
        
        # Additional intensity-dependent quenching
        # Empirical: additional quenching at high intensity
        intensity_quenching = 1.0 / (1.0 + 5e-12 * beam_intensity**1.5)
        total_light *= intensity_quenching
        
        # ADC counts
        adc_conversion = 1e8  # photons per ADC count
        adc_counts = total_light / adc_conversion
        
        return {
            'light_yield_per_mev': light_yield_per_mev,
            'total_light': total_light,
            'intensity_quenching': intensity_quenching,
            'adc_counts': adc_counts,
            'adc_saturated': adc_counts > 65535,  # 16-bit ADC
            'saturation_factor': intensity_quenching,
        }
    
    def compare_with_experiment(self, experimental_data):
        """
        Compare theoretical predictions with experimental results.
        
        Args:
            experimental_data: Dictionary with 'beam', 'chere', 'scint'
        
        Returns:
            Comparison results
        """
        beam = experimental_data['beam']
        chere_exp = experimental_data['chere']
        scint_exp = experimental_data['scint']
        
        # Predict for each beam intensity
        chere_pred = []
        scint_pred = []
        
        for intensity in beam:
            # Cherenkov prediction
            cher_pred = self.predict_cherenkov_response(intensity)
            chere_pred.append(cher_pred['photons_per_electron'])  # Normalized
            
            # Scintillator prediction
            scint_pred_dict = self.predict_scintillator_response(intensity)
            scint_pred.append(scint_pred_dict['total_light'] / intensity)  # Normalized
        
        chere_pred = np.array(chere_pred)
        scint_pred = np.array(scint_pred)
        
        # Calculate agreement metrics
        cher_ratio = chere_exp / chere_pred
        scint_ratio = scint_exp / scint_pred
        
        cher_agreement = {
            'mean_ratio': np.mean(cher_ratio),
            'std_ratio': np.std(cher_ratio),
            'relative_difference': np.abs(np.mean(cher_ratio) - 1) * 100,
        }
        
        scint_agreement = {
            'mean_ratio': np.mean(scint_ratio),
            'std_ratio': np.std(scint_ratio),
            'relative_difference': np.abs(np.mean(scint_ratio) - 1) * 100,
        }
        
        return {
            'cherenkov': {
                'predicted': chere_pred,
                'experimental': chere_exp,
                'agreement': cher_agreement,
            },
            'scintillator': {
                'predicted': scint_pred,
                'experimental': scint_exp,
                'agreement': scint_agreement,
            },
            'summary': {
                'cherenkov_agreement_percent': cher_agreement['relative_difference'],
                'scintillator_agreement_percent': scint_agreement['relative_difference'],
                'theory_explains_data': cher_agreement['relative_difference'] < 50 and 
                                       scint_agreement['relative_difference'] < 50,
            }
        }


# Example usage
if __name__ == "__main__":
    print("Theory Models for Cherenkov FLASH Experiment")
    print("="*60)
    
    theory = TheoryModels()
    
    # Test Cherenkov calculations
    print("\nCherenkov Theory:")
    for energy in [10, 50, 100, 200]:
        yield_ft = theory.cherenkov_yield_frank_tamm(energy)
        yield_approx = theory.cherenkov_yield_approximate(energy)
        print(f"  {energy:3d} MeV: Frank-Tamm={yield_ft:6.0f} photons/e, "
              f"Approx={yield_approx:6.0f}")
    
    # Test scintillator calculations
    print("\nScintillator Theory (Birks' law):")
    for dEdx in [1, 2, 5, 10]:  # MeV/cm
        light_yield = theory.scintillator_birks_law(dEdx, L0=10000, kB=0.1)
        print(f"  dE/dx={dEdx:2d} MeV/cm: Light yield={light_yield:.0f} photons/MeV")
    
    # Test predictions
    print("\nPredictions for 1e9 particles/pulse:")
    cher_pred = theory.predict_cherenkov_response(1e9)
    scint_pred = theory.predict_scintillator_response(1e9)
    
    print(f"  Cherenkov: {cher_pred['adc_counts']:.0f} ADC counts")
    print(f"  Scintillator: {scint_pred['adc_counts']:.0f} ADC counts")
    print(f"  Scintillator saturation: {scint_pred['saturation_factor']:.3f}")