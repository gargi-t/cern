"""
fixed harrdware
"""

import numpy as np
from typing import Dict
import time

class BeamSimulator:
    """Simulates beam parameters for FLASH conditions."""
    
    def __init__(self, beam_energy_mev: float = 100.0):
        self.beam_energy = beam_energy_mev  # MeV
        self.pulse_width_ns = 1.0  # FLASH pulse width
        self.repetition_rate_hz = 10  # Pulse repetition
        
    def generate_pulse(self, intensity: float) -> Dict:
        """Generate a simulated beam pulse.
        
        Args:
            intensity: Relative beam intensity (10^6 to 10^12 particles/pulse)
            
        Returns:
            Dictionary with beam parameters
        """
        # Base parameters
        pulse = {
            'timestamp': time.time_ns(),
            'intensity_requested': intensity,
            'beam_energy_mev': self.beam_energy,
            'pulse_width_ns': self.pulse_width_ns,
        }
        
        #  realistic fluctuations (1% Gaussian noise)
        actual_intensity = intensity * np.random.normal(1.0, 0.01)
        pulse['intensity_actual'] = max(actual_intensity, 0)
        
        # Simulate beam current monitor reading
        # 1e10 particles ≈ 1.6 nC charge
        charge_nc = actual_intensity * 1.6e-19 * 1e9  # Simplified conversion
        pulse['beam_current_monitor'] = charge_nc * np.random.normal(1.0, 0.02)
        
        return pulse

class CherenkovDetectorSim:
    """Simulates Cherenkov radiator + PMT response with PROPER ADC."""
    
    def __init__(self, material: str = 'lead_glass', adc_bits: int = 14):
        self.materials = {
            'lead_glass': {'n': 1.65, 'radiation_length': 2.5},
            'fused_silica': {'n': 1.46, 'radiation_length': 12.0},
        }
        self.material = self.materials.get(material, self.materials['lead_glass'])
        
        # PMT parameters
        self.pmt_gain = 1e6  # Typical PMT gain
        self.dark_current_rate = 100  # Hz
        self.transit_time_spread = 0.3  # ns
        
        # Cherenkov physics
        self.photons_per_particle = 500  # Photons per relativistic electron
        
        # ADC parameters 
        self.adc_bits = adc_bits
        self.adc_max_value = 2**adc_bits - 1
        
        # Calculate ADC conversion to cover FLASH range (10^6 to 10^12 particles)
        # We want: 10^12 particles → ~80% of ADC range (avoid saturation)
        max_particles_for_calibration = 1e12
        max_photons = max_particles_for_calibration * self.photons_per_particle
        max_electrons = max_photons * self.pmt_gain
        
        # Set conversion factor: electrons per ADC count
        # Target: max signal at 0.8 * ADC range
        self.adc_conversion = 5e12 
        
        print(f"[Cherenkov] {adc_bits}-bit ADC, max={self.adc_max_value}, "
              f"{self.adc_conversion:.1f} electrons/ADC count")
        
    def detect_pulse(self, beam_pulse: Dict) -> Dict:
        """Simulate Cherenkov response with proper ADC digitization."""
        intensity = beam_pulse['intensity_actual']
        
        # 1. Physics: Generate Cherenkov photons
        mean_photons = intensity * self.photons_per_particle
        
        # Statistical fluctuations (Poisson for low counts, Gaussian for high)
        if mean_photons < 1e9:
            cherenkov_photons = np.random.poisson(max(mean_photons, 0))
        else:
            std_dev = np.sqrt(mean_photons)
            cherenkov_photons = max(0, np.random.normal(mean_photons, std_dev))
        
        # 2. PMT: Convert photons to electrons
        signal_electrons = cherenkov_photons * self.pmt_gain
        
        # 3. Add electronic noise (1% of signal)
        noise_std = 0.01 * signal_electrons
        electronic_noise = np.random.normal(0, noise_std)
        
        # 4. Add dark current (negligible for short pulses)
        dark_counts = np.random.poisson(self.dark_current_rate * 1e-9)
        dark_electrons = dark_counts * self.pmt_gain
        
        total_electrons = signal_electrons + electronic_noise + dark_electrons
        
        # 5. ADC DIGITIZATION
        # Convert electrons to ADC counts
        adc_counts_float = total_electrons / self.adc_conversion
        
        # ADC saturation: clip at maximum value
        adc_counts_clipped = np.clip(adc_counts_float, 0, self.adc_max_value)
        
        # ADC quantization: round to nearest integer
        adc_counts = np.round(adc_counts_clipped).astype(np.int32)
        
        # Check for saturation
        adc_saturated = (adc_counts_float >= self.adc_max_value)
        
        # Calculate actual photons detected (after all effects)
        actual_photons_detected = (adc_counts * self.adc_conversion) / self.pmt_gain
        
        return {
            'timestamp': beam_pulse['timestamp'],
            'adc_counts_raw': int(adc_counts),  # INTEGER ADC counts
            'adc_saturated': bool(adc_saturated),
            'electrons_before_adc': float(total_electrons),
            'cherenkov_photons_generated': float(cherenkov_photons),
            'cherenkov_photons_detected': float(actual_photons_detected),
            'photons_per_particle': self.photons_per_particle,
            'pmt_gain': self.pmt_gain,
            'adc_conversion_factor': float(self.adc_conversion),  # electrons per ADC count
            'adc_max_value': self.adc_max_value,
            'adc_bits': self.adc_bits,
        }


class ScintillatorDetectorSim:
    """Simulates plastic scintillator with proper ADC."""
    
    def __init__(self, adc_bits: int = 16):
        self.birks_constant = 0.01  # mm/MeV 
        self.light_yield = 10000  # photons/MeV at low intensity
        self.saturation_factor = 5e-12  # Empirical high-intensity quenching
        
        # ADC parameters 
        self.adc_bits = adc_bits
        self.adc_max_value = 2**adc_bits - 1
        
        # Calculate ADC conversion for scintillator
        # Different scaling than Cherenkov
        max_particles_for_calibration = 1e10  # Scintillator saturates earlier
        beam_energy = 100.0  # MeV
        max_deposited_energy = max_particles_for_calibration * beam_energy
        max_light_yield = self.light_yield * max_deposited_energy
        
        # Account for saturation at high intensity
        saturation_at_max = 0.1  # Assume 90% saturation at 1e10 particles
        max_light_after_saturation = max_light_yield * saturation_at_max
        
        self.adc_conversion = 1e8
        
        print(f"[Scintillator] {adc_bits}-bit ADC, max={self.adc_max_value}, "
              f"{self.adc_conversion:.1f} photons/ADC count")
        
    def detect_pulse(self, beam_pulse: Dict) -> Dict:
        """Simulate scintillator response with saturation and proper ADC."""
        intensity = beam_pulse['intensity_actual']
        beam_energy = beam_pulse['beam_energy_mev']
        
        # 1. Physics: Energy deposition
        deposited_energy = beam_energy * intensity  # MeV
        
        # 2. Birks' law saturation (dE/dx effect)
        dE_dx = 2.0  # MeV/mm (approximate for electrons in plastic)
        birks_factor = 1.0 / (1.0 + self.birks_constant * dE_dx * np.sqrt(intensity))
        
        # 3. High-intensity quenching (empirical)
        intensity_saturation = 1.0 / (1.0 + self.saturation_factor * intensity**1.5)
        
        total_saturation = birks_factor * intensity_saturation
        
        # 4. Light yield with saturation
        mean_light_yield = self.light_yield * deposited_energy * total_saturation
        
        # Statistical fluctuations
        if mean_light_yield < 1e9:
            light_yield = np.random.poisson(max(mean_light_yield, 0))
        else:
            std_dev = np.sqrt(mean_light_yield)
            light_yield = max(0, np.random.normal(mean_light_yield, std_dev))
        
        # 5. ADC DIGITIZATION
        adc_counts_float = light_yield / self.adc_conversion
        adc_counts_clipped = np.clip(adc_counts_float, 0, self.adc_max_value)
        adc_counts = np.round(adc_counts_clipped).astype(np.int32)
        adc_saturated = (adc_counts_float >= self.adc_max_value)
        
        # Determine if severely saturated
        severely_saturated = total_saturation < 0.5 or adc_saturated
        
        # Calculate actual light yield after ADC effects
        actual_light_detected = adc_counts * self.adc_conversion
        
        return {
            'timestamp': beam_pulse['timestamp'],
            'adc_counts_raw': int(adc_counts),  # INTEGER ADC counts
            'adc_saturated': bool(adc_saturated),
            'light_yield_generated': float(light_yield),
            'light_yield_detected': float(actual_light_detected),
            'birks_factor': float(birks_factor),
            'intensity_saturation': float(intensity_saturation),
            'total_saturation': float(total_saturation),
            'saturated': severely_saturated,
            'light_yield_per_mev': self.light_yield,
            'adc_conversion_factor': float(self.adc_conversion),  # photons per ADC count
            'adc_max_value': self.adc_max_value,
            'adc_bits': self.adc_bits,
        }
