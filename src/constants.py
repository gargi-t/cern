"""
Centralized constants for Cherenkov FLASH Experiment.
Single source of truth for all calibration factors and material properties.
"""

# =============================================================================
# ADC CALIBRATION CONSTANTS
# =============================================================================

# Cherenkov detector calibration chain
# ADC → electrons → photons → calibrated response
ADC_CONVERSION = {
    'cherenkov': {
        'electrons_per_adc': 5e12,      # electrons per ADC count
        'pmt_gain': 1e6,                 # electrons per photon
        'quantum_efficiency': 0.25,       # photons detected / photons generated
        'adc_bits': 14,                   
        'adc_max_value': 2**14 - 1,       # 16383
        'description': 'Cherenkov: ADC counts → electrons ÷ PMT gain ÷ QE = photons'
    },
    'scintillator': {
        'photons_per_adc': 1e8,           # photons per ADC count
        'adc_bits': 16,
        'adc_max_value': 2**16 - 1,       # 65535
        'description': 'Scintillator: ADC counts × photons_per_adc = photons'
    }
}

# Dark current parameters
DARK_CURRENT = {
    'rate_hz': 100,
    'integration_time_ns': 1,             # ns per pulse
}

# =============================================================================
# MATERIAL PROPERTIES
# =============================================================================

MATERIALS = {
    'lead_glass': {
        'name': 'Lead Glass',
        'refractive_index': 1.65,
        'radiation_length_cm': 2.5,
        'density_g_cm3': 4.0,
        'cherenkov_photons_per_cm': 500,   # Approximate photons/cm for relativistic e-
        'description': 'High-Z Cherenkov radiator'
    },
    'fused_silica': {
        'name': 'Fused Silica',
        'refractive_index': 1.46,
        'radiation_length_cm': 12.0,
        'density_g_cm3': 2.2,
        'cherenkov_photons_per_cm': 200,
        'description': 'Low-Z Cherenkov radiator'
    },
    'plastic_scintillator': {
        'name': 'Plastic Scintillator (BC-408)',
        'refractive_index': 1.58,
        'density_g_cm3': 1.03,
        'light_yield_photons_per_mev': 10000,  # at low intensity
        'birks_constant_mm_per_mev': 0.01,      # mm/MeV
        'birks_constant_cm_per_mev': 0.1,       # cm/MeV (converted)
        'description': 'Standard plastic scintillator'
    }
}

# =============================================================================
# BEAM PARAMETERS
# =============================================================================

BEAM = {
    'electron_mass_mev': 0.511,
    'conventional_intensity_range': (1e6, 1e8),    # particles/pulse
    'flash_intensity_threshold': 1e10,              # particles/pulse
    'flash_intensity_range': (1e10, 1e12),          # particles/pulse
    'default_energy_mev': 100.0,
}

# =============================================================================
# PHYSICS CONSTANTS
# =============================================================================

PHYSICS = {
    'h_planck_j_s': 6.626e-34,
    'c_light_m_s': 2.998e8,
    'e_charge_c': 1.602e-19,
    'fine_structure': 1/137.036,
    'avogadro': 6.022e23,
}

# =============================================================================
# EXPERIMENT PATHS
# =============================================================================

PATHS = {
    'data': 'data/',
    'processed_data': 'processed_data/',
    'plots': 'analysis/plots/',
    'reports': 'analysis/reports/',
    'configs': 'configs/',
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_calibration_factor(detector_type, quantity='photons'):
    """
    Get calibration factor for converting ADC counts to physical units.
    
    Args:
        detector_type: 'cherenkov' or 'scintillator'
        quantity: 'photons', 'electrons', or 'adc'
    
    Returns:
        Conversion factor
    """
    if detector_type == 'cherenkov':
        # ADC → electrons → photons
        if quantity == 'photons':
            return (ADC_CONVERSION['cherenkov']['electrons_per_adc'] / 
                   (ADC_CONVERSION['cherenkov']['pmt_gain'] * 
                    ADC_CONVERSION['cherenkov']['quantum_efficiency']))
        elif quantity == 'electrons':
            return ADC_CONVERSION['cherenkov']['electrons_per_adc']
        else:
            return 1.0
            
    elif detector_type == 'scintillator':
        # ADC → photons directly
        if quantity == 'photons':
            return ADC_CONVERSION['scintillator']['photons_per_adc']
        else:
            return 1.0
    
    raise ValueError(f"Unknown detector type: {detector_type}")


def get_material(material_name):
    """Get material properties by name."""
    if material_name in MATERIALS:
        return MATERIALS[material_name]
    else:
        # Return default
        return MATERIALS['lead_glass']