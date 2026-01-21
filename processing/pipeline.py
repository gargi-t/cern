import h5py
import numpy as np
import glob
import matplotlib.pyplot as plt
from datetime import datetime
import json

def process_latest_file():
    """Processing pipeline"""
    
    # 1. Find and read HDF5
    files = sorted(glob.glob("data/*.h5"))
    if not files:
        print("No HDF5 files in data/ folder")
        print("   Run: python -m src.daq.control simulate")
        return None
    
    file_path = files[-1]
    print(f"Using: {file_path}")
    
    with h5py.File(file_path, "r") as f:
        run_name = list(f.keys())[0]
        run = f[run_name]
        
        # Load datasets 
        beam_intensity = run['beam_intensity'][:]    # Particles/pulse (GOOD)
        cher_adc = run['cherenkov_adc'][:]           # ADC counts
        scint_adc = run['scintillator_adc'][:]       # ADC counts
        # IGNORE beam_current - it's in wrong units for normalization
    
    print(f"\nRaw data: {len(beam_intensity)} events")
    print(f"Beam intensity range: {beam_intensity.min():.2e} to {beam_intensity.max():.2e}")
    print(f"Cherenkov ADC: {cher_adc.min()} to {cher_adc.max()} (max 16383 = 14-bit saturation)")
    print(f"Scintillator ADC: {scint_adc.min()} to {scint_adc.max()}")
    
    # 2. CALIBRATION FACTORS (from hardware_sim.py)
    ADC_CONV_CHER = 5e12      # electrons per ADC count
    ADC_CONV_SCINT = 1e8      # photons per ADC count  
    PMT_GAIN = 1e6            # electrons per photon
    QE_CHER = 0.25            # quantum efficiency
    
    print(f"\nCalibration:")
    print(f"  Cherenkov: {ADC_CONV_CHER:.1e} electrons/ADC → ÷{PMT_GAIN:.0e} ÷{QE_CHER} = photons")
    print(f"  Scintillator: {ADC_CONV_SCINT:.1e} photons/ADC")
    

    # Cherenkov: ADC → electrons → photons
    cher_electrons = cher_adc.astype(np.float64) * ADC_CONV_CHER
    cher_photons = cher_electrons / PMT_GAIN
    cher_calibrated = cher_photons / QE_CHER  # Final Cherenkov photons
    cher_saturated = cher_adc == 16383
    print(f"Cherenkov saturated events: {np.sum(cher_saturated)}/{len(cher_adc)} will be excluded")
    
    # Scintillator: ADC → photons directly
    scint_calibrated = scint_adc.astype(np.float64) * ADC_CONV_SCINT
    
    print(f"\nCalibrated signals:")
    print(f"  Cherenkov photons: {cher_calibrated.min():.2e} to {cher_calibrated.max():.2e}")
    print(f"  Scintillator photons: {scint_calibrated.min():.2e} to {scint_calibrated.max():.2e}")
    
    # Check for saturation
    cher_saturated = (cher_adc == 16383)  # 14-bit ADC max
    print(f"  Cherenkov saturated events: {np.sum(cher_saturated)}/{len(cher_adc)}")
    
    
    # Basic mask: remove clearly invalid
    mask = (
        (beam_intensity > 1e5) &           # Reasonable beam
        (cher_calibrated > 0) &            # Positive signals
        (scint_calibrated > 0) &
        np.isfinite(beam_intensity) &
        np.isfinite(cher_calibrated) &
        np.isfinite(scint_calibrated) &
        (~cher_saturated)
    )
    
    print(f"\nBasic validity mask: {np.sum(mask)}/{len(beam_intensity)} events")
    
    if np.sum(mask) == 0:
        print("ERROR: No valid events!")
        return None
    
    # Normalize: photons per particle
    beam_valid = beam_intensity[mask]
    cher_response = cher_calibrated[mask] / beam_valid
    scint_response = scint_calibrated[mask] / beam_valid
    
    print(f"\nResponse per particle:")
    print(f"  Cherenkov: {cher_response.min():.2e} to {cher_response.max():.2e}")
    print(f"  Scintillator: {scint_response.min():.2e} to {scint_response.max():.2e}")
    
    # SIMPLE QUALITY CUTS 
    # Just remove extreme outliers
    
    quality_flag = np.ones(len(beam_valid), dtype=bool)
    
    # Remove events with response >100x or <0.01x median
    median_cher = np.median(cher_response)
    median_scint = np.median(scint_response)
    
    quality_flag &= (cher_response > median_cher * 0.1) & (cher_response < median_cher * 10)
    quality_flag &= (scint_response > median_scint * 0.1) & (scint_response < median_scint * 10)
    
    #  flag saturated Cherenkov events
    saturated_in_mask = cher_saturated[mask]
    print(f"  Saturated events in valid set: {np.sum(saturated_in_mask)}")
    
    print(f"\nQuality cuts: {np.sum(quality_flag)}/{len(beam_valid)} events pass")
    
    # Final clean data
    beam_clean = beam_valid[quality_flag]
    cher_clean = cher_response[quality_flag]
    scint_clean = scint_response[quality_flag]
    saturated_clean = saturated_in_mask[quality_flag]
    
    # 6. CREATE OUTPUT
    clean_data = {
        'intensity_calibrated': beam_clean,           # Beam particles/pulse
        'cherenkov_response': cher_clean,             # Cherenkov photons/particle
        'scintillator_response': scint_clean,         # Scintillator photons/particle
        'cherenkov_saturated': saturated_clean,       # Which events are saturated
        'normalization_factors': {
            'adc_conversion_cherenkov': float(ADC_CONV_CHER),
            'adc_conversion_scintillator': float(ADC_CONV_SCINT),
            'pmt_gain': float(PMT_GAIN),
            'quantum_efficiency': float(QE_CHER),
        },
        'quality_flags': quality_flag,
        'metadata': {
            'source_file': file_path,
            'run_id': run_name,
            'n_events_total': len(beam_intensity),
            'n_events_valid': np.sum(mask),
            'n_events_final': len(beam_clean),
            'n_saturated': int(np.sum(cher_saturated)),
            'processing_timestamp': datetime.now().isoformat(),
            'beam_min': float(beam_intensity.min()),
            'beam_max': float(beam_intensity.max()),
            'note': 'Normalized by beam_intensity (particles/pulse), not beam_current',
        }
    }
    
    # 7. PRINT SUMMARY
    print(f"\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Total events: {len(beam_intensity)}")
    print(f"Valid events: {np.sum(mask)}")
    print(f"Quality events: {len(beam_clean)}")
    print(f"Cherenkov saturated: {np.sum(cher_saturated)} events")
    
    if len(cher_clean) > 0:
        print(f"\nCherenkov (excluding saturated):")
        print(f"  Mean: {np.mean(cher_clean[~saturated_clean]):.2e} photons/particle")
        print(f"  Std: {np.std(cher_clean[~saturated_clean]):.2e}")
        print(f"  Variation: {np.std(cher_clean[~saturated_clean])/np.mean(cher_clean[~saturated_clean])*100:.1f}%")
        
        print(f"\nScintillator:")
        print(f"  Mean: {np.mean(scint_clean):.2e} photons/particle")
        print(f"  Std: {np.std(scint_clean):.2e}")
        print(f"  Variation: {np.std(scint_clean)/np.mean(scint_clean)*100:.1f}%")
        
        # Key insight: Scintillator response should DECREASE with intensity
        # Sort by intensity and check
        sorted_idx = np.argsort(beam_clean)
        sorted_beam = beam_clean[sorted_idx]
        sorted_scint = scint_clean[sorted_idx]
        
        print(f"\nScintillator saturation check:")
        print(f"  Low intensity ({sorted_beam[0]:.1e}): {sorted_scint[0]:.2e}")
        print(f"  High intensity ({sorted_beam[-1]:.1e}): {sorted_scint[-1]:.2e}")
        print(f"  Ratio (high/low): {sorted_scint[-1]/sorted_scint[0]:.3f}")
    
    print("="*60)
    
    return clean_data

def plot_results(clean_data):
    """Plot the results."""
    if clean_data is None or len(clean_data['intensity_calibrated']) == 0:
        print("No data to plot")
        return
    
    beam = clean_data['intensity_calibrated']
    chere = clean_data['cherenkov_response']
    scint = clean_data['scintillator_response']
    saturated = clean_data.get('cherenkov_saturated', np.zeros_like(beam, dtype=bool))
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Cherenkov vs Beam (log-log)
    ax = axes[0, 0]
    # Plot non-saturated in blue, saturated in red
    if np.any(~saturated):
        ax.scatter(beam[~saturated], chere[~saturated], alpha=0.7, s=50, 
                  color='blue', label='Not saturated')
    if np.any(saturated):
        ax.scatter(beam[saturated], chere[saturated], alpha=0.7, s=50,
                  color='red', marker='x', label='Saturated (ADC max)')
    ax.set_xlabel('Beam Intensity (particles/pulse)')
    ax.set_ylabel('Cherenkov Response (photons/particle)')
    ax.set_title('Cherenkov: Should be CONSTANT (linear)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    if np.any(saturated) or np.any(~saturated):
        ax.legend()
    
    # Plot 2: Scintillator vs Beam (log-log)
    ax = axes[0, 1]
    ax.scatter(beam, scint, alpha=0.7, s=50, color='red')
    ax.set_xlabel('Beam Intensity (particles/pulse)')
    ax.set_ylabel('Scintillator Response (photons/particle)')
    ax.set_title('Scintillator: Should DECREASE (saturation)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Cherenkov histogram
    ax = axes[1, 0]
    if np.any(~saturated):
        ax.hist(chere[~saturated], bins=min(20, len(chere[~saturated])//2), 
               alpha=0.7, color='blue')
        mean_cher = np.mean(chere[~saturated])
        ax.axvline(mean_cher, color='red', linestyle='--', 
                  label=f'Mean: {mean_cher:.2e}')
    ax.set_xlabel('Cherenkov Response (photons/particle)')
    ax.set_ylabel('Counts')
    ax.set_title('Cherenkov Response Distribution')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 4: Scintillator histogram
    ax = axes[1, 1]
    ax.hist(scint, bins=min(20, len(scint)//2), alpha=0.7, color='red')
    mean_scint = np.mean(scint)
    ax.axvline(mean_scint, color='blue', linestyle='--', 
              label=f'Mean: {mean_scint:.2e}')
    ax.set_xlabel('Scintillator Response (photons/particle)')
    ax.set_ylabel('Counts')
    ax.set_title('Scintillator Response Distribution')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.suptitle('FLASH Experiment: Cherenkov vs Scintillator', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # plot linear scale to see saturation clearly
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Cherenkov: response should be flat vs intensity
    ax1.scatter(beam, chere, alpha=0.7, s=50, color='blue')
    if len(chere) > 1:
        # Fit horizontal line
        mean_resp = np.mean(chere[~saturated]) if np.any(~saturated) else np.mean(chere)
        ax1.axhline(mean_resp, color='red', linestyle='--', 
                   label=f'Mean: {mean_resp:.2e}')
    ax1.set_xlabel('Beam Intensity')
    ax1.set_ylabel('Cherenkov Response')
    ax1.set_title('Cherenkov: Flat = Linear')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Scintillator: response should decrease
    ax2.scatter(beam, scint, alpha=0.7, s=50, color='red')
    ax2.set_xlabel('Beam Intensity')
    ax2.set_ylabel('Scintillator Response')
    ax2.set_title('Scintillator: Decreasing = Saturation')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def save_data(clean_data, filename="processed_data/clean_output.npz"):
    """Save the processed data."""
    if clean_data is None:
        return
    
    from pathlib import Path
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    
    np.savez(
        filename,
        intensity_calibrated=clean_data['intensity_calibrated'],
        cherenkov_response=clean_data['cherenkov_response'],
        scintillator_response=clean_data['scintillator_response'],
        cherenkov_saturated=clean_data.get('cherenkov_saturated', []),
        normalization_factors=clean_data['normalization_factors'],
        quality_flags=clean_data['quality_flags'],
        metadata=clean_data['metadata']
    )
    
    print(f"\nSaved to: {filename}")

# Main
if __name__ == "__main__":
    print("Cherenkov FLASH Processing Pipeline")
    print("="*60)
    
    data = process_latest_file()
    
    if data is not None and len(data['intensity_calibrated']) > 0:
        plot_results(data)
        save_data(data)
        
        print("sucess")
    else:
        print("\nProcessing failed or no data")


# EXPORTED FUNCTIONS FOR run_processing.py

def process_specific_file(file_path):
    """
    Same as process_latest_file() but takes file_path as argument.
    """
    print(f"Processing: {file_path}")
    
    with h5py.File(file_path, "r") as f:
        run_name = list(f.keys())[0]
        run = f[run_name]
        
        # Load datasets
        beam_intensity = run['beam_intensity'][:]
        cher_adc = run['cherenkov_adc'][:]
        scint_adc = run['scintillator_adc'][:]
    
    # Calibration factors
    ADC_CONV_CHER = 5e12
    ADC_CONV_SCINT = 1e8  
    PMT_GAIN = 1e6
    QE_CHER = 0.25
    
    # Calibration
    cher_electrons = cher_adc.astype(np.float64) * ADC_CONV_CHER
    cher_photons = cher_electrons / PMT_GAIN
    cher_calibrated = cher_photons / QE_CHER
    
    scint_calibrated = scint_adc.astype(np.float64) * ADC_CONV_SCINT
    
    # Check saturation
    cher_saturated = (cher_adc == 16383)
    
    # Normalize
    mask = (
        (beam_intensity > 1e5) &
        (cher_calibrated > 0) &
        (scint_calibrated > 0) &
        np.isfinite(beam_intensity) &
        np.isfinite(cher_calibrated) &
        np.isfinite(scint_calibrated)
    )
    
    if np.sum(mask) == 0:
        print(f"  No valid events")
        return None
    
    beam_valid = beam_intensity[mask]
    cher_response = cher_calibrated[mask] / beam_valid
    scint_response = scint_calibrated[mask] / beam_valid
    saturated_in_mask = cher_saturated[mask]
    
    quality_flag = np.ones(len(beam_valid), dtype=bool)
    median_cher = np.median(cher_response)
    median_scint = np.median(scint_response)
    
    quality_flag &= (cher_response > median_cher * 0.1) & (cher_response < median_cher * 10)
    quality_flag &= (scint_response > median_scint * 0.1) & (scint_response < median_scint * 10)
    
    # Final data
    beam_clean = beam_valid[quality_flag]
    cher_clean = cher_response[quality_flag]
    scint_clean = scint_response[quality_flag]
    saturated_clean = saturated_in_mask[quality_flag]
    
    # Create output
    clean_data = {
        'intensity_calibrated': beam_clean,
        'cherenkov_response': cher_clean,
        'scintillator_response': scint_clean,
        'cherenkov_saturated': saturated_clean,
        'normalization_factors': {
            'adc_conversion_cherenkov': float(ADC_CONV_CHER),
            'adc_conversion_scintillator': float(ADC_CONV_SCINT),
            'pmt_gain': float(PMT_GAIN),
            'quantum_efficiency': float(QE_CHER),
        },
        'quality_flags': quality_flag,
        'metadata': {
            'source_file': str(file_path),
            'run_id': run_name,
            'n_events_total': len(beam_intensity),
            'n_events_valid': np.sum(mask),
            'n_events_final': len(beam_clean),
            'n_saturated': int(np.sum(cher_saturated)),
            'processing_timestamp': datetime.now().isoformat(),
            'beam_min': float(beam_intensity.min()),
            'beam_max': float(beam_intensity.max()),
        }
    }
    
    print(f"  Processed: {len(beam_clean)}/{len(beam_intensity)} events")
    
    return clean_data



files = sorted(glob.glob("data/*.h5"))
if not files:
    print("no HDF5 files in data/ folder")
    print("   run: python -m src.daq.control simulate")
    exit()

file_path = files[-1]  # Most recent
print(f"Using: {file_path}")

with h5py.File(file_path, "r") as f:
    run_name = list(f.keys())[0]
    run = f[run_name]
    print(run.keys())
    
    beam = run['beam_intensity'][:]
    cher_adc = run['cherenkov_adc'][:]
    scint_adc = run['scintillator_adc'][:]

    print(f"Beam Intensity: {beam}")
    print(f"Cherenkov ADC: {cher_adc}")
    print(f"Scintilator ADC: {scint_adc}")
    


#ADC → electrons

ADC_CONV_CHER = 1.0
cher_electrons = cher_adc / ADC_CONV_CHER

# Dark current subtraction

DARK_CURRENT_RATE = 100        # Hz
INTEGRATION_TIME = 1e-9        # seconds

dark_electrons = DARK_CURRENT_RATE * INTEGRATION_TIME
cher_electrons_corr = cher_electrons - dark_electrons
cher_electrons_corr = np.clip(cher_electrons_corr, 0, None)

# PMT gain + QE correction

PMT_GAIN_CHER = 1e6
QE_CHER = 0.25

cher_photons = cher_electrons_corr / PMT_GAIN_CHER
cher_cal = cher_photons / QE_CHER   # final Cherenkov signal


# Scintillator calibration

ADC_CONV_SCINT = 1.0
scint_cal = scint_adc / ADC_CONV_SCINT


# Normalize by beam

mask = (beam > 0) & (cher_cal > 0) & (scint_cal > 0)

beam_n = beam[mask]
cher_resp = cher_cal[mask] / beam_n
scint_resp = scint_cal[mask] / beam_n


# Quality cuts

quality_flag = (
    np.isfinite(beam_n) &
    np.isfinite(cher_resp) &
    np.isfinite(scint_resp)
)

def robust_outlier_flag(data, sigma=3.0):
    med = np.median(data)
    mad = np.median(np.abs(data - med))
    if mad == 0:
        return np.ones_like(data, dtype=bool)
    return np.abs(data - med) < sigma * 1.4826 * mad

quality_flag &= robust_outlier_flag(cher_resp)
quality_flag &= robust_outlier_flag(scint_resp)

beam_clean = beam_n[quality_flag]
cher_clean = cher_resp[quality_flag]
scint_clean = scint_resp[quality_flag]
quality_flags = quality_flag

#ts just for checking
plt.figure(figsize=(9,4))
plt.plot(cher_cal, marker="o", label="Cherenkov (calibrated)")
plt.plot(scint_cal, marker="s", label="Scintillator (calibrated)")
plt.xlabel("Event index")
plt.ylabel("Calibrated signal (arb. units)")
plt.title("Calibrated detector signals per event")
plt.legend()
plt.grid(True)
plt.show()

