
import h5py
import numpy as np
import glob
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.visualization import ExperimentVisualizer
from src.constants import ADC_CONVERSION


def process_latest_file():
    """Process the most recent HDF5 file."""
    
    # Find and read HDF5
    files = sorted(glob.glob("data/*.h5"))
    if not files:
        print("No HDF5 files in data/ folder")
        print("   Run: python -m src.daq.control simulate")
        return None
    
    file_path = files[-1]
    print(f"Processing: {file_path}")
    
    with h5py.File(file_path, "r") as f:
        run_name = list(f.keys())[0]
        run = f[run_name]
        
        # Load datasets 
        beam_intensity = run['beam_intensity'][:]    # Particles/pulse
        cher_adc = run['cherenkov_adc'][:]           # ADC counts
        scint_adc = run['scintillator_adc'][:]       # ADC counts
    
    print(f"\nRaw data: {len(beam_intensity)} events")
    print(f"Beam intensity range: {beam_intensity.min():.2e} to {beam_intensity.max():.2e}")
    print(f"Cherenkov ADC: {cher_adc.min()} to {cher_adc.max()} (max 16383 = saturation)")
    print(f"Scintillator ADC: {scint_adc.min()} to {scint_adc.max()}")
    
    # Get calibration constants from central source
    ADC_CONV_CHER = ADC_CONVERSION['cherenkov']['electrons_per_adc']  # 5e12
    ADC_CONV_SCINT = ADC_CONVERSION['scintillator']['photons_per_adc']  # 1e8
    PMT_GAIN = ADC_CONVERSION['cherenkov']['pmt_gain']  # 1e6
    QE_CHER = ADC_CONVERSION['cherenkov']['quantum_efficiency']  # 0.25
    
    print(f"\nCalibration:")
    print(f"  Cherenkov: {ADC_CONV_CHER:.1e} electrons/ADC → ÷{PMT_GAIN:.0e} ÷{QE_CHER} = photons")
    print(f"  Scintillator: {ADC_CONV_SCINT:.1e} photons/ADC")
    
    # Cherenkov calibration: ADC → electrons → photons
    cher_electrons = cher_adc.astype(np.float64) * ADC_CONV_CHER
    cher_photons = cher_electrons / PMT_GAIN
    cher_calibrated = cher_photons / QE_CHER  # Final Cherenkov photons
    
    # Scintillator calibration: ADC → photons directly
    scint_calibrated = scint_adc.astype(np.float64) * ADC_CONV_SCINT
    
    # Identify saturated events (but DON'T exclude them)
    cher_saturated = (cher_adc == 16383)  # 14-bit ADC max
    print(f"\nCherenkov saturated events: {np.sum(cher_saturated)}/{len(cher_adc)} (will be flagged, not excluded)")
    
    # Basic validity mask - REMOVED saturation exclusion
    mask = (
        (beam_intensity > 1e5) &           # Reasonable beam
        (cher_calibrated >= 0) &            # Non-negative signals
        (scint_calibrated >= 0) &
        np.isfinite(beam_intensity) &
        np.isfinite(cher_calibrated) &
        np.isfinite(scint_calibrated)
        # (~cher_saturated)  ← REMOVED - saturated events are important data!
    )
    
    print(f"\nBasic validity mask: {np.sum(mask)}/{len(beam_intensity)} events")
    
    if np.sum(mask) == 0:
        print("ERROR: No valid events!")
        return None
    
    # Normalize: photons per particle
    beam_valid = beam_intensity[mask]
    cher_response = cher_calibrated[mask] / beam_valid
    scint_response = scint_calibrated[mask] / beam_valid
    
    print(f"\nResponse per particle (before quality cuts):")
    print(f"  Cherenkov: {cher_response.min():.2e} to {cher_response.max():.2e}")
    print(f"  Scintillator: {scint_response.min():.2e} to {scint_response.max():.2e}")
    
    # Quality cuts - VERY loose, only remove unphysical outliers
    quality_flag = np.ones(len(beam_valid), dtype=bool)
    
    # Remove only events with response < 0 (shouldn't happen after mask)
    quality_flag &= (cher_response >= 0)
    quality_flag &= (scint_response >= 0)
    
    # Keep data within 0.01x to 100x of median (very loose)
    median_cher = np.median(cher_response)
    median_scint = np.median(scint_response)
    
    quality_flag &= (cher_response > median_cher * 0.001) & (cher_response < median_cher * 1000)
    quality_flag &= (scint_response > median_scint * 0.001) & (scint_response < median_scint * 1000)
    
    # Track which events were saturated
    saturated_in_mask = cher_saturated[mask]
    
    print(f"\nQuality cuts: {np.sum(quality_flag)}/{len(beam_valid)} events pass")
    print(f"  Saturated events in passing set: {np.sum(saturated_in_mask[quality_flag])}")
    
    # Final clean data
    beam_clean = beam_valid[quality_flag]
    cher_clean = cher_response[quality_flag]
    scint_clean = scint_response[quality_flag]
    saturated_clean = saturated_in_mask[quality_flag]
    
    # Create output dictionary
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
            'n_events_valid': int(np.sum(mask)),
            'n_events_final': len(beam_clean),
            'n_saturated': int(np.sum(cher_saturated)),
            'n_saturated_in_final': int(np.sum(saturated_clean)),
            'processing_timestamp': datetime.now().isoformat(),
            'beam_min': float(beam_intensity.min()),
            'beam_max': float(beam_intensity.max()),
        }
    }
    
    # Print summary
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Total events: {len(beam_intensity)}")
    print(f"Valid events (after basic cuts): {np.sum(mask)}")
    print(f"Final events (after quality cuts): {len(beam_clean)}")
    print(f"Cherenkov saturated (total): {np.sum(cher_saturated)}")
    print(f"Cherenkov saturated (in final): {np.sum(saturated_clean)}")
    
    if len(cher_clean) > 0:
        print(f"\nCherenkov (final clean data):")
        print(f"  Mean: {np.mean(cher_clean):.2e} photons/particle")
        print(f"  Std: {np.std(cher_clean):.2e}")
        print(f"  Variation: {np.std(cher_clean)/np.mean(cher_clean)*100:.1f}%")
        print(f"  Intensity range: {beam_clean.min():.2e} to {beam_clean.max():.2e}")
        
        print(f"\nScintillator (final clean data):")
        print(f"  Mean: {np.mean(scint_clean):.2e} photons/particle")
        print(f"  Std: {np.std(scint_clean):.2e}")
        print(f"  Variation: {np.std(scint_clean)/np.mean(scint_clean)*100:.1f}%")
        
        # Check scintillator saturation by comparing low vs high intensity
        if len(beam_clean) > 10:
            sorted_idx = np.argsort(beam_clean)
            sorted_beam = beam_clean[sorted_idx]
            sorted_scint = scint_clean[sorted_idx]
            
            # Take lowest 10% and highest 10%
            n_low = max(1, len(sorted_beam) // 10)
            n_high = max(1, len(sorted_beam) // 10)
            low_scint = np.mean(sorted_scint[:n_low])
            high_scint = np.mean(sorted_scint[-n_high:])
            
            print(f"\nScintillator saturation check:")
            print(f"  Low intensity (mean of lowest 10%): {low_scint:.2e} at ~{sorted_beam[:n_low].mean():.2e}")
            print(f"  High intensity (mean of highest 10%): {high_scint:.2e} at ~{sorted_beam[-n_high:].mean():.2e}")
            print(f"  Ratio (high/low): {high_scint/low_scint:.3f}")
            print(f"  Signal loss: {(1 - high_scint/low_scint)*100:.1f}%")
    
    print("="*60)
    
    return clean_data


def plot_results(clean_data):
    """Plot the results using centralized visualizer."""
    if clean_data is None or len(clean_data['intensity_calibrated']) == 0:
        print("No data to plot")
        return
    
    beam = clean_data['intensity_calibrated']
    chere = clean_data['cherenkov_response']
    scint = clean_data['scintillator_response']
    
    viz = ExperimentVisualizer()
    viz.plot_cherenkov_linearity(beam, chere, show=True)
    viz.plot_scintillator_saturation(beam, scint, show=True)
    viz.plot_distributions(chere, scint, show=True)
    viz.plot_detector_comparison(beam, chere, scint, show=True)


def save_data(clean_data, filename="processed_data/clean_output.npz"):
    """Save the processed data."""
    if clean_data is None:
        return
    
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    
    np.savez(
        filename,
        beam_intensity=clean_data['intensity_calibrated'],
        cherenkov_response=clean_data['cherenkov_response'],
        scintillator_response=clean_data['scintillator_response'],
        cherenkov_saturated=clean_data.get('cherenkov_saturated', []),
        normalization_factors=clean_data['normalization_factors'],
        quality_flags=clean_data['quality_flags'],
        metadata=clean_data['metadata']
    )
    
    print(f"\nSaved to: {filename}")


def process_specific_file(file_path):
    """
    Process a specific file by path.
    Used by run_processing.py for batch processing.
    """
    print(f"Processing: {file_path}")
    
    with h5py.File(file_path, "r") as f:
        run_name = list(f.keys())[0]
        run = f[run_name]
        
        beam_intensity = run['beam_intensity'][:]
        cher_adc = run['cherenkov_adc'][:]
        scint_adc = run['scintillator_adc'][:]
    
    # Get calibration constants
    ADC_CONV_CHER = ADC_CONVERSION['cherenkov']['electrons_per_adc']
    ADC_CONV_SCINT = ADC_CONVERSION['scintillator']['photons_per_adc']
    PMT_GAIN = ADC_CONVERSION['cherenkov']['pmt_gain']
    QE_CHER = ADC_CONVERSION['cherenkov']['quantum_efficiency']
    
    # Calibration
    cher_electrons = cher_adc.astype(np.float64) * ADC_CONV_CHER
    cher_photons = cher_electrons / PMT_GAIN
    cher_calibrated = cher_photons / QE_CHER
    
    scint_calibrated = scint_adc.astype(np.float64) * ADC_CONV_SCINT
    
    # Check saturation (but DON'T exclude)
    cher_saturated = (cher_adc == 16383)
    
    # Basic mask - REMOVED saturation exclusion
    mask = (
        (beam_intensity > 1e5) &
        (cher_calibrated >= 0) &
        (scint_calibrated >= 0) &
        np.isfinite(beam_intensity) &
        np.isfinite(cher_calibrated) &
        np.isfinite(scint_calibrated)
        # (~cher_saturated)  ← REMOVED
    )
    
    if np.sum(mask) == 0:
        print(f"  No valid events")
        return None
    
    beam_valid = beam_intensity[mask]
    cher_response = cher_calibrated[mask] / beam_valid
    scint_response = scint_calibrated[mask] / beam_valid
    saturated_in_mask = cher_saturated[mask]
    
    # Quality cuts - very loose
    quality_flag = np.ones(len(beam_valid), dtype=bool)
    
    # Keep data within 0.001x to 1000x of median (extremely loose)
    median_cher = np.median(cher_response)
    median_scint = np.median(scint_response)
    
    quality_flag &= (cher_response > median_cher * 0.001) & (cher_response < median_cher * 1000)
    quality_flag &= (scint_response > median_scint * 0.001) & (scint_response < median_scint * 1000)
    
    # Final data
    beam_clean = beam_valid[quality_flag]
    cher_clean = cher_response[quality_flag]
    scint_clean = scint_response[quality_flag]
    saturated_clean = saturated_in_mask[quality_flag]
    
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
        'metadata': {
            'source_file': str(file_path),
            'run_id': run_name,
            'n_events_total': len(beam_intensity),
            'n_events_final': len(beam_clean),
            'n_saturated': int(np.sum(cher_saturated)),
            'n_saturated_in_final': int(np.sum(saturated_clean)),
            'processing_timestamp': datetime.now().isoformat(),
        }
    }
    
    print(f"  Processed: {len(beam_clean)}/{len(beam_intensity)} events")
    print(f"  Saturated in final: {np.sum(saturated_clean)}")
    
    return clean_data


# Main execution
if __name__ == "__main__":
    print("Cherenkov FLASH Processing Pipeline")
    print("="*60)
    
    data = process_latest_file()
    
    if data is not None and len(data['intensity_calibrated']) > 0:
        plot_results(data)
        save_data(data, filename="processed_data/all_runs_combined.npz")
        print("\n✅ Processing successful!")
    else:
        print("\n❌ Processing failed or no data")
