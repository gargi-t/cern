import numpy as np
import h5py
import glob

# Check the raw files directly
files = sorted(glob.glob("data/*.h5"))
print(f"Found {len(files)} files")

for i, file_path in enumerate(files[-3:]):  # Last 3 files
    print(f"\nFile {i+1}: {file_path}")
    with h5py.File(file_path, "r") as f:
        run_name = list(f.keys())[0]
        run = f[run_name]
        
        beam = run['beam_intensity'][:]
        cher_adc = run['cherenkov_adc'][:]
        
        print(f"  Events: {len(beam)}")
        print(f"  Beam range: {beam.min():.2e} to {beam.max():.2e}")
        print(f"  Cherenkov ADC range: {cher_adc.min()} to {cher_adc.max()}")
        print(f"  Saturated events: {np.sum(cher_adc == 16383)}")