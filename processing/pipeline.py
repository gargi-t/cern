import h5py
import numpy as np
import glob

# Find latest file
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
    
    print(f"\nRun: {run_name}")
    print("Available datasets:")
    for key in run.keys():
        if isinstance(run[key], h5py.Dataset):
            print(f"  {key}")
    
    # Get data
    beam = run['beam_intensity'][:]
    cher = run['cherenkov_adc'][:]  
    scint = run['scintillator_adc'][:]  
    
    print(f"\nBeam: {beam.shape}, {beam.min():.2e} to {beam.max():.2e}")
    print(f"Cherenkov: {cher.shape}, min={cher.min()}, max={cher.max()}")
    print(f"Scintillator: {scint.shape}, min={scint.min()}, max={scint.max()}")