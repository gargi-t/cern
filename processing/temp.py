#we must debug
#we must debug
#we must debug
#we must debug
#we must debug
#we must debug
#we must debug
#we must debug
#we must debug

import h5py
import numpy as np

filepath = 'data/cherenkov_run_20260114_200638.h5'
with h5py.File('data/cherenkov_run_20260114_200638.h5', 'r') as f:
    run = f[list(f.keys())[0]]
    
    print(f"File: {filepath}")
    print(f"Run: {run.name}")
    
   
    print("\nuuu we check dataset woawo checking fataset rn ‼️‼️‼️ 'cherenkov_adc' dataset:")
    cher_dset = run['cherenkov_adc']
    print(f"  Shape: {cher_dset.shape}")
    print(f"  Dtype: {cher_dset.dtype}")
    print(f"  Chunks: {cher_dset.chunks}")
    
    cher_data = cher_dset[:]
    print(f"  Data min: {cher_data.min()}")
    print(f"  Data max: {cher_data.max()}")
    print(f"  Data mean: {cher_data.mean():.6f}")
    print(f"  All zeros? {np.all(cher_data == 0)}")
    print(f"  First 5 values: {cher_data[:5]}")
    
    print("\nnow scintillator ‼️‼️‼️ checking yay'scintillator_adc' dataset:")
    scint_dset = run['scintillator_adc']
    scint_data = scint_dset[:]
    print(f"  Shape: {scint_data.shape}")
    print(f"  Min: {scint_data.min()}, Max: {scint_data.max()}")
    print(f"  First 5: {scint_data[:5]}")
    
    print("\nChecking 'beam_intensity':")
    beam_data = run['beam_intensity'][:]
    print(f"  Min: {beam_data.min():.2e}, Max: {beam_data.max():.2e}")
    print(f"  First 5: {beam_data[:5]}")