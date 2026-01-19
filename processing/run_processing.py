#!/usr/bin/env python
"""
Batch processing using the pipeline.
"""

import sys
from pathlib import Path
import numpy as np
import glob

sys.path.insert(0, str(Path(__file__).parent.parent))

from processing.pipeline import process_specific_file

def main():
    """Process all files."""
    print("Batch Processing")
    print("="*60)
    
    files = sorted(glob.glob("data/*.h5"))
    if not files:
        print("No files found")
        return
    
    print(f"Found {len(files)} file(s)")
    
    all_data = []
    
    for file_path in files:
        data = process_specific_file(file_path)
        if data:
            all_data.append(data)
    
    # Save combined
    if all_data:
        combined_beam = np.concatenate([d['intensity_calibrated'] for d in all_data])
        combined_cher = np.concatenate([d['cherenkov_response'] for d in all_data])
        combined_scint = np.concatenate([d['scintillator_response'] for d in all_data])
        
        output_dir = Path("processed_data")
        output_dir.mkdir(exist_ok=True)
        
        np.savez(
            output_dir / "all_runs_combined.npz",
            beam_intensity=combined_beam,
            cherenkov_response=combined_cher,
            scintillator_response=combined_scint,
            run_ids=[d['metadata']['run_id'] for d in all_data]
        )
        
        print(f"\nCombined {len(all_data)} runs")
        print(f"   Total events: {len(combined_beam)}")

if __name__ == "__main__":
    main()