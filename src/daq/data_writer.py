"""
HDF5 data writer for Cherenkov experiment.
"""

import h5py
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import json
import time


class HDF5DataWriter:
    """Writes experiment data to HDF5 format with metadata."""
    
    def __init__(self, filename: str):
        self.filename = Path(filename)
        self.file = None
        self.run_group = None  # Reference to the current run group
        self.run_id = None
        
    def create_run(self, run_config: Dict):
        """Create a new run group in the HDF5 file."""
        
        # Ensure directory exists
        self.filename.parent.mkdir(parents=True, exist_ok=True)
        
        # Create file with latest libver for better performance
        self.file = h5py.File(self.filename, 'w', libver='latest')
        
        # Create run ID
        self.run_id = run_config.get('run_id', 
                                   datetime.now().strftime('%Y%m%d_%H%M%S'))
        
        # Create run group
        self.run_group = self.file.create_group(f'run_{self.run_id}')
        
        # Create metadata group
        meta_group = self.run_group.create_group('metadata')
        for key, value in run_config.items():
            if isinstance(value, (str, int, float, bool, type(None))):
                meta_group.attrs[key] = value
            elif isinstance(value, (list, dict)):
                # Store complex types as JSON strings
                meta_group.attrs[key] = json.dumps(value)
        
        # Add creation timestamp
        meta_group.attrs['creation_time'] = datetime.now().isoformat()
        meta_group.attrs['daq_version'] = '1.0.0'
        meta_group.attrs['file_path'] = str(self.filename)
        
        # Initialize empty datasets with chunking for performance
        chunk_size = 1000
        
        # Timestamps (int64 for nanosecond precision)
        self.run_group.create_dataset('timestamps', 
                                     shape=(0,), 
                                     maxshape=(None,), 
                                     dtype=np.int64,
                                     chunks=(chunk_size,))
        
        # Beam data
        self.run_group.create_dataset('beam_intensity', 
                                     shape=(0,), 
                                     maxshape=(None,), 
                                     dtype=np.float64,
                                     chunks=(chunk_size,))
        
        self.run_group.create_dataset('beam_current', 
                                     shape=(0,), 
                                     maxshape=(None,), 
                                     dtype=np.float64,
                                     chunks=(chunk_size,))
        
        # Detector signals
        self.run_group.create_dataset('cherenkov_adc', 
                                     shape=(0,), 
                                     maxshape=(None,), 
                                     dtype=np.int32,
                                     chunks=(chunk_size,))
        
        self.run_group.create_dataset('scintillator_adc', 
                                     shape=(0,), 
                                     maxshape=(None,), 
                                     dtype=np.int32,
                                     chunks=(chunk_size,))
        
        # Create events group for detailed event data
        self.run_group.create_group('events')
        
        # Initialize event counter
        self.run_group.attrs['total_events'] = 0
        self.run_group.attrs['last_update'] = time.time()
        
        print(f"Created run: {self.run_id} in {self.filename}")
        return self.run_id
        
    def write_event(self, event_data: Dict):
        """Write a single event to the run datasets."""
        if self.file is None or self.run_group is None:
            raise RuntimeError("No active run. Call create_run() first.")
        
        # Get current index
        current_idx = self.run_group.attrs['total_events']
        
        # Resize datasets if needed
        for dset_name in ['timestamps', 'beam_intensity', 'beam_current', 
                         'cherenkov_adc', 'scintillator_adc']:
            dset = self.run_group[dset_name]
            if current_idx >= dset.shape[0]:
                # Resize with extra buffer
                new_size = current_idx + 1000
                dset.resize((new_size,))
        
        # Write data to datasets
        self.run_group['timestamps'][current_idx] = event_data.get('timestamp', 0)
        self.run_group['beam_intensity'][current_idx] = event_data.get('beam_intensity', 0)
        self.run_group['beam_current'][current_idx] = event_data.get('beam_current', 0)
        self.run_group['cherenkov_adc'][current_idx] = event_data.get('cherenkov_adc', 0)
        self.run_group['scintillator_adc'][current_idx] = event_data.get('scintillator_adc', 0)
        
        # Update event counter
        self.run_group.attrs['total_events'] = current_idx + 1
        
        # Update timestamp
        self.run_group.attrs['last_update'] = time.time()
        
        # Flush every 100 events
        if current_idx % 100 == 0:
            self.file.flush()
            
        return current_idx
    
    def write_event_detailed(self, event_id: int, 
                            beam_data: Dict, 
                            cherenkov_data: Dict, 
                            scint_data: Dict):
        """Write detailed event data to separate group (for first N events)."""
        if self.file is None or self.run_group is None:
            raise RuntimeError("No active run. Call create_run() first.")
        
        # Only store detailed data for first 100 events to save space
        if event_id >= 100:
            return
        
        event_group = self.run_group['events'].create_group(f'event_{event_id:06d}')
        
        # Store beam parameters
        beam_group = event_group.create_group('beam')
        for key, value in beam_data.items():
            if isinstance(value, (int, float, str, bool)):
                beam_group.attrs[key] = value
            elif isinstance(value, dict):
                beam_group.attrs[key] = json.dumps(value)
        
        # Store Cherenkov detector data
        cherenkov_group = event_group.create_group('cherenkov')
        for key, value in cherenkov_data.items():
            if isinstance(value, (int, float, str, bool)):
                cherenkov_group.attrs[key] = value
            elif isinstance(value, dict):
                cherenkov_group.attrs[key] = json.dumps(value)
                
        # Store scintillator data
        scint_group = event_group.create_group('scintillator')
        for key, value in scint_data.items():
            if isinstance(value, (int, float, str, bool)):
                scint_group.attrs[key] = value
            elif isinstance(value, dict):
                scint_group.attrs[key] = json.dumps(value)
    
    def add_step_metadata(self, step_idx: int, intensity: float, n_events: int):
        """Add metadata for an intensity scan step."""
        if self.file is None or self.run_group is None:
            raise RuntimeError("No active run. Call create_run() first.")
        
        # Create steps group if it doesn't exist
        if 'steps' not in self.run_group:
            self.run_group.create_group('steps')
        
        # Create step group
        step_group = self.run_group['steps'].create_group(f'step_{step_idx:03d}')
        step_group.attrs['step_index'] = step_idx
        step_group.attrs['intensity_setting'] = float(intensity)
        step_group.attrs['n_events'] = n_events
        step_group.attrs['start_event_idx'] = self.run_group.attrs['total_events'] - n_events
        step_group.attrs['end_event_idx'] = self.run_group.attrs['total_events'] - 1
        step_group.attrs['timestamp'] = time.time()
    
    def close(self):
        """Close the HDF5 file properly."""
        if self.file:
            try:
                # Trim datasets to actual size
                if self.run_group:
                    total_events = self.run_group.attrs.get('total_events', 0)
                    
                    # Trim all datasets
                    for dset_name in ['timestamps', 'beam_intensity', 'beam_current',
                                     'cherenkov_adc', 'scintillator_adc']:
                        if dset_name in self.run_group:
                            dset = self.run_group[dset_name]
                            if total_events < dset.shape[0]:
                                dset.resize((total_events,))
                
                # Add final metadata
                self.run_group.attrs['completion_time'] = datetime.now().isoformat()
                self.run_group.attrs['file_size_bytes'] = self.filename.stat().st_size
                
                self.file.close()
                print(f"Closed file: {self.filename} (events: {total_events})")
                
            except Exception as e:
                print(f"Warning: Error closing file {self.filename}: {e}")
                try:
                    self.file.close()
                except:
                    pass
            finally:
                self.file = None
                self.run_group = None
                self.run_id = None
    
    def __del__(self):
        """Ensure file is closed on destruction."""
        if self.file:
            self.close()
