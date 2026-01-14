"""
Main DAQ control system for Cherenkov experiment.
Coordinates hardware simulation, data acquisition, and run control.
"""

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import sys
import signal
import atexit
import numpy as np

#test
# Import our modules
try:
    from .hardware_sim import BeamSimulator, CherenkovDetectorSim, ScintillatorDetectorSim
    from .data_writer import HDF5DataWriter
    from .config import load_config
except ImportError:
    from hardware_sim import BeamSimulator, CherenkovDetectorSim, ScintillatorDetectorSim
    from data_writer import HDF5DataWriter
    from config import load_config

class DAQController:
    """Main controller for data acquisition."""
    
    def __init__(self, config_path: str = None):
        # Load configuration
        self.config = load_config(config_path) if config_path else {}
        
        # Initialize simulators
        self.beam_sim = BeamSimulator(
            beam_energy_mev=self.config.get('beam', {}).get('energy_mev', 100.0)
        )
        
        self.cherenkov_det = CherenkovDetectorSim(
            material=self.config.get('detectors', {}).get('cherenkov', {}).get('material', 'lead_glass')
        )
        
        self.scint_det = ScintillatorDetectorSim()
        
        # DAQ state
        self.is_running = False
        self.current_run_config = None
        self.data_writer = None
        self.total_events = 0
        self.current_intensity = 0.0
        
        # Register cleanup handlers
        atexit.register(self._cleanup)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _cleanup(self):
        """Cleanup resources on exit."""
        if self.data_writer:
            self.data_writer.close()
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully."""
        print(f"\nReceived signal {signum}, stopping...")
        self.stop_run()
        sys.exit(0)
    
    def configure_run(self, run_params: Dict) -> str:
        """Configure a new run. Returns run_id."""
        
        # Generate run ID
        run_id = run_params.get('run_id', 
                               datetime.now().strftime('%Y%m%d_%H%M%S'))
        
        # Build complete configuration
        self.current_run_config = {
            'run_id': run_id,
            'daq_version': '1.0.0',
            'start_time': datetime.now().isoformat(),
            'beam_energy_mev': self.beam_sim.beam_energy,
            'cherenkov_material': self.cherenkov_det.material,
            'config_file': str(Path(self.config.get('config_path', ''))),
            **run_params,
        }
        
        # Create output filename
        output_dir = Path(self.config.get('output', {}).get('directory', 'data'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = output_dir / f'cherenkov_run_{run_id}.h5'
        
        # Initialize data writer
        self.data_writer = HDF5DataWriter(str(filename))
        self.data_writer.create_run(self.current_run_config)
        
        self.total_events = 0
        self.is_running = False
        
        print(f"Configured run: {run_id}")
        print(f"Output file: {filename}")
        
        return run_id
    
    def start_acquisition(self, intensity: float, n_events: int) -> int:
        """
        Start data acquisition at a specific intensity.
        Returns number of events acquired.
        """
        if not self.data_writer:
            raise RuntimeError("Run not configured. Call configure_run() first.")
        
        self.is_running = True
        self.current_intensity = intensity
        
        print(f"Starting acquisition at intensity: {intensity:.2e} particles/pulse")
        print(f"  Target events: {n_events}")
        
        events_acquired = 0
        
        try:
            for event_idx in range(n_events):
                if not self.is_running:
                    break
                
                # Generate beam pulse with some jitter in intensity
                actual_intensity = intensity * np.random.normal(1.0, 0.02)
                beam_pulse = self.beam_sim.generate_pulse(actual_intensity)
                
                # Get detector responses
                cherenkov_data = self.cherenkov_det.detect_pulse(beam_pulse)
                scint_data = self.scint_det.detect_pulse(beam_pulse)
                
                # Create event data
                event_data = {
                    'timestamp': beam_pulse['timestamp'],
                    'beam_intensity': beam_pulse['intensity_actual'],
                    'cherenkov_adc': cherenkov_data['adc_counts_raw'],
                    'scintillator_adc': scint_data['adc_counts_raw'],
                    'beam_current': beam_pulse['beam_current_monitor'],
                }
                
                # Write to file
                event_id = self.data_writer.write_event(event_data)
                
                # Write detailed data for first few events
                if event_idx < 5:  # Only first 5 events per intensity step
                    self.data_writer.write_event_detailed(
                        self.total_events + event_idx,
                        beam_pulse,
                        cherenkov_data,
                        scint_data
                    )
                
                events_acquired += 1
                
                # Progress update
                if events_acquired % 10 == 0 and events_acquired > 0:
                    print(f"    Acquired: {events_acquired}/{n_events}")
                
                # Simulate time between pulses (1 ms for FLASH rates)
                time.sleep(0.001)
                
        except KeyboardInterrupt:
            print("\nAcquisition interrupted by user")
        except Exception as e:
            print(f"\nError during acquisition: {e}")
            raise
        finally:
            self.is_running = False
        
        return events_acquired
    
    def run_intensity_scan(self, 
                          intensity_values: List[float], 
                          events_per_step: int = 100,
                          step_delay: float = 0.5) -> Dict:
        """
        Run a full intensity scan (main experiment).
        
        Args:
            intensity_values: List of intensity values to scan
            events_per_step: Number of events to acquire at each intensity
            step_delay: Delay between intensity steps (seconds)
            
        Returns:
            Dictionary with scan results
        """
        print("=" * 60)
        print("STARTING INTENSITY SCAN EXPERIMENT")
        print("=" * 60)
        print(f"Intensity steps: {len(intensity_values)}")
        print(f"Events per step: {events_per_step}")
        print(f"Intensity range: {min(intensity_values):.2e} to {max(intensity_values):.2e}")
        print("=" * 60)
        
        # Configure run for the entire scan
        run_params = {
            'experiment_type': 'intensity_scan',
            'intensity_values': [float(v) for v in intensity_values],
            'events_per_step': events_per_step,
            'step_delay': step_delay,
            'description': 'Cherenkov vs Scintillator linearity scan',
        }
        
        run_id = self.configure_run(run_params)
        
        scan_results = {
            'run_id': run_id,
            'steps': [],
            'total_events': 0,
            'start_time': time.time(),
        }
        
        try:
            # Run through each intensity step
            for step_idx, intensity in enumerate(intensity_values):
                step_start_time = time.time()
                
                print(f"\n[Step {step_idx + 1}/{len(intensity_values)}]")
                print(f"  Intensity: {intensity:.2e} particles/pulse")
                print(f"  Events to acquire: {events_per_step}")
                
                # Record step start in data file
                self.data_writer.add_step_metadata(
                    step_idx=step_idx,
                    intensity=intensity,
                    n_events=events_per_step
                )
                
                # Acquire data at this intensity
                events_acquired = self.start_acquisition(intensity, events_per_step)
                
                # Update totals
                self.total_events += events_acquired
                scan_results['total_events'] = self.total_events
                
                # Record step results
                step_result = {
                    'step_index': step_idx,
                    'intensity': float(intensity),
                    'events_acquired': events_acquired,
                    'duration': time.time() - step_start_time,
                    'start_event_idx': step_idx * events_per_step,
                    'end_event_idx': step_idx * events_per_step + events_acquired - 1,
                }
                scan_results['steps'].append(step_result)
                
                print(f"  ✓ Completed: {events_acquired} events")
                print(f"  Duration: {step_result['duration']:.2f} seconds")
                print(f"  Events rate: {events_acquired/step_result['duration']:.1f} Hz")
                
                # Brief delay between steps (simulating beam tuning)
                if step_idx < len(intensity_values) - 1:
                    print(f"  Pausing for {step_delay:.1f}s before next step...")
                    time.sleep(step_delay)
            
            # Scan completed successfully
            scan_results['success'] = True
            scan_results['duration'] = time.time() - scan_results['start_time']
            
            print("\n" + "=" * 60)
            print("INTENSITY SCAN COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"Run ID: {run_id}")
            print(f"Total events acquired: {self.total_events}")
            print(f"Total duration: {scan_results['duration']:.2f} seconds")
            print(f"Average rate: {self.total_events/scan_results['duration']:.1f} events/sec")
            print("=" * 60)
            
        except Exception as e:
            print(f"\n Error during intensity scan: {e}")
            scan_results['success'] = False
            scan_results['error'] = str(e)
            raise
            
        finally:
            # Always close the writer
            if self.data_writer:
                self.data_writer.close()
        
        return scan_results
    
    def stop_run(self):
        """Stop data acquisition."""
        self.is_running = False
        print("Stopping acquisition...")
    
    def get_status(self) -> Dict:
        """Get current DAQ status."""
        return {
            'is_running': self.is_running,
            'run_id': self.current_run_config.get('run_id') if self.current_run_config else None,
            'total_events': self.total_events,
            'current_intensity': self.current_intensity,
            'config': {
                'beam_energy': self.beam_sim.beam_energy,
                'cherenkov_material': self.cherenkov_det.material,
            }
        }


def main():
    """Command-line interface for DAQ."""
    parser = argparse.ArgumentParser(
        description='Cherenkov DAQ Control System - FLASH Radiotherapy Testbed',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--config', '-c', type=str, 
                       default='configs/detector_default.yaml',
                       help='Configuration file path')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Single acquisition command
    run_parser = subparsers.add_parser('acquire', help='Run single acquisition')
    run_parser.add_argument('--events', '-n', type=int, default=1000,
                          help='Number of events to acquire')
    run_parser.add_argument('--intensity', '-i', type=float, default=1e9,
                          help='Beam intensity (particles/pulse)')
    run_parser.add_argument('--output', '-o', type=str,
                          help='Output directory (overrides config)')
    
    # Intensity scan command
    scan_parser = subparsers.add_parser('scan', help='Run intensity scan')
    scan_parser.add_argument('--min', type=float, default=1e6,
                           help='Minimum intensity')
    scan_parser.add_argument('--max', type=float, default=1e12,
                           help='Maximum intensity')
    scan_parser.add_argument('--steps', '-n', type=int, default=10,
                           help='Number of intensity steps (log spacing)')
    scan_parser.add_argument('--events-per-step', '-e', type=int, default=100,
                           help='Events per intensity step')
    scan_parser.add_argument('--output', '-o', type=str,
                           help='Output directory (overrides config)')
    
    # Simulate command
    sim_parser = subparsers.add_parser('simulate', help='Quick simulation test')
    sim_parser.add_argument('--events', '-n', type=int, default=10,
                          help='Number of events to simulate')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Check DAQ status')
    
    args = parser.parse_args()
    
    # Initialize DAQ
    daq = DAQController(args.config)
    
    # Override output directory if specified
    if hasattr(args, 'output') and args.output:
        daq.config['output']['directory'] = args.output
    
    if args.command == 'acquire':
        print(f"Starting single acquisition...")
        print(f"  Intensity: {args.intensity:.2e}")
        print(f"  Events: {args.events}")
        
        run_id = daq.configure_run({
            'run_type': 'single_acquisition',
            'intensity': args.intensity,
            'target_events': args.events,
        })
        
        events_acquired = daq.start_acquisition(args.intensity, args.events)
        
        print(f"\nAcquisition completed.")
        print(f"  Run ID: {run_id}")
        print(f"  Events acquired: {events_acquired}")
        
    elif args.command == 'scan':
        # Create log-spaced intensity values
        import numpy as np
        intensity_values = np.logspace(
            np.log10(args.min),
            np.log10(args.max),
            args.steps
        ).tolist()
        
        print(f"Preparing intensity scan...")
        print(f"  Intensity range: {args.min:.2e} to {args.max:.2e}")
        print(f"  Steps: {args.steps}")
        print(f"  Events per step: {args.events_per_step}")
        print(f"  Total events: {args.steps * args.events_per_step}")
        
        results = daq.run_intensity_scan(
            intensity_values=intensity_values,
            events_per_step=args.events_per_step
        )
        
        if results.get('success'):
            print(f"\n✓ Scan completed successfully!")
            print(f"  Output file: data/cherenkov_run_{results['run_id']}.h5")
            print(f"  Total events: {results['total_events']}")
        else:
            print(f"\n✗ Scan failed: {results.get('error', 'Unknown error')}")
            
    elif args.command == 'simulate':
        print("Running quick simulation test...")
        
        # Simple test: 3 intensity points
        import numpy as np
        intensity_values = np.logspace(7, 9, 3)  # 10^7, 10^8, 10^9
        
        results = daq.run_intensity_scan(
            intensity_values=intensity_values,
            events_per_step=5,  # Small number for quick test
            step_delay=0.1
        )
        
        if results.get('success'):
            print(f"\n✓ Simulation test passed!")
            print(f"  Run ID: {results['run_id']}")
            print(f"  Events: {results['total_events']}")
        else:
            print(f"\n✗ Simulation test failed")
            
    elif args.command == 'status':
        status = daq.get_status()
        print("DAQ Status:")
        print(f"  Running: {status['is_running']}")
        print(f"  Current run: {status['run_id']}")
        print(f"  Total events: {status['total_events']}")
        print(f"  Beam energy: {status['config']['beam_energy']} MeV")
        print(f"  Cherenkov material: {status['config']['cherenkov_material']}")
        
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python -m src.daq.control simulate")
        print("  python -m src.daq.control acquire --intensity 1e9 --events 100")
        print("  python -m src.daq.control scan --min 1e6 --max 1e12 --steps 10")


if __name__ == '__main__':
    main()
