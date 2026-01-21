"""
Configuration management for DAQ system.
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    
    # Default configuration
    default_config = {
        'daq': {
            'sample_rate_hz': 1000,
            'buffer_size': 10000,
            'trigger_threshold': 50.0,
            'pre_trigger_samples': 100,
        },
        'beam': {
            'energy_mev': 100.0,
            'pulse_width_ns': 1.0,
            'repetition_rate_hz': 10,
        },
        'detectors': {
            'cherenkov': {
                'material': 'lead_glass',
                'pmt_voltage': -1000,  # V
                'gain': 1e6,
            },
            'scintillator': {
                'type': 'plastic',
                'birks_constant': 0.01,
            }
        },
        'output': {
            'directory': 'data',
            'format': 'hdf5',
            'compression': True,
        }
    }
    
    # If no config file specified, return defaults
    if config_path is None:
        return default_config
    
    # Load from file
    config_file = Path(config_path)
    if not config_file.exists():
        print(f"Warning: Config file {config_path} not found. Using defaults.")
        return default_config
    
    try:
        with open(config_file, 'r') as f:
            file_config = yaml.safe_load(f)
        
        # Merge with defaults (file config overrides defaults)
        merged_config = default_config.copy()
        
        def merge_dicts(base, override):
            for key, value in override.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    merge_dicts(base[key], value)
                else:
                    base[key] = value
        
        if file_config:
            merge_dicts(merged_config, file_config)
        
        return merged_config
        
    except yaml.YAMLError as e:
        print(f"Error parsing config file: {e}")
        return default_config


def save_config(config: Dict[str, Any], config_path: str):
    """Save configuration to YAML file."""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
