# cern
A Cherenkov-Based Optical Testbed for Non-Saturating Beam Diagnostics in FLASH Radiotherapy
# Cherenkov-Based Optical Testbed for FLASH Radiotherapy

## Project Overview
This repository contains the Data Acquisition (DAQ) and control system for the Cherenkov-based beam diagnostics experiment proposed for CERN test beam facilities.

## Repository Structure
- `src/daq/`: Core DAQ control and data acquisition modules
- `configs/`: Detector and run configuration files (YAML format)
- `tests/`: Unit tests for DAQ components
- `notebooks/`: Prototyping and exploratory analysis

## Quick Start
1. Install: `pip install -r requirements.txt`
2. Run simulation: `python -m src.daq.control --simulate --intensity 1e10 --events 1000`
3. View data: See notebooks/01_daq_prototype.ipynb

## Data Format
Raw data is stored in HDF5 format with the following structure:
- `/run_metadata/`: Timestamp, run configuration, hardware settings
- `/events/`: Time-series data per event (beam monitors, PMT signals)
- `/triggers/`: Trigger timestamps and types
