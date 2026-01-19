def test_hdf5_writer():
    """Test HDF5 writer creates valid files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = Path(tmpdir) / "test.h5"
        writer = HDF5DataWriter(str(filename))
        
        # Create a run
        writer.create_run({
            'run_id': 'test_001',
            'experiment': 'Cherenkov_test',
            'beam_energy': 100.0
        })
        
        # Write some events
        for i in range(5):
            event_data = {
                'timestamp': i * 1000,
                'beam_intensity': float(i * 1e9),
                'cherenkov_signal': float(i * 1000),
                'scintillator_signal': float(i * 800),
                'beam_current': float(i * 1.6e-10)
            }
            writer.write_event(event_data)
        
        writer.close()
        
        # Verify file was created
        assert filename.exists()
        
        # Verify content
        with h5py.File(filename, 'r') as f:
            assert 'run_test_001' in f
            run = f['run_test_001']
            
            # Check metadata in metadata group
            assert 'metadata' in run
            assert run['metadata'].attrs['run_id'] == 'test_001'
            
            # Check datasets exist
            assert 'timestamps' in run
            assert 'beam_intensity' in run
            assert 'cherenkov_signal' in run
            assert 'scintillator_signal' in run
            assert 'beam_current' in run
            
            # Check data
            assert len(run['timestamps']) == 5
            assert run['beam_intensity'][4] == 4e9
            assert run.attrs['next_index'] == 5