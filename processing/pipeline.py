import h5py
import numpy as np
import glob
import matplotlib.pyplot as plt


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
    print(run.keys())
    
    beam = run['beam_intensity'][:]
    cher_adc = run['cherenkov_adc'][:]
    scint_adc = run['scintillator_adc'][:]

    print(f"Beam Intensity: {beam}")
    print(f"Cherenkov ADC: {cher_adc}")
    print(f"Scintilator ADC: {scint_adc}")
    


#ADC → electrons

ADC_CONV_CHER = 1.0
cher_electrons = cher_adc / ADC_CONV_CHER

# Dark current subtraction

DARK_CURRENT_RATE = 100        # Hz
INTEGRATION_TIME = 1e-9        # seconds

dark_electrons = DARK_CURRENT_RATE * INTEGRATION_TIME
cher_electrons_corr = cher_electrons - dark_electrons
cher_electrons_corr = np.clip(cher_electrons_corr, 0, None)

# PMT gain + QE correction

PMT_GAIN_CHER = 1e6
QE_CHER = 0.25

cher_photons = cher_electrons_corr / PMT_GAIN_CHER
cher_cal = cher_photons / QE_CHER   # final Cherenkov signal


# Scintillator calibration

ADC_CONV_SCINT = 1.0
scint_cal = scint_adc / ADC_CONV_SCINT


# Normalize by beam

mask = (beam > 0) & (cher_cal > 0) & (scint_cal > 0)

beam_n = beam[mask]
cher_resp = cher_cal[mask] / beam_n
scint_resp = scint_cal[mask] / beam_n


# Quality cuts

quality_flag = (
    np.isfinite(beam_n) &
    np.isfinite(cher_resp) &
    np.isfinite(scint_resp)
)

def robust_outlier_flag(data, sigma=3.0):
    med = np.median(data)
    mad = np.median(np.abs(data - med))
    if mad == 0:
        return np.ones_like(data, dtype=bool)
    return np.abs(data - med) < sigma * 1.4826 * mad

quality_flag &= robust_outlier_flag(cher_resp)
quality_flag &= robust_outlier_flag(scint_resp)

beam_clean = beam_n[quality_flag]
cher_clean = cher_resp[quality_flag]
scint_clean = scint_resp[quality_flag]
quality_flags = quality_flag

#ts just for checking
plt.figure(figsize=(9,4))
plt.plot(cher_cal, marker="o", label="Cherenkov (calibrated)")
plt.plot(scint_cal, marker="s", label="Scintillator (calibrated)")
plt.xlabel("Event index")
plt.ylabel("Calibrated signal (arb. units)")
plt.title("Calibrated detector signals per event")
plt.legend()
plt.grid(True)
plt.show()
