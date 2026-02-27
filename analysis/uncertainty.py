"""
Uncertainty analysis for FLASH experiment.
Calculate statistical and systematic errors.
"""

import numpy as np
from scipy import stats
import uncertainties as unc
from uncertainties import unumpy

class UncertaintyAnalyzer:
    """Analyze uncertainties in measurements."""
    
    def __init__(self, experimental_data):
        """
        Initialize with experimental data.
        
        Args:
            experimental_data: Dict with 'beam', 'chere', 'scint' arrays
        """
        self.beam = experimental_data['beam']
        self.chere = experimental_data['chere']
        self.scint = experimental_data['scint']
        
        # Calibration uncertainties (typical values)
        self.uncertainties = {
            'adc_counts': 0.01,           # 1% ADC counting error
            'pmt_gain': 0.05,             # 5% PMT gain uncertainty
            'quantum_efficiency': 0.10,   # 10% QE uncertainty
            'beam_intensity': 0.02,       # 2% beam intensity uncertainty
            'birks_constant': 0.20,       # 20% Birks' constant uncertainty
            'dark_current': 0.01,         # 1% dark current uncertainty
        }
    
    def statistical_uncertainties(self):
        """
        Calculate statistical uncertainties.
        
        Returns:
            Dict with mean, std, SEM for each detector
        """
        # Basic statistics
        cher_mean = np.mean(self.chere)
        cher_std = np.std(self.chere)
        cher_sem = cher_std / np.sqrt(len(self.chere))  # Standard error of mean
        
        scint_mean = np.mean(self.scint)
        scint_std = np.std(self.scint)
        scint_sem = scint_std / np.sqrt(len(self.scint))
        
        # Bootstrap confidence intervals
        cher_ci = self._bootstrap_ci(self.chere)
        scint_ci = self._bootstrap_ci(self.scint)
        
        return {
            'cherenkov': {
                'mean': cher_mean,
                'std': cher_std,
                'sem': cher_sem,
                'relative_std': cher_std / cher_mean * 100,
                'relative_sem': cher_sem / cher_mean * 100,
                'confidence_95': cher_ci,
            },
            'scintillator': {
                'mean': scint_mean,
                'std': scint_std,
                'sem': scint_sem,
                'relative_std': scint_std / scint_mean * 100,
                'relative_sem': scint_sem / scint_mean * 100,
                'confidence_95': scint_ci,
            }
        }
    
    def _bootstrap_ci(self, data, n_bootstrap=1000, confidence=0.95):
        """Calculate bootstrap confidence interval."""
        bootstrap_means = []
        n = len(data)
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            sample = np.random.choice(data, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))
        
        lower = np.percentile(bootstrap_means, (1 - confidence) / 2 * 100)
        upper = np.percentile(bootstrap_means, (1 + confidence) / 2 * 100)
        
        return {
            'lower': lower,
            'upper': upper,
            'width': upper - lower,
            'relative_width': (upper - lower) / np.mean(data) * 100,
        }
    
    def systematic_uncertainties(self):
        """
        Calculate systematic uncertainties from calibration errors.
        
        Uses error propagation through calibration chain.
        """
        # Define calibration parameters with uncertainties
        ADC_CONV_CHER = unc.ufloat(5e12, 5e12 * self.uncertainties['adc_counts'])
        ADC_CONV_SCINT = unc.ufloat(1e8, 1e8 * self.uncertainties['adc_counts'])
        PMT_GAIN = unc.ufloat(1e6, 1e6 * self.uncertainties['pmt_gain'])
        QE = unc.ufloat(0.25, 0.25 * self.uncertainties['quantum_efficiency'])
        
        # Typical beam intensity for calculation
        beam_typical = unc.ufloat(np.mean(self.beam), 
                                 np.mean(self.beam) * self.uncertainties['beam_intensity'])
        
        # Cherenkov calibration chain with uncertainties
        # ADC → electrons → photons → calibrated
        adc_typical = 1000  # Typical ADC count
        adc_cher = unc.ufloat(adc_typical, adc_typical * self.uncertainties['adc_counts'])
        
        # Calibration steps
        electrons_cher = adc_cher * ADC_CONV_CHER
        photons_cher = electrons_cher / PMT_GAIN
        calibrated_cher = photons_cher / QE
        
        # Normalize by beam
        response_cher = calibrated_cher / beam_typical
        
        # Scintillator calibration (simpler)
        adc_scint = unc.ufloat(1000, 1000 * self.uncertainties['adc_counts'])
        calibrated_scint = adc_scint * ADC_CONV_SCINT
        response_scint = calibrated_scint / beam_typical
        
        # Birks' law uncertainty
        kB = unc.ufloat(0.01, 0.01 * self.uncertainties['birks_constant'])
        L0 = 10000
        dEdx = 2.0
        
        # Birks' law with uncertainty
        light_yield = L0 * dEdx / (1 + kB * dEdx)
        
        return {
            'cherenkov_response': {
                'nominal': response_cher.nominal_value,
                'uncertainty': response_cher.std_dev,
                'relative_uncertainty': response_cher.std_dev / response_cher.nominal_value * 100,
                'error_breakdown': {
                    'adc': (adc_cher.std_dev / adc_cher.nominal_value * 100),
                    'adc_conversion': (ADC_CONV_CHER.std_dev / ADC_CONV_CHER.nominal_value * 100),
                    'pmt_gain': (PMT_GAIN.std_dev / PMT_GAIN.nominal_value * 100),
                    'qe': (QE.std_dev / QE.nominal_value * 100),
                    'beam_intensity': (beam_typical.std_dev / beam_typical.nominal_value * 100),
                }
            },
            'scintillator_response': {
                'nominal': response_scint.nominal_value,
                'uncertainty': response_scint.std_dev,
                'relative_uncertainty': response_scint.std_dev / response_scint.nominal_value * 100,
            },
            'birks_law': {
                'light_yield': light_yield.nominal_value,
                'uncertainty': light_yield.std_dev,
                'relative_uncertainty': light_yield.std_dev / light_yield.nominal_value * 100,
            }
        }
    
    def total_uncertainty(self):
        """
        Combine statistical and systematic uncertainties.
        
        Returns total uncertainty for each measurement.
        """
        stat = self.statistical_uncertainties()
        syst = self.systematic_uncertainties()
        
        # Combine using root sum of squares
        cher_stat_rel = stat['cherenkov']['relative_sem']  # Use SEM for mean
        cher_syst_rel = syst['cherenkov_response']['relative_uncertainty']
        cher_total_rel = np.sqrt(cher_stat_rel**2 + cher_syst_rel**2)
        
        scint_stat_rel = stat['scintillator']['relative_sem']
        scint_syst_rel = syst['scintillator_response']['relative_uncertainty']
        scint_total_rel = np.sqrt(scint_stat_rel**2 + scint_syst_rel**2)
        
        return {
            'cherenkov': {
                'statistical_relative': cher_stat_rel,
                'systematic_relative': cher_syst_rel,
                'total_relative': cher_total_rel,
                'mean_with_uncertainty': f"{stat['cherenkov']['mean']:.2e} ± {cher_total_rel:.1f}%",
            },
            'scintillator': {
                'statistical_relative': scint_stat_rel,
                'systematic_relative': scint_syst_rel,
                'total_relative': scint_total_rel,
                'mean_with_uncertainty': f"{stat['scintillator']['mean']:.2e} ± {scint_total_rel:.1f}%",
            },
            'dominant_uncertainty': {
                'cherenkov': 'systematic' if cher_syst_rel > cher_stat_rel else 'statistical',
                'scintillator': 'systematic' if scint_syst_rel > scint_stat_rel else 'statistical',
            }
        }
    
    def plot_uncertainties(self):
        """Create uncertainty visualization."""
        import matplotlib.pyplot as plt
        
        total = self.total_uncertainty()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Cherenkov uncertainty breakdown
        labels = ['Statistical', 'Systematic', 'Total']
        cher_values = [
            total['cherenkov']['statistical_relative'],
            total['cherenkov']['systematic_relative'],
            total['cherenkov']['total_relative']
        ]
        
        bars1 = ax1.bar(labels, cher_values, color=['blue', 'red', 'green'])
        ax1.set_ylabel('Relative Uncertainty (%)')
        ax1.set_title('Cherenkov Detector Uncertainty Breakdown')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        # Scintillator uncertainty breakdown
        scint_values = [
            total['scintillator']['statistical_relative'],
            total['scintillator']['systematic_relative'],
            total['scintillator']['total_relative']
        ]
        
        bars2 = ax2.bar(labels, scint_values, color=['blue', 'red', 'green'])
        ax2.set_ylabel('Relative Uncertainty (%)')
        ax2.set_title('Scintillator Detector Uncertainty Breakdown')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        return fig


# Example usage
if __name__ == "__main__":
    print("Uncertainty Analysis for FLASH Experiment")
    print("="*60)
    
    # Create sample data
    sample_data = {
        'beam': np.random.normal(1e9, 1e8, 100),
        'chere': np.random.normal(2000, 100, 100),
        'scint': np.random.normal(1000, 200, 100),
    }
    
    analyzer = UncertaintyAnalyzer(sample_data)
    
    print("\nStatistical Uncertainties:")
    stat = analyzer.statistical_uncertainties()
    print(f"  Cherenkov: {stat['cherenkov']['mean']:.2e} ± {stat['cherenkov']['relative_sem']:.1f}% (SEM)")
    print(f"  Scintillator: {stat['scintillator']['mean']:.2e} ± {stat['scintillator']['relative_sem']:.1f}% (SEM)")
    
    print("\nSystematic Uncertainties:")
    syst = analyzer.systematic_uncertainties()
    print(f"  Cherenkov: {syst['cherenkov_response']['relative_uncertainty']:.1f}%")
    print(f"  Scintillator: {syst['scintillator_response']['relative_uncertainty']:.1f}%")
    
    print("\nTotal Uncertainties:")
    total = analyzer.total_uncertainty()
    print(f"  Cherenkov: {total['cherenkov']['mean_with_uncertainty']}")
    print(f"  Scintillator: {total['scintillator']['mean_with_uncertainty']}")