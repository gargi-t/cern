"""
Centralized visualization for Cherenkov FLASH Experiment.
All plotting functions in one place for consistency.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from src.constants import PATHS


class ExperimentVisualizer:
    """Unified visualization class for all experiment plots."""
    
    def __init__(self, save_dir=None):
        """Initialize with save directory."""
        self.save_dir = Path(save_dir) if save_dir else Path(PATHS['plots'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_cherenkov_linearity(self, beam, chere, results=None, show=True, save=True):
        """
        Plot Cherenkov linearity analysis.
        
        Args:
            beam: Beam intensity array
            chere: Cherenkov response array
            results: Analysis results dictionary (optional)
            show: Whether to display plot
            save: Whether to save plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Cherenkov response vs intensity (log-log)
        ax1.scatter(beam, chere, alpha=0.7, s=40, color='blue')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel('Beam Intensity (particles/pulse)', fontsize=11)
        ax1.set_ylabel('Cherenkov Response (photons/particle)', fontsize=11)
        ax1.set_title('Cherenkov: Linear Response', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add fit line if results provided
        if results and 'cherenkov' in results:
            mean_response = results['cherenkov']['constancy']['mean_response']
            x_fit = np.logspace(np.log10(beam.min()), np.log10(beam.max()), 100)
            y_fit = np.ones_like(x_fit) * mean_response
            ax1.plot(x_fit, y_fit, 'r--', linewidth=2,
                    label=f'Mean: {mean_response:.2e}')
            ax1.legend()
        
        # Plot 2: Residuals
        if results and 'cherenkov' in results:
            mean_response = results['cherenkov']['constancy']['mean_response']
            predicted = np.ones_like(beam) * mean_response
            residuals = (chere - predicted) / predicted * 100
            ax2.scatter(beam, residuals, alpha=0.6, s=30, color='blue')
            ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax2.set_xscale('log')
            ax2.set_xlabel('Beam Intensity', fontsize=11)
            ax2.set_ylabel('Deviation from Linearity (%)', fontsize=11)
            ax2.set_title('Cherenkov Linearity Residuals', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim([-50, 50])
        
        plt.tight_layout()
        
        if save:
            path = self.save_dir / 'cherenkov_linearity.png'
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved: {path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def plot_scintillator_saturation(self, beam, scint, results=None, show=True, save=True):
        """
        Plot scintillator saturation analysis.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.scatter(beam, scint, alpha=0.7, s=40, color='red')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Beam Intensity (particles/pulse)', fontsize=12)
        ax.set_ylabel('Scintillator Response (photons/particle)', fontsize=12)
        ax.set_title('Scintillator Detector: Saturation Effect', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add Birks' law fit if available
        if results and 'scintillator' in results:
            x_fit = np.logspace(np.log10(beam.min()), np.log10(beam.max()), 100)
            L0 = results['scintillator']['birks_fit']['L0']
            k = results['scintillator']['birks_fit']['k']
            y_fit = L0 / (1 + k * x_fit)
            ax.plot(x_fit, y_fit, 'b--', linewidth=2,
                    label=f"Birks' law: L₀={L0:.2e}, k={k:.2e}")
            ax.legend()
        
        plt.tight_layout()
        
        if save:
            path = self.save_dir / 'scintillator_saturation.png'
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved: {path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def plot_detector_comparison(self, beam, chere, scint, results=None, show=True, save=True):
        """
        Plot detector comparison (ratio and power law fit).
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ratio = chere / scint
        ax.scatter(beam, ratio, alpha=0.7, s=40, color='green')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Beam Intensity (particles/pulse)', fontsize=12)
        ax.set_ylabel('Cherenkov / Scintillator Ratio', fontsize=12)
        ax.set_title('Detector Comparison: Cherenkov Relatively Better at High Intensity', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add power law fit if available
        if results and 'comparison' in results:
            slope = results['comparison']['power_law_fit']['slope']
            intercept = results['comparison']['power_law_fit']['intercept']
            x_fit = np.logspace(np.log10(beam.min()), np.log10(beam.max()), 100)
            ax.plot(x_fit, 10**(intercept + slope * np.log10(x_fit)), 
                    'purple', linewidth=2, linestyle='--',
                    label=f'Power law: ∝ I^{slope:.3f}')
            ax.legend()
        
        plt.tight_layout()
        
        if save:
            path = self.save_dir / 'detector_comparison.png'
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved: {path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def plot_final_summary(self, beam, chere, scint, results=None, uncertainty=None, 
                          show=True, save=True):
        """
        Create final 4-panel summary plot for publication/proposal.
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Cherenkov linearity
        ax1.scatter(beam, chere, alpha=0.7, s=40, color='blue')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel('Beam Intensity (particles/pulse)', fontsize=11)
        ax1.set_ylabel('Cherenkov Response\n(photons/particle)', fontsize=11)
        ax1.set_title('Cherenkov: Linear Response', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        if results and 'cherenkov' in results:
            mean_response = results['cherenkov']['constancy']['mean_response']
            x_fit = np.logspace(np.log10(beam.min()), np.log10(beam.max()), 100)
            y_fit = np.ones_like(x_fit) * mean_response
            ax1.plot(x_fit, y_fit, 'r--', linewidth=2,
                    label=f'Mean: {mean_response:.2e}')
            ax1.legend()
        
        # Plot 2: Scintillator saturation
        ax2.scatter(beam, scint, alpha=0.7, s=40, color='red')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel('Beam Intensity (particles/pulse)', fontsize=11)
        ax2.set_ylabel('Scintillator Response\n(photons/particle)', fontsize=11)
        ax2.set_title('Scintillator: Saturation Effect', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        if results and 'scintillator' in results:
            L0 = results['scintillator']['birks_fit']['L0']
            k = results['scintillator']['birks_fit']['k']
            x_fit = np.logspace(np.log10(beam.min()), np.log10(beam.max()), 100)
            ax2.plot(x_fit, L0 / (1 + k * x_fit), 'b--', linewidth=2,
                    label=f"Birks' law (k={k:.2e})")
            ax2.legend()
        
        # Plot 3: Detector comparison ratio
        ratio = chere / scint
        ax3.scatter(beam, ratio, alpha=0.7, s=40, color='green')
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.set_xlabel('Beam Intensity (particles/pulse)', fontsize=11)
        ax3.set_ylabel('Cherenkov / Scintillator Ratio', fontsize=11)
        ax3.set_title('Relative Performance', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        if results and 'comparison' in results:
            slope = results['comparison']['power_law_fit']['slope']
            intercept = results['comparison']['power_law_fit']['intercept']
            x_fit = np.logspace(np.log10(beam.min()), np.log10(beam.max()), 100)
            ax3.plot(x_fit, 10**(intercept + slope * np.log10(x_fit)), 
                    'purple', linewidth=2, linestyle='--',
                    label=f'∝ I^{slope:.3f}')
            ax3.legend()
        
        # Plot 4: Key metrics text
        ax4.axis('off')
        metrics_text = self._generate_metrics_text(results, uncertainty)
        ax4.text(0.1, 0.5, metrics_text, fontfamily='monospace', 
                verticalalignment='center', fontsize=10, linespacing=1.5)
        
        plt.suptitle('Cherenkov FLASH Experiment: Key Results', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save:
            path = self.save_dir / 'final_summary.png'
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f"Saved: {path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def plot_distributions(self, chere, scint, results=None, show=True, save=True):
        """
        Plot response distributions.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Cherenkov histogram
        ax1.hist(chere, bins=min(30, len(chere)//2), alpha=0.7, color='blue')
        mean_cher = np.mean(chere)
        ax1.axvline(mean_cher, color='red', linestyle='--', 
                   label=f'Mean: {mean_cher:.2e}')
        ax1.set_xlabel('Cherenkov Response (photons/particle)', fontsize=11)
        ax1.set_ylabel('Counts', fontsize=11)
        ax1.set_title('Cherenkov Response Distribution', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Scintillator histogram
        ax2.hist(scint, bins=min(30, len(scint)//2), alpha=0.7, color='red')
        mean_scint = np.mean(scint)
        ax2.axvline(mean_scint, color='blue', linestyle='--', 
                   label=f'Mean: {mean_scint:.2e}')
        ax2.set_xlabel('Scintillator Response (photons/particle)', fontsize=11)
        ax2.set_ylabel('Counts', fontsize=11)
        ax2.set_title('Scintillator Response Distribution', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        if save:
            path = self.save_dir / 'distributions.png'
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved: {path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def plot_uncertainty_breakdown(self, uncertainty_results, show=True, save=True):
        """
        Plot uncertainty breakdown.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        if not uncertainty_results:
            return fig
        
        # Cherenkov uncertainty breakdown
        labels = ['Statistical', 'Systematic', 'Total']
        cher_values = [
            uncertainty_results['cherenkov']['statistical_relative'],
            uncertainty_results['cherenkov']['systematic_relative'],
            uncertainty_results['cherenkov']['total_relative']
        ]
        
        bars1 = ax1.bar(labels, cher_values, color=['blue', 'red', 'green'])
        ax1.set_ylabel('Relative Uncertainty (%)')
        ax1.set_title('Cherenkov Detector Uncertainty Breakdown')
        ax1.grid(True, alpha=0.3, axis='y')
        
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        # Scintillator uncertainty breakdown
        scint_values = [
            uncertainty_results['scintillator']['statistical_relative'],
            uncertainty_results['scintillator']['systematic_relative'],
            uncertainty_results['scintillator']['total_relative']
        ]
        
        bars2 = ax2.bar(labels, scint_values, color=['blue', 'red', 'green'])
        ax2.set_ylabel('Relative Uncertainty (%)')
        ax2.set_title('Scintillator Detector Uncertainty Breakdown')
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save:
            path = self.save_dir / 'uncertainty_breakdown.png'
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved: {path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def _generate_metrics_text(self, results, uncertainty):
        """Generate metrics text for summary plot."""
        text = "KEY EXPERIMENTAL METRICS\n"
        text += "="*40 + "\n\n"
        
        if results and 'cherenkov' in results:
            cher = results['cherenkov']
            text += "CHERENKOV:\n"
            text += f"• Variation: {cher['constancy']['relative_variation']:.1f}%\n"
            text += f"• Response: {cher['constancy']['mean_response']:.2e}\n\n"
        
        if results and 'scintillator' in results:
            scint = results['scintillator']
            text += "SCINTILLATOR:\n"
            text += f"• Birks' k: {scint['birks_fit']['k']:.2e}\n"
            if 'saturation_factors' in scint and 'flash' in scint['saturation_factors']:
                text += f"• Signal loss: {scint['saturation_factors']['flash']['signal_loss']:.1f}%\n\n"
            elif 'saturation_factors' in scint and 'high' in scint['saturation_factors']:
                text += f"• Signal loss: {scint['saturation_factors']['high']['signal_loss']:.1f}%\n\n"
        
        if uncertainty:
            text += "UNCERTAINTIES:\n"
            text += f"• Cherenkov: {uncertainty['cherenkov']['total_relative']:.1f}%\n"
            text += f"• Scintillator: {uncertainty['scintillator']['total_relative']:.1f}%\n\n"
        
        text += "CONCLUSION:\n"
        text += "Cherenkov detectors maintain linearity at\n"
        text += "FLASH intensities while scintillators saturate.\n"
        text += "Recommended for accurate dosimetry in FLASH RT."
        
        return text


# Convenience functions for backward compatibility
def plot_cherenkov_linearity(*args, **kwargs):
    viz = ExperimentVisualizer()
    return viz.plot_cherenkov_linearity(*args, **kwargs)

def plot_scintillator_saturation(*args, **kwargs):
    viz = ExperimentVisualizer()
    return viz.plot_scintillator_saturation(*args, **kwargs)

def plot_detector_comparison(*args, **kwargs):
    viz = ExperimentVisualizer()
    return viz.plot_detector_comparison(*args, **kwargs)

def plot_final_summary(*args, **kwargs):
    viz = ExperimentVisualizer()
    return viz.plot_final_summary(*args, **kwargs)