import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

@xr.register_dataset_accessor('pt')
@xr.register_dataarray_accessor('pt')
class PlotTools():
    def __init__(self,xr_obj):
        self._obj=xr_obj
        
        self._pyhyper_type = 'reduced'
        try:
            self._chi_min = np.min(xr_obj.chi)
            self._chi_max = np.max(xr_obj.chi)
            self._chi_range = [self._chi_min,self._chi_max]
        except AttributeError:
            self._pyhyper_type = 'raw'
    
    def plot_loglog(self):
        '''
        
        '''
        return self._obj.plot(xscale='log',yscale='log')

    def plot_ISI(self, pol, chi_width, q_slice, e_slice, sample_name):
        """
        Plot the integrated scattered intensity:
        
        Inputs: 
            pol (int): X-ray polarization to determine para/perp chi regions
            chi_width (int): width of chi wedge for para/perp slices
            q_slice (slice): q range entered as slice object
            e_slice (slice): energy range entered as slice object 
            sample_name (str): sample name to be included in plot title  

        Returns:
            fig: matplotlib figure object of the ISI plot
            ax: matplotlib axes object of the ISI plot
        """
        # Slice parallel & perpendicular DataArrays (for polarization = 0)
        if pol == 0:
            para_DA = self._obj.rsoxs.slice_chi(180, chi_width=(chi_width/2))
            perp_DA = self._obj.rsoxs.slice_chi(90, chi_width=(chi_width/2))
        elif pol == 90:
            perp_DA = self._obj.rsoxs.slice_chi(180, chi_width=(chi_width/2))
            para_DA = self._obj.rsoxs.slice_chi(90, chi_width=(chi_width/2))  

        # Integrate along q / chi to get integrated scattering intensities
        para_ISI = para_DA.sel(q=q_slice, energy=e_slice).interpolate_na(dim='q').mean('chi').integrate('q')
        perp_ISI = perp_DA.sel(q=q_slice, energy=e_slice).interpolate_na(dim='q').mean('chi').integrate('q')
        
        # Plot
        fig, ax = plt.subplots(figsize=(6,4), dpi=120)
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        
        para_ISI.sel(energy=e_slice).plot.line(ax=ax, label=r'$\parallel$', yscale='log')
        perp_ISI.sel(energy=e_slice).plot.line(ax=ax, label=r'$\perp$', yscale='log')
        fig.suptitle(f'ISI: {sample_name}', fontsize=14, x=0.55)
        ax.set(title=f'Pol = {pol}°, chi width = {chi_width}°, Q = ({q_slice.start}, {q_slice.stop}) ' + 'Å$^{-1}$', 
            xlabel='X-ray energy [eV]', ylabel='Intensity [arb. units]')
        ax.legend(loc='upper right', fontsize=14)
        ax.grid(axis='x', which='both')
        plt.subplots_adjust(top=0.86, bottom=0.2, left=0.2)

        return fig, ax

        