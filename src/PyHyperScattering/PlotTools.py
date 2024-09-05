import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pathlib

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

    def plot_ISI(self, 
                 pol=None, 
                 chi_width=None, 
                 q_slice=None, 
                 e_slice=None, 
                 sample_name=None,
                 save=True):
        """
        Plot the integrated scattered intensity:
        
        Inputs: (default will be to pull from 'plot_ROIs' attribute)
            pol (int): X-ray polarization to determine para/perp chi regions
            chi_width (int): width of chi wedge for para/perp slices
            q_slice (slice): q range entered as slice object
            e_slice (slice): energy range entered as slice object 
            sample_name (str): sample name to be included in plot title  
            save (bool, default True): save figure to new folder in notebook directory

        Returns:
            fig: matplotlib figure object of the ISI plot
            ax: matplotlib axes object of the ISI plot
        """

        # Load default plot ROI values from 'plot_ROIs' attribute / dictionary
        # Can be overwritten in the function call
        if pol is None:
            pol = int(self._obj.polarization)
        if chi_width is None:
            chi_width = self._obj.plot_ROIs['chi_width']
        if q_slice is None:
            q_tup = self._obj.plot_ROIs['q_range']
            q_slice = slice(q_tup[0], q_tup[1])
        if e_slice is None:
            e_tup = self._obj.plot_ROIs['energy_range']
            e_slice = slice(e_tup[0], e_tup[1])
        if sample_name is None:
            sample_name = str(self._obj.sample_name.values)

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

        # Save plot if true (saves to notebook working directory)
        if save:
            filename = f'{sample_name}_chi-{chi_width}_q-{q_slice.start}-{q_slice.stop}_energy-{e_slice.start}-{e_slice.stop}_.png'
            savePath = pathlib.Path.cwd().joinpath('ISI_plots')
            savePath.mkdir(exist_ok=True)
            fig.savefig(savePath.joinpath(filename))

        return fig, ax

        