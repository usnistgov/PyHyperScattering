import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import LogNorm
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
                 save=True,
                 filename=None,
                 savePath=None):
        """
        Plot the integrated scattered intensity:
        
        Inputs: (default will be to pull from 'plot_roi' attribute)
            pol (int): X-ray polarization to determine para/perp chi regions
            chi_width (int): width of chi wedge for para/perp slices
            q_slice (slice): q range entered as slice object
            e_slice (slice): energy range entered as slice object 
            sample_name (str): sample name to be included in plot title  
            save (bool, default True): save figure to new folder in notebook directory
            savePath (pathlib.Path): pathlib directory for where to save plots (will create directory)
                defaults to a new 'ISI_plots' folder in notebook working directory
            filename (str): filename to name saved figure
                defaults to f'{sample_name}_chi-{chi_width}_q-{q_slice.start}-{q_slice.stop}_energy-{e_slice.start}-{e_slice.stop}_.png'

            Example 'plot_roi' format:
            plot_roi = {'chi_width': 90,
                        'q_range': (0.01, 0.09),
                        'energy_range': (280, 295),
                        'energy_default': 285}

        Returns:
            fig: matplotlib figure object of the ISI plot
            ax: matplotlib axes object of the ISI plot
        """

        # Load default plot ROI values from 'plot_ROIs' attribute / dictionary
        # Can be overwritten in the function call
        if pol is None:
            pol = int(self._obj.polarization)
        if chi_width is None:
            chi_width = self._obj.plot_roi['chi_width']
        if q_slice is None:
            q_tup = self._obj.plot_roi['q_range']
            q_slice = slice(q_tup[0], q_tup[1])
        if e_slice is None:
            e_tup = self._obj.plot_roi['energy_range']
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
            if filename is None:
                filename = f'{sample_name}_chi-{chi_width}_q-{q_slice.start}-{q_slice.stop}_energy-{e_slice.start}-{e_slice.stop}_.png'
            if savePath is None:
                savePath = pathlib.Path.cwd().joinpath('ISI_plots')
            savePath.mkdir(exist_ok=True)
            fig.savefig(savePath.joinpath(filename))

        return fig, ax

    def plot_Imap(self,
                  pol=None,
                  chi_width=None,
                  q_slice=None,
                  e_slice=None,
                  I_cmap=None,
                  xscale=None,
                  sample_name=None,
                  save=True,
                  filename=None,
                  savePath=None):
        """
        Plot an intensity heatmap (2D). Q along x, energy along y, intensity colormap

        Inputs:
            pol (int): X-ray polarization to determine para/perp chi regions
            chi_width (int): width of chi wedge for para/perp slices
            q_slice (slice): q range entered as slice object
            e_slice (slice): energy range entered as slice object 
            I_cmap (str or plt.cm): matplotlib colormap for intensity, default is 'turbo'
            xscale (str): 'log' (default) or 'linear'
            sample_name (str): sample name to be included in plot title  
            save (bool, default True): save figure to new folder in notebook directory
            savePath (pathlib.Path): pathlib directory for where to save plots (will create directory)
                defaults to a new 'Imap_plots' folder in notebook working directory
            filename (str): filename to name saved figure
                defaults to f'{sample_name}_chi-{chi_width}_q-{q_slice.start}-{q_slice.stop}_energy-{e_slice.start}-{e_slice.stop}_.png'

            Example 'plot_roi' format:
            plot_roi = {'chi_width': 90,
                        'q_range': (0.01, 0.09),
                        'energy_range': (280, 295),
                        'energy_default': 285}

            Example 'plot_hints' format:
            plot_hints = {'I_cmap': 'turbo',
                          'xscale': 'log'}

        Returns:
            fig: matplotlib figure object of the intensity map plots
            ax: list of the 2 matplotlib axes object of the intensity map plots
        """

        # Load default plot roi values from 'plot_roi' attribute / dict
        # Load default plot hint values from 'plot_hints' attribute / dict
        # Can be overwritten in the function call
        if pol is None:
            pol = int(self._obj.polarization)
        if chi_width is None:
            chi_width = self._obj.plot_roi['chi_width']
        if q_slice is None:
            q_tup = self._obj.plot_roi['q_range']
            q_slice = slice(q_tup[0], q_tup[1])
        if e_slice is None:
            e_tup = self._obj.plot_roi['energy_range']
            e_slice = slice(e_tup[0], e_tup[1])
        if I_cmap is None:
            I_cmap = self._obj.plot_hints['I_cmap']
        if xscale is None:
            xscale = self._obj.plot_hints['xscale']
        if sample_name is None:
            sample_name = str(self._obj.sample_name.values)

        # Slice parallel & perpendicular DataArrays (for polarization = 0)
        if pol == 0:
            para_DA = self._obj.rsoxs.slice_chi(180, chi_width=(chi_width/2))
            perp_DA = self._obj.rsoxs.slice_chi(90, chi_width=(chi_width/2))
        elif pol == 90:
            perp_DA = self._obj.rsoxs.slice_chi(180, chi_width=(chi_width/2))
            para_DA = self._obj.rsoxs.slice_chi(90, chi_width=(chi_width/2))  

        # Plotting
        fig, axs = plt.subplots(1, 2, figsize=(8,4))

        # Downselect data to selected region
        para_slice = para_DA.mean('chi').sel(q=q_slice, energy=e_slice)
        perp_slice = perp_DA.mean('chi').sel(q=q_slice, energy=e_slice)

        # Get colorlimits
        cmin = float(perp_slice.quantile(0.01))
        cmax = float(para_slice.quantile(0.995))

        # Generate plot
        para_slice.plot(ax=axs[0], xscale=xscale, cmap=I_cmap, norm=LogNorm(cmin, cmax), add_colorbar=False)
        perp_slice.plot(ax=axs[1], xscale=xscale, cmap=I_cmap, norm=LogNorm(cmin, cmax), add_colorbar=False)

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=I_cmap, norm=LogNorm(cmin, cmax)) 
        cax = axs[1].inset_axes([1.03, 0, 0.05, 1])
        cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
        cbar.set_label(label='Intensity [arb. units]', labelpad=12, rotation=270)

        # Set title & labels
        fig.suptitle(f'Intensity Maps: {sample_name}, Polarization = {pol}°, Chi Width = {chi_width}°')
        fig.set(tight_layout=True)
        axs[0].set(title='Parallel to $E_p$', ylabel='Photon energy [eV]', xlabel='Q [$Å^{-1}$]')
        axs[1].set(title='Perpendicular to $E_p$ ', ylabel=None, xlabel='Q [$Å^{-1}$]')
        
        # Save plot if true (saves to notebook working directory)
        if save:
            if filename is None:
                filename = f'{sample_name}_chi-{chi_width}_q-{q_slice.start}-{q_slice.stop}_energy-{e_slice.start}-{e_slice.stop}_.png'
            if savePath is None:
                savePath = pathlib.Path.cwd().joinpath('Imap_plots')
            savePath.mkdir(exist_ok=True)
            fig.savefig(savePath.joinpath(filename))

        return fig, axs

    def plot_IvQ(self,
                 pol=None,
                 chi_width=None,
                 q_slice=None,
                 selected_energies=None,
                 I_cmap=None,
                 xscale=None,
                 yscale=None,
                 sample_name=None,
                 save=True, 
                 filename=None,
                 savePath=None):
        """
        Plot intensity vs Q for parallel and perpendicular cuts at selected energies

        Inputs:
            pol (int): X-ray polarization to determine para/perp chi regions
            chi_width (int): width of chi wedge for para/perp slices
            q_slice (slice): q range entered as slice object
            e_slice (slice): energy range entered as slice object 
            I_cmap (str or plt.cm): matplotlib colormap for intensity, default is 'turbo'
            xscale (str): 'log' (default) or 'linear'
            sample_name (str): sample name to be included in plot title  
            save (bool, default True): save figure to new folder in notebook directory
            savePath (pathlib.Path): pathlib directory for where to save plots (will create directory)
                defaults to a new 'IvQ_plots' folder in notebook working directory
            filename (str): filename to name saved figure
                defaults to f'{sample_name}_chi-{chi_width}_q-{q_slice.start}-{q_slice.stop}.png'

            Example 'plot_roi' format:
            plot_roi = {'chi_width': 90,
                        'q_range': (0.01, 0.09),
                        'energy_range': (280, 295),
                        'energy_default': 285,
                        'selected_energies': np.array(
                        [275, 284, 284.4, 284.8, 285.2, 285.6, 286.2, 287, 300, 335])}

            Example 'plot_hints' format:
            plot_hints = {'I_cmap': 'turbo',
                          'xscale': 'log',
                          'yscale': 'log}

        Returns:
            fig: matplotlib figure object of the intensity vs Q linecuts
            ax: list of the 2 matplotlib axes object of the intensity vs Q plots
        """

        # Load default plot roi values from 'plot_roi' attribute / dict
        # Load default plot hint values from 'plot_hints' attribute / dict
        # Can be overwritten in the function call
        if pol is None:
            pol = int(self._obj.polarization)
        if chi_width is None:
            chi_width = self._obj.plot_roi['chi_width']
        if q_slice is None:
            q_tup = self._obj.plot_roi['q_range']
            q_slice = slice(q_tup[0], q_tup[1])
        if selected_energies is None:
            selected_energies = self._obj.plot_roi['selected_energies']
        if I_cmap is None:
            I_cmap = self._obj.plot_hints['I_cmap']
            if isinstance(I_cmap, str):  # convert to matplotlib ListedColormap object
                I_cmap = getattr(plt.cm, I_cmap)
        if xscale is None:
            xscale = self._obj.plot_hints['xscale']
        if yscale is None:
            yscale = self._obj.plot_hints['yscale']
        if sample_name is None:
            sample_name = str(self._obj.sample_name.values)

        # Slice parallel & perpendicular DataArrays (for polarization = 0)
        if pol == 0:
            para_DA = self._obj.rsoxs.slice_chi(180, chi_width=(chi_width/2))
            perp_DA = self._obj.rsoxs.slice_chi(90, chi_width=(chi_width/2))
        elif pol == 90:
            perp_DA = self._obj.rsoxs.slice_chi(180, chi_width=(chi_width/2))
            para_DA = self._obj.rsoxs.slice_chi(90, chi_width=(chi_width/2))  

        # Downselect data to selected region
        para_slice = para_DA.mean('chi').sel(q=q_slice).sel(energy=selected_energies, method='nearest')
        perp_slice = perp_DA.mean('chi').sel(q=q_slice).sel(energy=selected_energies, method='nearest')

        # Plot
        fig, axs = plt.subplots(ncols=2, figsize=(8,4), tight_layout=True)

        colors = I_cmap(np.linspace(0,1,len(selected_energies)))
        for i, energy in enumerate(para_slice.energy.values):
            para_slice.sel(energy=energy).plot.line(ax=axs[0], color=colors[i], yscale=yscale, xscale=xscale, label=energy)
            perp_slice.sel(energy=energy).plot.line(ax=axs[1], color=colors[i], yscale=yscale, xscale=xscale, label=energy)

        fig.suptitle(f'Intensity vs Q, {pol}° pol, 90° chi width: {sample_name}', x=0.47)

        axs[0].set(title=f'Parallel to E$_p$', ylabel='Intensity [arb. units]', xlabel='Q [$Å^{-1}$]')
        axs[1].set(title=f'Perpendicular to E$_p$', ylabel='Intensity [arb. units]', xlabel='Q [$Å^{-1}$]')
        axs[1].legend(title='Energy [eV]', loc=(1.05,0.05))
        for ax in axs:
            ax.grid(visible=True, axis='x', which='both')

        # Save plot if true (saves to notebook working directory)
        if save:
            if filename is None:
                filename = f'{sample_name}_chi-{chi_width}_q-{q_slice.start}-{q_slice.stop}.png'
            if savePath is None:
                savePath = pathlib.Path.cwd().joinpath('IvQ_plots')
            savePath.mkdir(exist_ok=True)
            fig.savefig(savePath.joinpath(filename))            

        return fig, axs

    def plot_ARmap(self,
                   pol=None,
                   chi_width=None,
                   q_slice=None,
                   e_slice=None,
                   I_cmap=None,
                   xscale=None,
                   ar_vlim=None,
                   sample_name=None,
                   save=True,
                   filename=None,
                   savePath=None):
        """
        Plots anistropy ratio heatmap

        Inputs:
            pol (int): X-ray polarization to determine para/perp chi regions
            chi_width (int): width of chi wedge for para/perp slices
            q_slice (slice): q range entered as slice object
            e_slice (slice): energy range entered as slice object 
            I_cmap (str or plt.cm): matplotlib colormap for intensity, default is 'turbo'
            xscale (str): 'log' (default) or 'linear'
            sample_name (str): sample name to be included in plot title  
            save (bool, default True): save figure to new folder in notebook directory
            savePath (pathlib.Path): pathlib directory for where to save plots (will create directory)
                defaults to a new 'ARmap_plots' folder in notebook working directory
            filename (str): filename to name saved figure
                defaults to f'{sample_name}_chi-{chi_width}_q-{q_slice.start}-{q_slice.stop}_energy-{e_slice.start}-{e_slice.stop}_.png'

            Example 'plot_roi' format:
            plot_roi = {'chi_width': 90,
                        'q_range': (0.01, 0.09),
                        'energy_range': (280, 295),
                        'energy_default': 285}

            Example 'plot_hints' format:
            plot_hints = {'I_cmap': 'turbo',
                          'xscale': 'log'}

        Returns:
            fig: matplotlib figure object of the AR map plots
            ax: list of the 2 matplotlib axes object of the AR map plots
        """
        # Load default plot roi values from 'plot_roi' attribute / dict
        # Load default plot hint values from 'plot_hints' attribute / dict
        # Can be overwritten in the function call
        if pol is None:
            pol = int(self._obj.polarization)
        if chi_width is None:
            chi_width = self._obj.plot_roi['chi_width']
        if q_slice is None:
            q_tup = self._obj.plot_roi['q_range']
            q_slice = slice(q_tup[0], q_tup[1])
        if e_slice is None:
            e_tup = self._obj.plot_roi['energy_range']
            e_slice = slice(e_tup[0], e_tup[1])
        if I_cmap is None:
            I_cmap = self._obj.plot_hints['I_cmap']
        if xscale is None:
            xscale = self._obj.plot_hints['xscale']
        if ar_vlim is None:
            ar_vlim = 1
        if sample_name is None:
            sample_name = str(self._obj.sample_name.values)

        # Extract AR data:
        sel_DA = self._obj.sel(q=q_slice, energy=e_slice)
        ar_DA = sel_DA.rsoxs.AR(chi_width=chi_width/2)

        # Plot
        im = ar_DA.plot.pcolormesh(figsize=(6,4), norm=plt.Normalize(-ar_vlim, ar_vlim))

        im.figure.suptitle('Anisotropy Ratio (AR) Map', fontsize=14, x=0.43)
        im.axes.set(title=f'{sample_name}, Polarization = {pol}°, Chi Width = {chi_width}°', ylabel='Photon Energy [eV]', xlabel='q [$Å^{-1}$]', xscale='log')
        im.colorbar.set_label('AR [arb. units]', rotation=270, labelpad=12)

        # Extract figure and axes to return matplotlib object
        fig = im.figure
        ax = im.axes

        # Save plot if true (saves to notebook working directory)
        if save:
            if filename is None:
                filename = f'{sample_name}_chi-{chi_width}_q-{q_slice.start}-{q_slice.stop}_energy-{e_slice.start}-{e_slice.stop}_.png'
            if savePath is None:
                savePath = pathlib.Path.cwd().joinpath('ARmap_plots')
            savePath.mkdir(exist_ok=True)
            fig.savefig(savePath.joinpath(filename))

        return fig, ax
