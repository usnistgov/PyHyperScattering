import warnings
import xarray as xr
import numpy as np
import math
from scipy.ndimage import label
import scipy.optimize
from scipy.optimize import least_squares

try:
    import holoviews as hv
    import hvplot.xarray

    import skimage.draw
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm,Normalize
except (ModuleNotFoundError,ImportError):
    warnings.warn('Could not import package for interactive integration utils.  Install holoviews and scikit-image.',stacklevel=2)
import pandas as pd

import json

class Check:
    '''
    Quick Utility to display a mask next to an image, to sanity check the orientation of e.g. an imported mask
    
    '''
    def checkMask(integrator,img,img_min=1,img_max=10000,img_scaling='log',alpha=1):
        '''
            draw an overlay of the mask and an image

            Args:
                integrator: a PyHyper integrator object
                img: a PyHyper raw image (single frame, please!) to draw
                img_min: min value to display
                img_max: max value to display
                img_scaling: 'lin' or 'log'
        '''
        if len(img.shape) > 2:
                warnings.warn('This tool needs a single frame, not a stack!  .sel down to a single frame before starting!',stacklevel=2)

        fig,ax=plt.subplots(1,1)
        if img_scaling == 'log':
            norm=LogNorm(img_min,img_max)
        else:
            norm=Normalize(img_min,img_max)
        img.plot(norm=norm,ax=ax)
        ax.set_aspect(1)
        ax.imshow(integrator.mask,origin='lower',alpha=alpha)
    def checkCenter(integrator,img,img_min=1,img_max=10000,img_scaling='log'):
        '''
            draw the beamcenter on an image

            Args:
                integrator: a PyHyper integrator object
                img: a PyHyper raw image (single frame, please!) to draw
                img_min: min value to display
                img_max: max value to display
                img_scaling: 'lin' or 'log'
        '''
        if len(img.shape) > 2:
                warnings.warn('This tool needs a single frame, not a stack!  .sel down to a single frame before starting!',stacklevel=2)

        fig,ax=plt.subplots()
        if img_scaling == 'log':
            norm=LogNorm(img_min,img_max)
        else:
            norm=Normalize(img_min,img_max)
        img.plot(norm=norm,ax=ax)
        ax.set_aspect(1)
        beamcenter = plt.Circle((integrator.ni_beamcenter_x, integrator.ni_beamcenter_y), 5, color='lawngreen')
        guide1 = plt.Circle((integrator.ni_beamcenter_x, integrator.ni_beamcenter_y), 50, color='lawngreen',fill=False)
        guide2 = plt.Circle((integrator.ni_beamcenter_x, integrator.ni_beamcenter_y), 150, color='lawngreen',fill=False)
        ax.add_patch(beamcenter)
        ax.add_patch(guide1)
        ax.add_patch(guide2)
    def checkAll(integrator,img,img_min=1,img_max=10000,img_scaling='log',alpha=1,d_inner=50,d_outer=150):
        '''
            draw the beamcenter and overlay mask on an image

            Args:
                integrator: a PyHyper integrator object
                img: a PyHyper raw image (single frame, please!) to draw
                img_min: min value to display
                img_max: max value to display
                img_scaling: 'lin' or 'log'
        '''
        if len(img.shape) > 2:
                warnings.warn('This tool needs a single frame, not a stack!  .sel down to a single frame before starting!',stacklevel=2)

        fig,ax=plt.subplots()
        if img_scaling == 'log':
            norm=LogNorm(img_min,img_max)
        else:
            norm=Normalize(img_min,img_max)
        img.plot(norm=norm,ax=ax)
        ax.set_aspect(1)
        beamcenter = plt.Circle((integrator.ni_beamcenter_x, integrator.ni_beamcenter_y), 5, color='lawngreen')
        guide1 = plt.Circle((integrator.ni_beamcenter_x, integrator.ni_beamcenter_y), d_inner, color='lawngreen',fill=False)
        guide2 = plt.Circle((integrator.ni_beamcenter_x, integrator.ni_beamcenter_y), d_outer, color='lawngreen',fill=False)
        ax.add_patch(beamcenter)
        ax.add_patch(guide1)
        ax.add_patch(guide2)
        ax.imshow(integrator.mask,origin='lower',alpha=alpha)
class DrawMask:
    '''
    Utility class for interactively drawing a mask in a Jupyter notebook.


    Usage: 

        Instantiate a DrawMask object using a PyHyper single image frame.

        Call DrawMask.ui() to generate the user interface

        Call DrawMask.mask to access the underlying mask, or save/load the raw mask data with .save or .load


    '''
    
    def __init__(self,frame):
        '''
        Construct a DrawMask object

        Args:
            frame (xarray): a single data frame with pix_x and pix_y axes


        '''
        if len(frame.shape) > 2:
            warnings.warn('This tool needs a single frame, not a stack!  .sel down to a single frame before starting!',stacklevel=2)
            
        self.frame=frame
        
        self.fig = frame.hvplot(cmap='terrain',clim=(5,5000),logz=True,data_aspect=1)

        self.poly = hv.Polygons([])
        self.path_annotator = hv.annotate.instance()

    def ui(self):
        '''
        Draw the DrawMask UI in a Jupyter notebook.


        Returns: the holoviews object



        '''
        print('Usage: click the "PolyAnnotator" tool at top right.  DOUBLE CLICK to start drawing a masked object, SINGLE CLICK to add a vertex, then DOUBLE CLICK to finish.  Click/drag individual vertex to adjust.')
        return self.path_annotator(
                self.fig * self.poly.opts(
                            width=self.frame.shape[0], 
                            height=self.frame.shape[1], 
                            responsive=False), 
                annotations=['Label'], 
            vertex_annotations=['Value'])


    def save(self,fname):
        '''
        Save a parametric mask description as a json dump file.

        Args:
            fname (str): name of the file to save

        '''
        dflist = []
        for i in range(len(self.path_annotator.annotated)):
            dflist.append(self.path_annotator.annotated.iloc[i].dframe(['x','y']).to_json())
        
        with open(fname, 'w') as outfile:
            json.dump(dflist, outfile)
            
    def load(self,fname):
        '''
        Load a parametric mask description from a json dump file.

        Args:
            fname (str): name of the file to read from

        '''
        with open(fname,'r') as f:
            strlist = json.load(f)
        print(strlist)
        dflist = []
        for item in strlist:
            dflist.append(pd.read_json(item))
        print(dflist)
        self.poly = hv.Polygons(dflist)
        
        self.path_annotator(
                self.fig * self.poly.opts(
                            width=self.frame.shape[1], 
                            height=self.frame.shape[0], 
                            responsive=False), 
                annotations=['Label'], 
            vertex_annotations=['Value'])
        
        
    @property
    def mask(self):
        '''
        Render the mask as a numpy boolean array.
        '''
        mask = np.zeros(self.frame.shape).astype(bool)
        for i in range(len(self.path_annotator.annotated)):
            mask |= skimage.draw.polygon2mask(self.frame.shape,self.path_annotator.annotated.iloc[i].dframe(['x','y']))

        return mask

def automask(image, max_size = 50):
    # Create binary mask of the image by thresholding at 0
    mask = (image <= 0.25)

    # Label the connected regions of the mask
    labels, num_features = label(mask)

    # Mask out regions that are smaller than max_size
    for i in range(1, num_features+1):
        size = np.sum(labels == i)
        if size <= max_size:
            mask[labels == i] = False

    # Convert the masked values to NaN
    image = image.astype(float)
    image[mask] = np.nan

    return image

def remove_zingers(data_array, threshold1 = 10, threshold2 = 10):        
    # Compute the mean intensity value across the chi axis for each q
    mean_intensity = np.nanmean(data_array, axis=1)

    # Compute the standard deviation of the intensity values at each q
    std_intensity = np.nanstd(data_array, axis=1)

    # Compute the z-score for each intensity value at each q
    z_score = (data_array - mean_intensity[:, np.newaxis]) / std_intensity[:, np.newaxis]

    # Identify outliers by thresholding the z-score array
    outliers = np.abs(z_score) > threshold1

    # Iterate over each q coordinate with outliers
    for i, q in enumerate(outliers):
        if np.any(q):
            # Compute the mean intensity value across all chi for this q
            mean_intensity_q = np.nanmean(data_array[i][~q])

            # Compute the standard deviation of the intensity values at this q
            std_intensity_q = np.nanstd(data_array[i][~q])

            # Compute the z-score for each intensity value at this q
            z_score_q = (data_array[i] - mean_intensity_q) / std_intensity_q

            # Identify outliers by thresholding the z-score array for this q
            outliers_q = np.abs(z_score_q) > threshold2

            # Mask the outliers with NaN values
            data_array[i][outliers_q] = np.nan

    return data_array

def remove_XRF_background(sample, q_lower1, q_upper1, q_lower2, q_upper2, pre, post, printE=None, make_plots=False):
    """
    This function fits an exponential model to the XRF background, calculates the XRF background for each energy 
    in the sample, and subtracts the background from the intensity data.
    
    Parameters:
    sample (xarray): The data sample containing intensity information.
    q_lower1, q_upper1 (float): The lower and upper bounds for the first q-range for XRF background calculation.
    q_lower2, q_upper2 (float): The lower and upper bounds for the second q-range for XRF background calculation.
    pre, post (float): The pre-edge and post-edge energies for XRF background calculation.
    printE (float, optional): Energy at which to print the fit. If not provided, defaults to the midpoint between pre and post.
    make_plots (bool, optional): If True, produces diagnostic plots. Defaults to False.
    
    Returns:
    list: Two xarray DataArray objects - the XRF-subtracted data, and the calculated XRF vs energy for each energy.
    """    
    # Exponential model function
    def exp_func(q, para):
        A, B, C = para
        return A * q**B + C

    # Error function for the exponential model
    def err(para, q, y):
        return exp_func(q, para) - y

    # Error function for the global fit
    def err_global(para, q1, q2, y1, y2):
        p1 = para[0], para[2], para[3]
        p2 = para[1], para[2], para[4]
        err1 = err(p1, q1, y1)
        err2 = err(p2, q2, y2)
        return np.concatenate((err1, err2))

    # Extract pre-edge and post-edge intensities and q values
    pre_int = sample.sel(energy=pre, method='nearest').sel(q=slice(q_lower1, q_upper1)).mean('chi').values.flatten()
    pre_q = sample.sel(energy=pre, method='nearest').sel(q=slice(q_lower1, q_upper1)).mean('chi')['q']
    post_int = sample.sel(energy=post, method='nearest').sel(q=slice(q_lower1, q_upper1)).mean('chi').values.flatten()
    post_q = sample.sel(energy=post, method='nearest').sel(q=slice(q_lower1, q_upper1)).mean('chi')['q']

    if make_plots:
        plt.plot(post_q, post_int, label='Post Edge Intensity')
        plt.plot(pre_q, pre_int, label='Pre Edge Intensity')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title('Pre and Post Edge Intensity')
        plt.xlabel('$\it{q}$ (Å$^{-1}$)')
        plt.ylabel('Intensity (a.u.)')
        plt.show()

    # Initial guess for the least squares fit
    init_guess = [0.05, 0.05, -1, 0, 20]

    # Perform the least squares fit
    result = least_squares(err_global, init_guess, bounds=([0, 0, -np.inf, 0, 0], [np.inf, np.inf, 0, np.inf, np.inf]), args=(pre_q, post_q, pre_int, post_int))
    para_best = result.x
    para_best1 = [para_best[0], para_best[2], para_best[3]]
    para_best2 = [para_best[1], para_best[2], para_best[4]]

    xrf_power_law_fit = para_best[2]
    
    if make_plots:
        print(f"XRF Power Law Fit: {xrf_power_law_fit}")

    if make_plots:
        sample.sel(energy=pre, method='nearest').mean('chi').plot(yscale='log', xscale='log', label='Pre-edge')
        sample.sel(energy=post, method='nearest').mean('chi').plot(yscale='log', xscale='log', label='Post-edge')
        plt.plot(pre_q, exp_func(pre_q, para_best1), label='Pre-edge fit', color='blue', linewidth=3)
        plt.plot(post_q, exp_func(post_q, para_best2), label='Post-edge fit', color='orange', linewidth=3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xlabel('$\it{q}$ (Å$^{-1}$)')
        plt.ylabel('Intensity (a.u.)')
        plt.show()

    # Exponential model function with fixed B parameter
    def exp_func2(q, para):
        A, C = para
        return A * q**xrf_power_law_fit + C

    # Error function for the second exponential model
    def err2(para, q, y):
        return exp_func2(q, para) - y

    # Initialize a copy of the xarray for the XRF background-removed data
    xrf_subtracted_data = sample.copy()
    xrf_fit_C = []  # List to store the fitted C values

    init_guess = [0.05, 20]  # Initial guess for the first iteration
    
    if printE is None:
        printE = (pre + post) / 2  # Default printE to the midpoint between pre and post

    # Iterate over each energy in the sample
    for v in sample.energy:
        intensity = sample.sel(energy=v, q=slice(q_lower2, q_upper2)).mean('chi').values.flatten()
        q_values = sample.sel(energy=v, q=slice(q_lower2, q_upper2)).mean('chi')['q']

        # Perform the least squares fit for the current energy
        result = least_squares(err2, init_guess, bounds=([0, 0], [np.inf, np.inf]), args=(q_values, intensity))
        para_best = result.x
        xrf_fit_C.append(para_best[1])
        
        # Subtract the constant offset for the current energy from all q values
        xrf_subtracted_data.loc[{'energy': v}] -= para_best[1]

        # Update the initial guess for the next iteration
        init_guess = para_best  

        if v == sample.sel(energy=printE, method='nearest').energy and make_plots:
            sample.sel(energy=printE, method='nearest').mean('chi').plot(yscale='log', xscale='log', label=str(np.asarray(v)) + ' eV Data')
            plt.plot(q_values, exp_func2(q_values, para_best), label=str(np.asarray(v)) + ' eV Fit', color='blue', linewidth=3)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xlabel('$\it{q}$ (Å$^{-1}$)')
            plt.ylabel('Intensity (a.u.)')
            plt.show()

    if make_plots:
        plt.plot(sample.energy, xrf_fit_C, marker='.')
        plt.xlabel('Energy (eV)')
        plt.ylabel('XRF Offset Value (a.u.)')
        plt.show()

    # Construct the xrf_fit xarray, which only depends on the energy dimension
    xrf_fit_C = xr.DataArray(xrf_fit_C, coords=[sample.energy], dims=['energy'])
    
    return xrf_subtracted_data, xrf_fit_C