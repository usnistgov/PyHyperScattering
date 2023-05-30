import warnings
import xarray as xr
import numpy as np
import math
from scipy.ndimage import label
import scipy.optimize

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

def remove_XRF_background(xarray, q_lower1, q_upper1, q_lower2, q_upper2, pre, post):
    """
    This function computes the X-ray Fluorescence (XRF) background from a xarray and subtracts it from the xarray's 
    intensity data. The XRF background is calculated for two different q-range regions, then subtracted from the 
    entire q-range. The function returns the xarray data with the XRF background subtracted, as well as the constant 
    offset xrf_fit. Both are returned as xarray objects.
    
    Parameters:
    - xarray: xarray DataArray containing the xarray data, with 'energy', 'q', 'chi', 'polarization', and 'xarray_name' 
              dimensions.
    - q_lower1, q_upper1: The lower and upper bounds for the first q-range to consider when calculating the XRF background.
    - q_lower2, q_upper2: The lower and upper bounds for the second q-range to consider when calculating the XRF background.
    - pre, post: Two specific energy values at which to compute the XRF background.

    Returns:
    - xrf_subtracted_data: The xarray data with the XRF background subtracted, as an xarray DataArray object.
    - xrf_fit: The constant offset used in the XRF background subtraction, as an xarray DataArray object.
    """
    
    # Define the functions to fit high q scattering with background fluorescence
    def exp_func(q, para):
        A, B, C = para
        return A * q**B + C

    def err(para,q,y):
        return abs(exp_func(q, para)-y)

    def err_global(para, q1, q2, y1, y2):
        p1 = para[0], para[2], 0
        p2 = para[1], para[2], para[3]
        err1 = err(p1,q1,y1)
        err2 = err(p2,q2,y2)
        return np.concatenate((err1,err2))

    # Extract [q,I] and fit to determine power law scaling
    pre_int = xarray.sel(energy=pre, q=slice(q_lower1,q_upper1)).mean('chi').values.flatten()
    pre_q = xarray.sel(energy=pre, q=slice(q_lower1,q_upper1)).mean('chi')['q']
    post_int = xarray.sel(energy=post, q=slice(q_lower1,q_upper1)).mean('chi').values.flatten()
    post_q = xarray.sel(energy=post, q=slice(q_lower1,q_upper1)).mean('chi')['q']

    init_guess = [1, 1, -4, 0]
    para_best, _ = scipy.optimize.leastsq(err_global, init_guess, args=(pre_q, post_q, pre_int, post_int))

    xrf_power_law_fit = para_best[2]

    # Redefine function with power law scaling set by previous optimization
    def exp_func2(q, para):
        A, C = para
        return A * q**xrf_power_law_fit + C

    def err2(para, q, y):
        return abs(exp_func2(q, para)-y)

    # Initialize a copy of the xarray xarray for the XRF background-removed data
    xrf_subtracted_data = xarray.copy()

    # Fit all energies
    xrf_fit = []
    for energy in xarray.energy:
        intensity = xarray.sel(energy=energy, q=slice(q_lower2,q_upper2)).mean('chi').values
        intensity = intensity[:,0,0]
        q_values = xarray.sel(energy=energy, q=slice(q_lower2,q_upper2)).mean('chi')['q']

        init_guess = [1, 80]
        para_best, _ = scipy.optimize.leastsq(err2, init_guess, args=(q_values, intensity))

        # Subtract the constant offset for the current energy from all q values
        xrf_subtracted_data.loc[{'energy': energy}] -= para_best[1]
        
        xrf_fit.append(para_best[1])

    # Construct the xrf_fit xarray, which only depends on the energy dimension
    xrf_fit = xr.DataArray(xrf_fit, coords=[xarray.energy], dims=['energy'])
    
    # xrf_fit = xrf_fit.where(xrf_fit > 0, 0)  # set negative values to 0
    # xrf_fit = xrf_fit.where(xarray.energy > 285, 0)  # set energy values less than 285 to 0

    return xrf_subtracted_data, xrf_fit