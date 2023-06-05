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

def automask(image, max_region_size = 50, threshold_value = 0.25):
    """
    This function generates an automatic mask for an input image. This mask is primarily used to hide regions 
    from the image that have a low-intensity value (less than or equal to a threshold value) and are larger 
    than a defined maximum size. The latter is particularly useful when you want to ignore non-contributing 
    regions in the image, such as the edges of a detector or structural components that are not X-ray sensitive.

    Default values for max_region_size and threshold_value are optimized for the Pilatus 1M and Pilatus 900k 
    detectors at NSLS-II SMI.

    Parameters:
    image (np.array): The input image provided as a NumPy array.
    max_region_size (int, optional): Defines the maximum size a region can be to remain unmasked. If a region 
                                     is larger than this value, it will be masked out. Default is 50.
    threshold_value (float, optional): A critical intensity value used to create the initial binary mask. Any 
                                       pixel with intensity less than or equal to this value will be marked for 
                                       potential masking. Default is 0.25.

    Returns:
    tuple: A tuple containing the masked image (np.array) and the binary mask (np.array). In the masked image, 
           the intensity of masked regions is replaced with NaN. In the binary mask, True values correspond to 
           the masked regions.
    """
    
    # Create a binary mask where each pixel is True if its intensity is less than or equal to the threshold_value
    binary_mask = (image <= threshold_value)

    # Identify and label the connected regions in the binary mask
    labels, num_features = label(binary_mask)

    # Loop over each labeled region
    for i in range(1, num_features+1):
        # Compute the total number of pixels in the current region
        region_size = np.sum(labels == i)
        
        # If the region's size exceeds the max_region_size, confirm its masking by setting the corresponding 
        # binary_mask values to True
        if region_size > max_region_size:
            binary_mask[labels == i] = True

    # Duplicate the original image to prevent unwanted modifications
    masked_image = np.copy(image)

    # Convert the image pixel values to float type to support NaN values
    masked_image = masked_image.astype(float)
    
    # In the masked_image, replace the intensity of all pixels that need to be masked (True in binary_mask) with NaN
    masked_image[binary_mask] = np.nan

    # Return the masked image and the binary mask
    return masked_image, binary_mask

def remove_zingers(data_array, z_score_threshold1 = 10, z_score_threshold2 = 10):        
    """
    This function removes outliers (zingers) from a data array. It operates by 
    identifying values that are extreme outliers in terms of their z-scores 
    (the number of standard deviations away from the mean). It returns the data
    with outliers removed, as well as an array of the outlier values themselves.
    
    Parameters:
    data_array (np.array): The input data array.
    z_score_threshold1 (float): The initial z-score threshold for detecting outliers.
    z_score_threshold2 (float): The refined z-score threshold for detecting outliers within suspected regions.
    
    Returns:
    tuple: The dezingered data array (np.array) and the zingers (np.array).
    """
    
    # Create a new array for the zingers, initialized with NaN values
    zingers = np.empty(data_array.shape)
    zingers[:] = np.nan

    # Compute the mean intensity value across the chi axis for each q
    mean_intensity = np.nanmean(data_array, axis=1)

    # Compute the standard deviation of the intensity values at each q
    std_intensity = np.nanstd(data_array, axis=1)

    # Compute the z-score for each intensity value at each q
    z_score = (data_array - mean_intensity[:, np.newaxis]) / std_intensity[:, np.newaxis]

    # Identify potential outliers by thresholding the z-score array with the initial threshold
    potential_outliers = np.abs(z_score) > z_score_threshold1

    # Iterate over each q coordinate with potential outliers
    for i, q in enumerate(potential_outliers):
        if np.any(q):
            # Compute the mean intensity value across all chi for this q, excluding potential outliers
            mean_intensity_q = np.nanmean(data_array[i][~q])

            # Compute the standard deviation of the intensity values at this q, excluding potential outliers
            std_intensity_q = np.nanstd(data_array[i][~q])

            # Compute the z-score for each intensity value at this q
            z_score_q = (data_array[i] - mean_intensity_q) / std_intensity_q

            # Identify confirmed outliers by thresholding the z-score array for this q with the refined threshold
            confirmed_outliers_q = np.abs(z_score_q) > z_score_threshold2

            # Transfer the zingers to the zingers array
            zingers[i][confirmed_outliers_q] = data_array[i][confirmed_outliers_q]

            # Mask the confirmed outliers with NaN values
            data_array[i][confirmed_outliers_q] = np.nan

    return data_array, zingers

def remove_XRF_background(sample, q_lower, q_upper, pre, post, printE=None, make_plots=False):
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
    # Define the exponential model function for initial fit.
    # A, B, C are the parameters of this model, where:
    # A is the amplitude of the exponential,
    # B is the power to which q is raised,
    # C is a constant offset.    
    def exp_func(q, para):
        A, B, C = para
        return A * q**B + C

    # The error function associated with the initial exponential model
    def err(para, q, y):
        return exp_func(q, para) - y

    # Error function for a combined fit over two regions (pre-edge and post-edge)
    def err_global(para, q1, q2, y1, y2):
        p1 = para[0], para[2], para[3]
        p2 = para[1], para[2], para[4]
        err1 = err(p1, q1, y1)
        err2 = err(p2, q2, y2)
        return np.concatenate((err1, err2))

    # Extracting intensity and q values for both pre-edge and post-edge regions
    pre_int = sample.sel(energy=pre, method='nearest').sel(q=slice(q_lower, q_upper)).mean('chi').values.flatten()
    pre_q = sample.sel(energy=pre, method='nearest').sel(q=slice(q_lower, q_upper))['q']
    post_int = sample.sel(energy=post, method='nearest').sel(q=slice(q_lower, q_upper)).mean('chi').values.flatten()
    post_q = sample.sel(energy=post, method='nearest').sel(q=slice(q_lower, q_upper))['q']

    if make_plots:
        plt.plot(post_q, post_int, label='Post Edge Intensity')
        plt.plot(pre_q, pre_int, label='Pre Edge Intensity')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title('Pre and Post Edge Intensity')
        plt.xlabel('$\it{q}$ (Å$^{-1}$)')
        plt.ylabel('Intensity (a.u.)')
        plt.show()

    # Initial guess for parameters used for the least squares fit.
    # The initial guess is chosen based on empirical observation and understanding of the underlying physics.
    # init_A_1: Initial guess for the amplitude (A) of the exponential for the pre-edge region.
    # init_A_2: Initial guess for the amplitude (A) of the exponential for the post-edge region.
    # init_B: Initial guess for the power to which q is raised (B), it's shared between the two regions (B is the XRF power law fit parameter).
    # init_C_1: Initial guess for the constant offset (C) in the pre-edge region.
    # init_C_2: Initial guess for the constant offset (C) in the post-edge region.
    init_A_1 = 1e-5
    init_A_2 = 1e-5
    init_B = -2
    init_C_1 = 0
    init_C_2 = 2
    init_guess = [init_A_1, init_A_2, init_B, init_C_1, init_C_2]

    # Fitting process using least squares fit. Here, the fit is performed over the pre-edge and post-edge regions
    result = least_squares(err_global, init_guess, bounds=([0, 0, -np.inf, 0, 0], [np.inf, np.inf, 0, np.inf, np.inf]), args=(pre_q, post_q, pre_int, post_int))
    
    # Extracting optimal parameters from fit
    para_best = result.x
    para_best1 = [para_best[0], para_best[2], para_best[3]]
    para_best2 = [para_best[1], para_best[2], para_best[4]]

    # Parameter B is taken as the XRF power law fit from the optimized parameters
    xrf_power_law_fit = para_best[2]
    
    # Plot of the fitted model and the original data        
    if make_plots:
        print(f"XRF Power Law Fit: {xrf_power_law_fit}")
        sample.sel(energy=pre, method='nearest').mean('chi').plot(yscale='log', xscale='log', label='Pre-edge')
        sample.sel(energy=post, method='nearest').mean('chi').plot(yscale='log', xscale='log', label='Post-edge')
        plt.plot(pre_q, exp_func(pre_q, para_best1), label='Pre-edge fit', color='blue', linewidth=3)
        plt.plot(post_q, exp_func(post_q, para_best2), label='Post-edge fit', color='orange', linewidth=3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xlabel('$\it{q}$ (Å$^{-1}$)')
        plt.ylabel('Intensity (a.u.)')
        plt.show()

    # The second exponential model function used for fitting at each energy. B is fixed to xrf_power_law_fit.
    def exp_func2(q, para):
        A, C = para
        return A * q**xrf_power_law_fit + C

    # The error function associated with the second exponential model
    def err2(para, q, y):
        return exp_func2(q, para) - y

    # Creating a copy of the original xarray to store the XRF background-subtracted data
    xrf_subtracted_data = sample.copy()
    xrf_fit_C = []  # List to store the fitted C values

    # Initialization of parameters for the iterative least squares fitting process
    # Here, we use the fitted amplitude and offset from the previous global fit as initial guesses.
    # init_A_iter: Initial guess for the amplitude (A) for iterative fitting at each energy level. It's based on the amplitude of the pre-edge fit.
    # init_C_iter: Initial guess for the constant offset (C) for iterative fitting at each energy level. It's based on the offset of the pre-edge fit.
    init_A_iter = para_best1[0]
    init_C_iter = para_best1[2]
    init_guess = [init_A_iter, init_C_iter]
    
    if printE is None:
        printE = (pre + post) / 2  # Default printE to the midpoint between pre and post

    # Flag to indicate if an error occurred at printE
    error_at_printE = False
    
    # catch all warnings to be displayed at the end for the user
    with warnings.catch_warnings(record=True) as caught_warnings:
        # Loop over each energy in the sample
        for v in sample.energy:
            intensity = sample.sel(energy=v, q=slice(q_lower, q_upper)).mean('chi').values.flatten()
            q_values = sample.sel(energy=v, q=slice(q_lower, q_upper)).mean('chi')['q']

            try:
                # Perform the least squares fit for the current energy. The fitted model is used to calculate the XRF background.
                result = least_squares(err2, init_guess, bounds=([0, 0], [np.inf, np.inf]), args=(q_values, intensity))
                para_best = result.x
                xrf_fit_C.append(para_best[1])

                # Subtract the constant offset for the current energy from all q values. This is the background subtraction step.
                xrf_subtracted_data.loc[{'energy': v}] -= para_best[1]
            except ValueError:
                # If an error occurs during the fit, the original data is kept and a warning is issued
                xrf_subtracted_data.loc[{'energy': v}] = sample.loc[{'energy': v}]
                xrf_fit_C.append(np.nan)
                warnings.warn(f'ValueError at {np.asarray(v)} eV. Original data used, XRF fit skipped. Please check data at this energy.')

            # Flag to mark if an error occurred at printE
            if v == sample.sel(energy=printE, method='nearest').energy:
                error_at_printE = True

            # The initial guess for the next iteration is updated to be the optimized parameters from the current iteration
            init_guess = para_best  

            # Plotting for data at printE if specified
            if v == sample.sel(energy=printE, method='nearest').energy and make_plots and not error_at_printE:
                sample.sel(energy=printE, method='nearest').mean('chi').plot(yscale='log', xscale='log', label=str(np.asarray(v)) + ' eV Data')
                plt.plot(q_values, exp_func2(q_values, para_best), label=str(np.asarray(v)) + ' eV Fit', color='blue', linewidth=3)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.xlabel('$\it{q}$ (Å$^{-1}$)')
                plt.ylabel('Intensity (a.u.)')
                plt.show()
    
    # Final plot of the fitted XRF offset as a function of energy
    if make_plots:
        plt.plot(sample.energy, xrf_fit_C, marker='.')
        plt.xlabel('Energy (eV)')
        plt.ylabel('XRF Offset Value (a.u.)')
        plt.show()

    # Print out all caught warnings
    for warn in caught_warnings:
        print(str(warn.message))
        
    # Constructing the xrf_fit xarray, which only depends on the energy dimension
    xrf_fit_C = xr.DataArray(xrf_fit_C, coords=[sample.energy], dims=['energy'])
    
    return xrf_subtracted_data, xrf_fit_C