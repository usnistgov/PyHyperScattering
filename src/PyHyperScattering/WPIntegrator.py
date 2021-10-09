MACHINE_HAS_CUDA = True
from PyHyperScattering.FileLoader import FileLoader
import os
import xarray as xr
import pandas as pd
import numpy as np
import warnings
import re
import os
import datetime
import time
import h5py
import skimage
try:
    import cupy as cp
    import cupyx.scipy.ndimage as ndigpu
except ImportError:
    MACHINE_HAS_CUDA=False
    warnings.warn('Could not import CuPy or ndigpu.  If you expect this machine to support CuPy, check dependencies.  Falling back to scikit-image/numpy CPU integration.',stacklevel=2)


class WPIntegrator():
    '''
    Integrator for qx/qy format xarrays using skimage.transform.warp_polar or a custom cuda-accelerated version, warp_polar_gpu
    '''
    
    def __init__(self,return_cupy=False,force_np_backend=False):
        '''
        Args:
            return_cupy (bool, default False): return arrays as cupy rather than numpy, for further GPU processing
            force_np_backend (bool, default False): if true, use numpy backend regardless of whether CuPy is available. 
        '''
        if MACHINE_HAS_CUDA and not force_np_backend:
            self.MACHINE_HAS_CUDA = True
        else:
            self.MACHINE_HAS_CUDA = False
            
        self.return_cupy = return_cupy
    
    def warp_polar_gpu(self,image, center=None, radius=None, output_shape=None, **kwargs):
        """
        Function to emulate warp_polar in skimage.transform on the GPU. Not all
        parameters are supported

        Parameters
        ----------
        image: cupy.ndarray
            Input image. Only 2-D arrays are accepted.         
        center: tuple (row, col), optional
            Point in image that represents the center of the transformation
            (i.e., the origin in cartesian space). Values can be of type float.
            If no value is given, the center is assumed to be the center point of the image.
        radius: float, optional
            Radius of the circle that bounds the area to be transformed.
        output_shape: tuple (row, col), optional

        Returns
        -------
        polar: numpy.ndarray or cupy.ndarray depending on value of return_cupy
            polar image
        """
        image = cp.asarray(image)
        if radius is None:
            radius = int(np.ceil(np.sqrt((image.shape[0] / 2)**2 + (image.shape[1] / 2)**2)))
        cx, cy = image.shape[1] // 2, image.shape[0] // 2
        if center is not None:
            cx, cy = center
        if output_shape is None:
            output_shape = (360, radius)
        delta_theta = 360 / output_shape[0]
        delta_r = radius / output_shape[1]
        t = cp.arange(output_shape[0])
        r = cp.arange(output_shape[1])
        R, T = cp.meshgrid(r, t)
        X = R * delta_r * cp.cos(cp.deg2rad(T * delta_theta)) + cx
        Y = R * delta_r * cp.sin(cp.deg2rad(T * delta_theta)) + cy
        coordinates = cp.stack([Y, X])
        polar = ndigpu.map_coordinates(image, coordinates, order=1)
        if not self.return_cupy:
            retval = cp.asnumpy(polar)
        else:
            retval = polar
        return retval
    
    def integrateSingleImage(self,img):
        img_to_integ = img.values.squeeze()
 
        center_x = float(xr.DataArray(np.linspace(0,len(img.qx)-1,len(img.qx)))
                    .assign_coords({'dim_0':img.qx.values})
                    .rename({'dim_0':'qx'})
                    .interp(qx=0)
                    .data)
        center_y = float(xr.DataArray(np.linspace(0,len(img.qy)-1,len(img.qy)))
                    .assign_coords({'dim_0':img.qy.values})
                    .rename({'dim_0':'qy'})
                    .interp(qy=0)
                    .data)        
        try:
            system_to_integ = img.system
        except AttributeError:
            pass
        
        if self.MACHINE_HAS_CUDA:
            TwoD = self.warp_polar_gpu(img_to_integ,center=(center_x,center_y))
        else:
            TwoD = skimage.transform.warp_polar(img_to_integ,center=(center_x,center_y))

        
        qx = img.qx
        qy = img.qy
        q = np.sqrt(qy**2+qx**2)
        chi = np.linspace(-179.5,179.5,360)
        q = np.linspace(0,np.amax(q), TwoD.shape[1])
        try:
            return xr.DataArray([TwoD],dims=['system','chi','q'],coords={'q':q,'chi':chi,'system':system_to_integ},attrs=img.attrs)
        except ValueError:
            return xr.DataArray(TwoD,dims=['chi','q'],coords={'q':q,'chi':chi},attrs=img.attrs)


    def integrateImageStack(self,img_stack):
        int_stack = img_stack.groupby('system').map(self.integrateSingleImage)   
        return int_stack
