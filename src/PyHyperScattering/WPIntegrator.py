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
try:
    import dask.array as da
    import dask
except ImportError:
    warnings.warn('Failed to import Dask, if Dask reduction is desired install pyhyperscattering[performance]',stacklevel=2)


class WPIntegrator():
    '''
    Integrator for qx/qy format xarrays using skimage.transform.warp_polar or a custom cuda-accelerated version, warp_polar_gpu
    '''
    
    def __init__(self,return_cupy=False,force_np_backend=False,use_chunked_processing=False):
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
        self.use_chunked_processing=use_chunked_processing
    
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
            stacked_axis = list(img.coords)
            stacked_axis.remove('qx')
            stacked_axis.remove('qy')
            assert len(stacked_axis)==1, f"More than one axis left ({stacked_axis}) after removing qx and qy, not sure how to handle"
            stacked_axis = stacked_axis[0]
            system_to_integ = img.__getattr__(stacked_axis)
        except AttributeError:
            pass
        
        if self.MACHINE_HAS_CUDA:
            TwoD = self.warp_polar_gpu(img_to_integ,center=(center_x,center_y), radius = np.sqrt((img_to_integ.shape[0] - center_x)**2 + (img_to_integ.shape[1] - center_y)**2))
        else:
            TwoD = skimage.transform.warp_polar(img_to_integ,center=(center_x,center_y), radius = np.sqrt((img_to_integ.shape[0] - center_x)**2 + (img_to_integ.shape[1] - center_y)**2))

        
        qx = img.qx
        qy = img.qy
        q = np.sqrt(qy**2+qx**2)
        q = np.linspace(0,float(np.amax(q)), int(TwoD.shape[1]))

        # warp_polar maps to 0-360 instead of -180-180
        chi = np.linspace(-179.5,179.5,360)
        # chi = np.linspace(0.5,359.5,360)
        
        try:
            return xr.DataArray([TwoD],dims=[stacked_axis,'chi','q'],coords={'q':q,'chi':chi,stacked_axis:system_to_integ},attrs=img.attrs)
        except ValueError:
            return xr.DataArray(TwoD,dims=['chi','q'],coords={'q':q,'chi':chi},attrs=img.attrs)


    def integrateImageStack(self,img_stack,method=None,chunksize=None):
        '''
        
        '''
        if (self.use_chunked_processing and method is None) or method=='dask':
            func_args = {}
            if chunksize is not None:
                func_args['chunksize'] = chunksize
            return self.integrateImageStack_dask(img_stack,**func_args)
        elif (method is None) or method == 'legacy':
            return self.integrateImageStack_legacy(img_stack)
        else:
            raise NotImplementedError(f'unsupported integration method {method}')

    def integrateImageStack_legacy(self,data):
        #int_stack = img_stack.groupby('system').map(self.integrateSingleImage)   
        #return int_stack
        indexes = list(data.indexes.keys())
        try:
            indexes.remove('pix_x')
            indexes.remove('pix_y')
        except ValueError:
            pass
        try:
            indexes.remove('qx')
            indexes.remove('qy')
        except ValueError:
            pass
        
        if len(indexes) == 1:
            if data.__getattr__(indexes[0]).to_pandas().drop_duplicates().shape[0] != data.__getattr__(indexes[0]).shape[0]:
                warnings.warn(f'Axis {indexes[0]} contains duplicate conditions.  This is not supported and may not work.  Try adding additional coords to separate image conditions',stacklevel=2)
            data_int = data.groupby(indexes[0],squeeze=False).progress_apply(self.integrateSingleImage)
        else:
            #some kinda logic to check for existing multiindexes and stack into them appropriately maybe
            data = data.stack({'pyhyper_internal_multiindex':indexes})
            if data.pyhyper_internal_multiindex.to_pandas().drop_duplicates().shape[0] != data.pyhyper_internal_multiindex.shape[0]:
                warnings.warn('Your index set contains duplicate conditions.  This is not supported and may not work.  Try adding additional coords to separate image conditions',stacklevel=2)
        
            data_int = data.groupby('pyhyper_internal_multiindex',squeeze=False).progress_apply(self.integrateSingleImage).unstack('pyhyper_internal_multiindex')
        return data_int
    
    
    def integrateImageStack_dask(self,data,chunksize=5):
        #int_stack = img_stack.groupby('system').map(self.integrateSingleImage)   
        #return int_stack
        indexes = list(data.indexes.keys())
        try:
            indexes.remove('pix_x')
            indexes.remove('pix_y')
        except ValueError:
            pass
        try:
            indexes.remove('qx')
            indexes.remove('qy')
        except ValueError:
            pass
        
        if len(indexes) == 1:
            if data.__getattr__(indexes[0]).to_pandas().drop_duplicates().shape[0] != data.__getattr__(indexes[0]).shape[0]:
                warnings.warn(f'Axis {indexes[0]} contains duplicate conditions.  This is not supported and may not work.  Try adding additional coords to separate image conditions',stacklevel=2)
            
            fake_image_to_process = data.isel(**{indexes[0]:0},drop=False)
            data = data.chunk({indexes[0]:chunksize})
        else:
            #some kinda logic to check for existing multiindexes and stack into them appropriately maybe
            data = data.stack({'pyhyper_internal_multiindex':indexes})
            if data.pyhyper_internal_multiindex.to_pandas().drop_duplicates().shape[0] != data.pyhyper_internal_multiindex.shape[0]:
                warnings.warn('Your index set contains duplicate conditions.  This is not supported and may not work.  Try adding additional coords to separate image conditions',stacklevel=2)
                
            fake_image_to_process = data.isel(**{'pyhyper_internal_multiindex':0},squeeze=False)
            data = data.chunk({'pyhyper_internal_multiindex':chunksize})
        coord_dict = {}
        shape = tuple([])
        demo_integration = self.integrateSingleImage(fake_image_to_process)
        coord_dict.update({'chi':demo_integration.chi,'q':demo_integration.q})
        npts_q = len(demo_integration.q)
              
        order_list = []
        for idx in indexes:
            order_list.append(idx)
            coord_dict[idx] = data.indexes[idx]
            shape = shape + tuple([len(data.indexes[idx])])
        shape = shape + (360,npts_q)
        print(shape)
        
        desired_order_list = order_list+['chi','q']
        coord_dict_sorted = {k: coord_dict[k] for k in desired_order_list}
        
        template = xr.DataArray(np.empty(shape),coords=coord_dict_sorted)  
        template = template.chunk({indexes[0]:chunksize})
        integ_fly = data.map_blocks(self.integrateImageStack_legacy,template=template)#integ_traditional.chunk({'energy':5}))
        return integ_fly 
            
