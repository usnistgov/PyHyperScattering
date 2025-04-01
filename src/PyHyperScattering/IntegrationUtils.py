import warnings
import xarray as xr
import numpy as np
import math
from tqdm.auto import tqdm

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm,Normalize
    import holoviews as hv
    import hvplot.xarray
    import skimage.draw
    
except (ModuleNotFoundError,ImportError):
    warnings.warn('Could not import a dependency for interactive integration utils.  Install pyhyperscattering[ui] or pyhyperscattering[all].',stacklevel=2)
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
    
    def __init__(self,frame, cmap='viridis', clim=(5e0, 5e3), width=800, height=700):
        '''
        Construct a DrawMask object

        Args:
            frame (xarray): a single data frame with pix_x and pix_y axes

        '''

        if len(frame.shape) > 2:
            warnings.warn('This tool needs a single frame, not a stack!  .sel down to a single frame before starting!',stacklevel=2)
            
        self.frame = frame
        
        self.fig = frame.hvplot(cmap=cmap, clim=clim, logz=True, data_aspect=1, 
                                width=width, height=height)

        self.poly = hv.Polygons([])
        self.path_annotator = hv.annotate.instance()

    def ui(self):
        '''
        Draw the DrawMask UI in a Jupyter notebook.


        Returns: the holoviews object

        '''
        print('Usage: click the "PolyAnnotator" tool at top right.  DOUBLE CLICK to start drawing a masked object, SINGLE CLICK to add a vertex, then DOUBLE CLICK to finish.  Click/drag individual vertex to adjust.')
        annotator_plot = self.path_annotator(
                                    self.fig * self.poly.opts(responsive=False), 
                                    annotations=['Label'], 
                                    vertex_annotations=['Value'])
        return annotator_plot.opts(toolbar='left')


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
        # print(strlist)
        dflist = []
        for item in strlist:
            dflist.append(pd.read_json(item))
        # print(dflist)
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


class CMSGIWAXS:
    """For streamlined loading for CMS data"""
    def __init__(self, files, loader, integrator):
        """
        Inputs: files: indexable object str or pathlib.Path filepaths to 
                       raw GIWAXS data
                loader: custom PyHyperScattering CMSGIWAXSLoader object, must 
                        return DataArray with attributes metadata
                integrator: instance of PGGeneralIntegrator object
        """
        self.files = files
        self.loader = loader
        self.integrator = integrator

    def single_images_to_dataset(self):
        """
        Method that takes a subscriptable object of filepaths corresponding to raw GIWAXS
        beamline data, loads the raw data into an xarray DataArray, generates pygix-transformed 
        cartesian and polar DataArrays, and creates 3 corresponding xarray Datasets 
        containing a DataArray per sample. 
        The raw dataarrays must contain the attributes 'scan_id' and 'incident_angle'

        Outputs: 2 Datasets: raw & reciprocal space (cartesian or polar based on integrator object)
        """
        # Select the first element of the sorted set outside of the for loop to initialize the xr.DataSet
        DA = self.loader.loadSingleImage(self.files[0])
        assert 'scan_id' in DA.attrs.keys(), "'scan_id' is a required attribute to use this function"

        # Update incident angle per sample:
        assert 'incident_angle' in DA.attrs.keys(), "'incident_angle' is a required attribute to use this function"
        self.integrator.incident_angle = float(DA.incident_angle[2:])

        # Integrate single image
        integ_DA = self.integrator.integrateSingleImage(DA)

        # Save coordinates for interpolating other dataarrays 
        integ_coords = integ_DA.coords

        # Create a DataSet, each DataArray will be named according to it's scan id
        raw_DS = DA.to_dataset(name=DA.scan_id)
        integ_DS = integ_DA.to_dataset(name=DA.scan_id)

        # Populate the DataSet with 
        for filepath in tqdm(self.files[1:], desc=f'Transforming Raw Data'):
            DA = self.loader.loadSingleImage(filepath)
            integ_DA = self.integrator.integrateSingleImage(DA)
            
            integ_DA = integ_DA.interp(integ_coords)

            raw_DS[f'{DA.scan_id}'] = DA
            integ_DS[f'{DA.scan_id}'] = integ_DA

        return raw_DS, integ_DS
