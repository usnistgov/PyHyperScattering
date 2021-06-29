import warnings
import xarray as xr
import numpy as np
import math

try:
    import holoviews as hv
    import hvplot.xarray

    import skimage.draw
except (ModuleNotFoundError,ImportError):
    warnings.warn('Could not import package for interactive integration utils.  Install holoviews and scikit-image.')
import pandas as pd

import json

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
            warnings.warn('This tool needs a single frame, not a stack!  .sel down to a single frame before starting!')
        
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
        
        
    @property
    def mask(self):
        '''
        Render the mask as a numpy boolean array.
        '''
        mask = np.zeros(self.frame.shape).astype(bool)
        for i in range(len(self.path_annotator.annotated)):
            mask |= skimage.draw.polygon2mask(self.frame.shape,self.path_annotator.annotated.iloc[i].dframe(['x','y']))

        return mask