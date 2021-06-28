import warnings
import xarray as xr
import numpy as np
import math

import holoviews as hv
import hvplot.xarray
import skimage.draw

import pandas as pd

import json

class DrawMask:
    
    def __init__(self,frame,existing_mask=None):
        if len(frame.shape) > 2:
            warnings.warn('This tool needs a single frame, not a stack!  .sel down to a single frame before starting!')
        
        self.frame=frame
        
        self.fig = frame.hvplot(cmap='terrain',clim=(5,5000),logz=True,data_aspect=1)

        self.poly = hv.Polygons([])
        self.path_annotator = hv.annotate.instance()

    def ui(self):
        print('Usage: click the "PolyAnnotator" tool at top right.  DOUBLE CLICK to start drawing a masked object, SINGLE CLICK to add a vertex, then DOUBLE CLICK to finish.  Click/drag individual vertex to adjust.')
        return self.path_annotator(
                self.fig * self.poly.opts(
                            width=self.frame.shape[0], 
                            height=self.frame.shape[1], 
                            responsive=False), 
                annotations=['Label'], 
            vertex_annotations=['Value'])


    def save(self,fname):
        dflist = []
        for i in range(len(self.path_annotator.annotated)):
            dflist.append(self.path_annotator.annotated.iloc[i].dframe(['x','y']).to_json())
        
        with open(fname, 'w') as outfile:
            json.dump(dflist, outfile)
            
    def load(self,fname):
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
        mask = np.zeros(self.frame.shape).astype(bool)
        for i in range(len(self.path_annotator.annotated)):
            mask |= skimage.draw.polygon2mask(self.frame.shape,self.path_annotator.annotated.iloc[i].dframe(['x','y']))

        return mask