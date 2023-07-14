from PIL import Image
from PyHyperScattering.FileLoader import FileLoader
import os
import pathlib
import xarray as xr
import pandas as pd
import datetime
import warnings
import json
#from pyFAI import azimuthalIntegrator
import numpy as np
from tqdm.auto import tqdm 


class CMSGIWAXSLoader(FileLoader):
    """
    Loader for TIFF files from NSLS-II 11-BM CMS
    """
    def __init__(self, md_naming_scheme=[]):
        self.md_naming_scheme = md_naming_scheme

    def loadSingleImage(self, filepath):
        """
        Loads a single xarray DataArray from a filepath to a raw TIFF
        """
        image = Image.open(filepath)
        image_data = np.array(image)
        attr_dict = self.loadMd(filepath)
        image_da = xr.DataArray(data = image_data, 
                                dims=['pix_y', 'pix_x'],
                                attrs=attr_dict)
        image_da = image_da.assign_coords({
            'pix_x': image_da.pix_x.data,
            'pix_y': image_da.pix_y.data
        })
        return image_da
    
    def loadMd(self, filepath):
        """
        Uses md_naming_scheme to generate dictionary of metadata based on filename
        """
        attr_dict = {}
        name = filepath.name
        md_list = name.split('_')
        for i, md_item in enumerate(self.md_naming_scheme):
            attr_dict[md_item] = md_list[i]
        return attr_dict
    
    def loadSeries(self, files, filter='', time_start=0):
        """
        Load many raw TIFFs into an xarray DataArray

        Input: files: Either a pathlib.Path object that can be filtered with a 
                      glob filter or an iterable that contains the filepaths
        Output: xr.DataArray with appropriate dimensions & coordinates
        """

        data_rows = []
        if issubclass(type(files), pathlib.Path):
            for filepath in tqdm(files.glob(f'*{filter}*')):
                image_da = self.loadSingleImage(filepath)
                image_da = image_da.assign_coords({'series_number': int(image_da.series_number)})
                image_da = image_da.expand_dims(dim={'series_number': 1})
                data_rows.append(image_da)
        else:
            try:
                for filepath in tqdm(files):
                    image_da = self.loadSingleImage(filepath)
                    image_da = image_da.assign_coords({'series_number': int(image_da.series_number)})
                    image_da = image_da.expand_dims(dim={'series_number': 1})
                    data_rows.append(image_da)  
            except TypeError:
                warnings.warn('"files" needs to be a pathlib.Path or iterable')  
                return None      

        out = xr.concat(data_rows, 'series_number')
        out = out.sortby('series_number')
        out = out.assign_coords({
            'series_number': out.series_number.data,
            'time': ('series_number', 
                     out.series_number.data*np.round(float(out.exposure_time[:-1]),
                                                     1)+np.round(float(out.exposure_time[:-1]),1)+time_start)
        })
        out = out.swap_dims({'series_number': 'time'})
        out = out.sortby('time')
        del out.attrs['series_number']

        return out


