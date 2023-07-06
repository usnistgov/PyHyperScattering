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


class CMSGIWAXSLoader(FileLoader):
    """
    Loader for TIFF files from NSLS-II 11-BM CMS
    """
    def __init__(self, md_scheme=None):
        self.md_scheme = md_scheme

    def loadSingleImage(self, filepath):
        image = Image.open(filepath)
        image_data = np.flipud(np.array(image))
        attr_dict = self.loadMd(filepath)
        image_da = xr.DataArray(data = image_data, 
                                dims=['pix_y', 'pix_x'],
                                attrs=attr_dict)
        return image_da
    
    def loadMd(self, filepath, md_naming_scheme):
        attr_dict = {}
        name = filepath.name
        md_list = name.split('_')
        for i, md_item in enumerate(md_naming_scheme):
            attr_dict[md_item] = md_list[i]
        return attr_dict

