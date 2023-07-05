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
        image_da = xr.DataArray(image_data, dims=['pix_y', 'pix_x'])
        return image_da
    