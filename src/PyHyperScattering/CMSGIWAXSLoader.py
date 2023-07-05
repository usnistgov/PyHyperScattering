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
    pass
