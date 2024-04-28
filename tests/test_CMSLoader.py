import sys
sys.path.append("src/")

from PyHyperScattering.load import CMSGIWAXSLoader
from PyHyperScattering.integrate import PGGeneralIntegrator

import numpy as np
import pandas as pd
import xarray as xr
import pytest

@pytest.fixture(autouse=True,scope='module')
def cmsloader():
    cmsloader = CMSGIWAXSLoader()
    return cmsloader

# Uncomment below once example data zip is added
# @pytest.fixture(autouse=True,scope='module')
# def CMS_giwaxs_series(cmsloader):
#     return cmsloader.loadFileSeries('CMS_giwaxs_series/pybtz_time_series',['time'])

# def test_CMS_giwaxs_series_import(CMS_giwaxs_series):
#     assert type(CMS_giwaxs_series)==xr.DataArray

# def test_load_insensitive_to_trailing_slash(cmsloader):
#     withslash = cmsloader.loadFileSeries('CMS_giwaxs_series/pybtz_time_series/',['time'])
        
#     withoutslash = cmsloader.loadFileSeries('CMS_giwaxs_series/pybtz_time_series',['time'])
        
#     assert np.allclose(withslash,withoutslash)