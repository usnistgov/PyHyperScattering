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
    time_series_scheme = ['material', 'solvent', 'concentration', 'gap_height', 
                          'blade_speed','solution_temperature', 
                          'stage_temperature', 'sample_number', 'time_start',
                          'x_position_offset', 'incident_angle', 
                          'exposure_time', 'scan_id','series_number', 
                          'detector']
    cmsloader = CMSGIWAXSLoader(md_naming_scheme = time_series_scheme)
    return cmsloader

@pytest.fixture(autouse=True,scope='module')
def CMS_giwaxs_series(cmsloader):
    return cmsloader.loadFileSeries('CMS_giwaxs_series/pybtz_time_series',['series_number'])

def test_CMS_giwaxs_series_import(CMS_giwaxs_series):
    assert type(CMS_giwaxs_series)==xr.DataArray

def test_load_insensitive_to_trailing_slash(cmsloader):
    withslash = cmsloader.loadFileSeries('CMS_giwaxs_series/pybtz_time_series/',['series_number'])
        
    withoutslash = cmsloader.loadFileSeries('CMS_giwaxs_series/pybtz_time_series',['series_number'])
        
    assert np.allclose(withslash,withoutslash)