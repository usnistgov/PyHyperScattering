import sys,os
sys.path.append("src/")

from PyHyperScattering.load import cyrsoxsLoader
from PyHyperScattering.integrate import WPIntegrator

import xarray as xr
import pathlib
import numpy as np
import unittest
import pytest
#import HDR

@pytest.fixture(autouse=True,scope='module')
def data():
        load = cyrsoxsLoader()
        integ = WPIntegrator()

        raw = load.loadDirectory(pathlib.Path('2021-09-03_pl-0.1_cyl10_core5_sp60'))
        reduced = integ.integrateImageStack(raw)

        assert type(reduced)==xr.DataArray
        return reduced

def test_chi_slice_outside_negative(data):
        assert(np.allclose(data.rsoxs.slice_chi(-270),data.rsoxs.slice_chi(90),
                           equal_nan=True))

def test_chi_slice_outside_positive(data):
        assert(np.allclose(data.rsoxs.slice_chi(90),data.rsoxs.slice_chi(450),
                           equal_nan=True))

def test_chi_slice_range_too_wide(data):
        with pytest.warns(UserWarning):
            data.rsoxs.slice_chi(0,chi_width=540)
            
def test_chi_slice_span_n180(data):
        assert(np.allclose(data.rsoxs.slice_chi(-180),
                          xr.concat(
                          [
                              data.sel(chi=slice(-180,-175)),
                              data.sel(chi=slice(175,180))
                          ],dim='chi').mean('chi'),
                          equal_nan=True))    
def test_chi_slice_span_p180(data):
        assert(np.allclose(data.rsoxs.slice_chi(180),
                          xr.concat(
                          [
                              data.sel(chi=slice(175,180)),
                              data.sel(chi=slice(-180,-175))
                          ],dim='chi').mean('chi'),
                          equal_nan=True))

        
def test_chi_select_outside_positive(data):
        assert(np.allclose(data.rsoxs.select_chi(450),data.rsoxs.select_chi(90),equal_nan=True))
def test_chi_select_outside_negative(data):
        assert(np.allclose(data.rsoxs.select_chi(-270),data.rsoxs.select_chi(90),equal_nan=True))

        