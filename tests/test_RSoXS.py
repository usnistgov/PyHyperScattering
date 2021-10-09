import sys,os
sys.path.append("src/")

from PyHyperScattering.load import cyrsoxsLoader
from PyHyperScattering.integrate import WPIntegrator

import xarray as xr
import pathlib
import numpy as np
import unittest
#import HDR


def test_rsoxs_chi_slice():
        load = cyrsoxsLoader()
        integ = WPIntegrator()

        raw = load.loadDirectory(pathlib.Path('2021-09-03_pl-0.1_cyl10_core5_sp60'))
        reduced = integ.integrateImageStack(raw)

        assert type(reduced)==xr.DataArray
        
        test_chi_slice_both_outside_negative(data)
        test_chi_slice_both_outside_positive(data)
        test_chi_slice_range_too_wide(data)
        test_chi_slice_span_n180(data)
        test_chi_slice_span_p180(data)
        
        test_chi_select_outside_positive(data)
        test
        
        return data

def test_chi_slice_both_outside_negative(data):
        assert(np.allclose(data.rsoxs.slice_chi(-270),data.rsoxs.slice_chi(90)))

def test_chi_slice_both_outside_positive(data):
        assert(np.allclose(data.rsoxs.slice_chi(90),data.rsoxs.slice_chi(450)))

def test_chi_slice_range_too_wide(data):
        with unittest.assertWarns(UserWarning):
            data.rsoxs.slice_chi(0,chi_width=540)
            
def test_chi_slice_span_n180(data):
        assert(np.allclose(data.rsoxs.slice_chi(-180),
                          xr.concat(
                          [
                              data.rsoxs.sel(chi=slice(-170,-180)).sum('chi'),
                              data.rsoxs.sel(chi=slice(170,179)).sum('chi')
                          ],axis='chi').sum('chi')
                          ))    
def test_chi_slice_span_p180(data):
        assert(np.allclose(data.rsoxs.slice_chi(180),
                          xr.concat(
                          [
                              data.rsoxs.sel(chi=slice(170,180)).sum('chi'),
                              data.rsoxs.sel(chi=slice(-179,-170)).sum('chi')
                          ],axis='chi').sum('chi')
                          ))
        
def test_chi_select_outside_positive(data):
        assert(np.allclose(data.rsoxs.select_chi(450),data.rsoxs_select_chi(90)))
def test_chi_select_outside_negative(data):
        assert(np.allclose(data.rsoxs.select_chi(-270),data.rsoxs_select_chi(90)))

        