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

@pytest.fixture(autouse=True,scope='module')
def aniso_test_data_zero_bkg(OFFSET=0,BACKGROUND=0):
    '''
    Make a sinusoidal set of anisotropic test data, with a q^-4 background that is radially symmetric and a q^-2 powerlaw that is sinusoidally distributed with max at OFFSET deg.

    Inputs:

    OFFSET (float, default 0): the angular offset of the center of the sine, in degrees
    BACKGROUND (float, default 1): a prefactor on the q^-4 background.  set to 0 to disable.
    '''
    chi = np.linspace(0,359,360)
    chi = xr.DataArray(chi,coords=[('chi',chi,{'unit':'deg'})])
    q = np.logspace(0.001,1,500)
    q = xr.DataArray(q,coords=[('q',q,{'unit':'A^-1'})])

    I_para = (np.cos(2*chi*np.pi/180+OFFSET)+1) * q**-2 + BACKGROUND* q**-4
    I_perp = (np.sin(2*chi*np.pi/180+OFFSET)+1) * q**-2 + BACKGROUND* q**-4
    aniso = xr.concat([I_para,I_perp],dim = xr.DataArray([0,90],[('polarization',[0,90],{'unit':'deg'})]))
    return aniso
    
def test_AR_unity(aniso_test_data_zero_bkg):
    AR = data.rsoxs.AR(aniso_test_data_zero_bkg)
    assert(np.allclose(AR,1,atol=1e-3))
