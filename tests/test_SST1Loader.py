import sys
sys.path.append("src/")

from PyHyperScattering.load import SST1RSoXSLoader

from PyHyperScattering.integrate import PFEnergySeriesIntegrator
import numpy as np
import pandas as pd
import xarray as xr
import pytest

@pytest.fixture(autouse=True,scope='module')
def sstloader():
    sstloader = SST1RSoXSLoader(corr_mode='none')
    return sstloader

@pytest.fixture(autouse=True,scope='module')
def SST1_single_scan(sstloader):
    return sstloader.loadFileSeries('Example/SST1/21792/',['energy','polarization'])

@pytest.fixture(autouse=True,scope='module')
def SST1_single_scan_qxy(sstloader):
    return sstloader.loadFileSeries('Example/SST1/21792/',['energy','polarization'],output_qxy=True)


def test_SST1_single_scan_import(SST1_single_scan):
    assert type(SST1_single_scan)==xr.DataArray
def test_SST1_single_scan_qxy_import(SST1_single_scan_qxy):
    assert type(SST1_single_scan_qxy)==xr.DataArray
    
def test_load_insensitive_to_trailing_slash(sstloader):
    withslash = sstloader.loadFileSeries('Example/SST1/21792/',['energy','polarization'])
        
    withoutslash = sstloader.loadFileSeries('Example/SST1/21792',['energy','polarization'])
        
    assert np.allclose(withslash,withoutslash)