import sys
sys.path.append("src/")

from PyHyperScattering.load import SST1RSoXSLoader

from PyHyperScattering.integrate import PFEnergySeriesIntegrator
import numpy as np
import pandas as pd
import xarray as xr

@pytest.fixture(autouse=True,scope='module')
def sstloader():
    sstloader = SST1RSoXSLoader(corr_mode='none')
    return sstloader

def test_SST1_single_scan_import(sstloader):
    return sstloader.loadFileSeries('Example/SST1/21792/',['energy','polarization'])

def test_SST1_single_scan_qxy_import(sstloader):
    return sstloader.loadFileSeries('Example/SST1/21792/',['energy','polarization'],output_qxy=True)

def test_load_insensitive_to_trailing_slash(sstloader):
    withslash = sstloader.loadFileSeries('Example/SST1/21792/',['energy','polarization'])
        
    withoutslash = sstloader.loadFileSeries('Example/SST1/21792',['energy','polarization'])
        
    assert np.allclose(withslash,withoutslash)