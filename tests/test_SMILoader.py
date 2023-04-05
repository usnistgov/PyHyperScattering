import sys
sys.path.append("src/")

from PyHyperScattering.load import SST1RSoXSLoader

from PyHyperScattering.integrate import PFEnergySeriesIntegrator
import numpy as np
import pandas as pd
import xarray as xr
import pytest

@pytest.fixture(autouse=True,scope='module')
def smiloader():
    smiloader = SMIRSoXSLoader()
    return smiloader

@pytest.fixture(autouse=True,scope='module')
def SMI_single_scan(smiloader):
    return smiloader.loadFileSeries('smi_example/PN_AGBEH_',['energy','polarization'])

def test_SMI_single_scan_import(SMI_single_scan):
    assert type(SMI_single_scan)==xr.DataArray
