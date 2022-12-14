import sys,os
sys.path.append("src/")

from PyHyperScattering.load import cyrsoxsLoader

import xarray as xr
import pathlib
import pytest
#import HDR

@pytest.fixture(autouse=True,scope='module')
def cyrsoxs_loader():
    return cyrsoxsLoader()

@pytest.fixture(autouse=True,scope='module')
def cyrsoxs_single_scan(cyrsoxs_loader):
    return cyrsoxs_loader.loadDirectory(pathlib.Path('2021-09-03_pl-0.1_cyl10_core5_sp60'))

def test_cyrsoxs_single_scan_import(cyrsoxs_single_scan):
    assert type(cyrsoxs_single_scan) == xr.DataArray
