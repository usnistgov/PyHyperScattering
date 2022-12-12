import sys,os
sys.path.append("src/")

from PyHyperScattering.load import cyrsoxsLoader
from PyHyperScattering.integrate import WPIntegrator

import xarray as xr
import pathlib
#import HDR
import math
import unittest
import pytest

@pytest.fixture(autouse=True,scope='module')
def wp_integrator_legacy():
    integ = WPIntegrator()
    return integ

@pytest.fixture(autouse=True,scope='module')
def wp_integrator_dask():
    integ = WPIntegrator(use_chunked_processing=True)
    return integ

@pytest.fixture(autouse=True,scope='module')
def cyrsoxs_data():
        load = cyrsoxsLoader()
        raw = load.loadDirectory(pathlib.Path('2021-09-03_pl-0.1_cyl10_core5_sp60'))
        return raw

def test_wp_integrator_legacy(cyrsoxs_data,wp_integrator_legacy):
        reduced = wp_integrator_legacy.integrateImageStack(cyrsoxs_data)
        assert type(reduced)==xr.DataArray

def test_wp_integrator_dask(cyrsoxs_data,wp_integrator_dask):
        reduced = wp_integrator_dask.integrateImageStack(cyrsoxs_data)
        assert type(reduced)==xr.DataArray

