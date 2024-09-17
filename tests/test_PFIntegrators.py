import sys,os
sys.path.append("src/")

from PyHyperScattering.load import cyrsoxsLoader
from PyHyperScattering.integrate import WPIntegrator

import xarray as xr
import pathlib
import numpy as np
import math
import unittest
import pytest

from PyHyperScattering.load import SST1RSoXSLoader

from PyHyperScattering.integrate import PFEnergySeriesIntegrator
from PyHyperScattering.integrate import PFGeneralIntegrator
#import HDR

@pytest.fixture(autouse=True,scope='module')
def sst_data():
        loader = SST1RSoXSLoader(corr_mode='none')
        return loader.loadFileSeries('Example/SST1/21792/',['energy','polarization'])
    
@pytest.fixture()#autouse=True,scope='module')
def pfgenint(sst_data):
    integrator = PFGeneralIntegrator(maskmethod='none',geomethod='template_xr',template_xr=sst_data)
    return integrator
    
@pytest.fixture()#autouse=True,scope='module')
def pfesint(sst_data):
    integrator = PFEnergySeriesIntegrator(maskmethod='none',geomethod='template_xr',template_xr=sst_data)
    return integrator

@pytest.fixture()#autouse=True,scope='module')
def pfesint_dask(sst_data):
    integrator = PFEnergySeriesIntegrator(maskmethod='none',geomethod='template_xr',template_xr=sst_data,use_chunked_processing=True)
    return integrator
@pytest.fixture()#autouse=True,scope='module')
def pfgenint_dask(sst_data):
    integrator = PFGeneralIntegrator(maskmethod='none',geomethod='template_xr',template_xr=sst_data,use_chunked_processing=True)
    return integrator



def test_integrator_loads_nika_mask_tiff(pfesint):
    pfesint.loadNikaMask(maskpath=pathlib.Path('mask-test-pack/37738-CB_TPD314K1_mask.tif'))
def test_integrator_loads_nika_mask_hdf5(pfesint):
    pfesint.loadNikaMask(maskpath=pathlib.Path('mask-test-pack/SST1-SAXS_mask.hdf'))
    
def test_integrator_loads_polygon_mask(pfesint):
    pfesint.loadPolyMask(maskpoints=[[[367, 545], [406, 578], [880, 0], [810, 0]]],maskshape=(1024,1026))
                         
def test_integrator_loads_pyhyper_mask(pfesint):
    pfesint.loadPyHyperMask(maskpath=pathlib.Path('mask-test-pack/liquidMask.json'),maskshape=(1024,1026))     
    
def test_integrator_beamcenter_to_poni(pfesint):
    pfesint.nika_beamcenter_x = 400
    pfesint.nika_beamcenter_y = 600
    assert(math.isclose(pfesint.poni1,0.029445))
    assert(math.isclose(pfesint.poni2,0.0293916))

def test_integration_runs_en_series_legacy_2dim_mi(sst_data,pfesint):
    pfesint.integrateImageStack(sst_data)

@pytest.mark.skip('broken due to upstream issue?')
def test_integration_runs_en_series_dask_2dim_mi(sst_data,pfesint_dask):
    pfesint_dask.integrateImageStack(sst_data).compute()

def test_integration_runs_gen_legacy_2dim_mi(sst_data,pfgenint):
    pfgenint.integrateImageStack(sst_data)

@pytest.mark.skip('broken due to upstream issue?')
def test_integration_runs_gen_dask_2dim_mi(sst_data,pfgenint_dask):
    pfgenint_dask.integrateImageStack(sst_data).compute()
    
def test_integration_runs_en_series_legacy_2dim(sst_data,pfesint):
    pfesint.integrateImageStack(sst_data.unstack('system'))

def test_integration_runs_en_series_dask_2dim(sst_data,pfesint_dask):
    pfesint_dask.integrateImageStack(sst_data.unstack('system'),chunksize=6).compute()

def test_integration_runs_gen_legacy_2dim(sst_data,pfgenint):
    pfgenint.integrateImageStack(sst_data.unstack('system'))

def test_integration_runs_gen_dask_2dim(sst_data,pfgenint_dask):
    pfgenint_dask.integrateImageStack(sst_data.unstack('system'),chunksize=6).compute()




def test_integration_runs_en_series_legacy_1dim_mi(sst_data,pfesint):
    pfesint.integrateImageStack(sst_data.sel(polarization=90).stack(system=['energy']))

@pytest.mark.skip('broken due to upstream issue?')
def test_integration_runs_en_series_dask_1dim_mi(sst_data,pfesint_dask):
    pfesint_dask.integrateImageStack(sst_data.sel(polarization=90).stack(system=['energy']),chunksize=6).compute()

def test_integration_runs_gen_legacy_1dim_mi(sst_data,pfgenint):
    pfgenint.integrateImageStack(sst_data.sel(polarization=90).stack(system=['energy']))

@pytest.mark.skip('broken due to upstream issue?')
def test_integration_runs_gen_dask_1dim_mi(sst_data,pfgenint_dask):
    pfgenint_dask.integrateImageStack(sst_data.sel(polarization=90).stack(system=['energy']),chunksize=6).compute()
    
def test_integration_runs_en_series_legacy_1dim(sst_data,pfesint):
    pfesint.integrateImageStack(sst_data.sel(polarization=90))

def test_integration_runs_en_series_dask_1dim(sst_data,pfesint_dask):
    pfesint_dask.integrateImageStack(sst_data.sel(polarization=90),chunksize=6).compute()

def test_integration_runs_gen_legacy_1dim(sst_data,pfgenint):
    pfgenint.integrateImageStack(sst_data.sel(polarization=90))

def test_integration_runs_gen_dask_1dim(sst_data,pfgenint_dask):
    pfgenint_dask.integrateImageStack(sst_data.sel(polarization=90),chunksize=6).compute()

