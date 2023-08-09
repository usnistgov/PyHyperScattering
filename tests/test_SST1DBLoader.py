import sys
sys.path.append("src/")

try:
    import tiled
    import tiled.client
    try:
        client = tiled.client.from_profile('rsoxs')
        from PyHyperScattering.load import SST1RSoXSDB
        SKIP_DB_TESTING=False
    except tiled.profiles.ProfileNotFound:
        SKIP_DB_TESTING=True
except ImportError:
    SKIP_DB_TESTING=True



    
import numpy as np
import pandas as pd
import xarray as xr
import pytest

must_have_tiled = pytest.mark.skipif(SKIP_DB_TESTING,reason='Connection to Tiled server not possible in this environment.')


@pytest.fixture(autouse=True,scope='module')
def sstdb():
    sstdb = SST1RSoXSDB(corr_mode='none')
    return sstdb

@must_have_tiled
def test_SST1DB_load_single_scan_legacy_hinted_dims(sstdb):
    run = sstdb.loadRun(21792).unstack('system')
    assert 'energy' in run.indexes

@must_have_tiled
def test_SST1DB_load_single_scan_legacy_explicit_dims(sstdb):
    run = sstdb.loadRun(21792,dims=['energy','polarization']).unstack('system')
    assert type(run) == xr.DataArray
    assert 'energy' in run.indexes
    assert 'polarization' in run.indexes

@must_have_tiled
def test_SST1DB_load_snake_scan_hinted_dims(sstdb):
    run = sstdb.loadRun(48812,dims=['sam_th','polarization']).unstack('system')
    assert type(run) == xr.DataArray
    assert 'sam_th' in run.indexes
    assert 'polarization' in run.indexes
    
    
@must_have_tiled
def test_SST1DB_load_snake_scan_explicit_dims(sstdb):
    run = sstdb.loadRun(48812).unstack('system')
    assert type(run) == xr.DataArray
    assert 'sam_th' in run.indexes
    assert 'polarization' in run.indexes
