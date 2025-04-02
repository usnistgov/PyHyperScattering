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
        try:
            import os
            api_key = os.environ['TILED_API_KEY']
            client = tiled.client.from_uri('https://tiled.nsls2.bnl.gov',api_key=api_key)
            SKIP_DB_TESTING=False
        except Exception:
            SKIP_DB_TESTING=True
except ImportError:
    SKIP_DB_TESTING=True



    
import numpy as np
import pandas as pd
import xarray as xr
import pytest
from PyHyperScattering.load import SST1RSoXSDB

must_have_tiled = pytest.mark.skipif(SKIP_DB_TESTING,reason='Connection to Tiled server not possible in this environment.')


@pytest.fixture(autouse=True,scope='module')
def sstdb():
    try:
        client = tiled.client.from_profile('rsoxs')
    except tiled.profiles.ProfileNotFound:
        import os
        api_key = os.environ['TILED_API_KEY']
        client = tiled.client.from_uri('https://tiled.nsls2.bnl.gov',api_key=api_key)['rsoxs']['raw']
    sstdb = SST1RSoXSDB(catalog=client,corr_mode='none')
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


## This is intended to test a scan that was run at a single energy and two polarizations
@must_have_tiled
def test_SST1DB_load_SingleEnergy2Polarizations_scan_hinted_dims(sstdb):
    run = sstdb.loadRun(87758).unstack('system')
    assert 'energy' in run.indexes
    assert 'polarization' in run.indexes
@must_have_tiled
def test_SST1DB_load_SingleEnergy2Polarizations_scan_explicit_dims(sstdb):
    run = sstdb.loadRun(87758,dims=['energy','polarization']).unstack('system')
    assert type(run) == xr.DataArray
    assert 'energy' in run.indexes
    assert 'polarization' in run.indexes



@must_have_tiled
def test_SST1DB_load_energy_scan_20241209(sstdb):
    run = sstdb.loadRun(91175).unstack('system')
    assert type(run) == xr.DataArray
    assert 'energy' in run.indexes

@must_have_tiled
def test_SST1DB_load_energy_scan_20250213(sstdb):
    run = sstdb.loadRun(92202).unstack('system')
    assert type(run) == xr.DataArray
    assert 'energy' in run.indexes

@must_have_tiled
def test_SST1DB_load_spiral_scan_20250221(sstdb):
    run = sstdb.loadRun(92770).unstack('system')
    assert type(run) == xr.DataArray
    assert 'sam_x' in run.indexes
    assert 'sam_y' in run.indexes

@must_have_tiled
def test_SST1DB_load_count_scan_20250222(sstdb):
    run = sstdb.loadRun(92849).unstack('system')
    assert type(run) == xr.DataArray
    assert 'time' in run.indexes

@must_have_tiled
def test_SST1DB_load_energy_scan_20250223(sstdb):
    run = sstdb.loadRun(93065).unstack('system')
    assert type(run) == xr.DataArray
    assert 'energy' in run.indexes

@must_have_tiled
def test_SST1DB_exposurewarnings(sstdb):
    with pytest.warns(UserWarning, match="Wide Angle CCD Detector is reported as underexposed"):
        sstdb.loadRun(83192)
    with pytest.warns(UserWarning, match="Wide Angle CCD Detector is reported as saturated"):
        sstdb.loadRun(67522)
