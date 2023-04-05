import sys,os
sys.path.append("src/")

from PyHyperScattering.load import ALS11012RSoXSLoader

from PyHyperScattering.integrate import PFEnergySeriesIntegrator

import xarray as xr
import numpy as np
import pytest

#import HDR

@pytest.fixture(autouse=True,scope='module')
def alsloader():
    loader = ALS11012RSoXSLoader(corr_mode='expt',dark_pedestal=200,constant_md={'sdd':1.0,'beamcenter_x':600,'beamcenter_y':600})
    loader.loadSampleSpecificDarks("Example/11012/CCD/",md_filter={'sampleid':1})
    return loader

@pytest.fixture(autouse=True,scope='module')
def alsloader_postmar21():
    loader = ALS11012RSoXSLoader(corr_mode='expt',dark_pedestal=200,constant_md={'sdd':1.0,'beamcenter_x':600,'beamcenter_y':600},data_collected_after_mar2021=True)
    loader.loadSampleSpecificDarks('PSS300nm_C_ccd100/CCD/')
    return loader

@pytest.fixture(autouse=True,scope='module')
def filenumber_coord():
    files = os.listdir('Example/11012/CCD/')

    filenumber_coord = {}
    for file in files:
        if '.fits' in file:
            filenumber_coord.update({file:int(file[-10:-5])})

    return filenumber_coord
@pytest.fixture(autouse=True,scope='module')
def b11012_single_scan(alsloader,filenumber_coord):
    res = alsloader.loadFileSeries(
                                'Example/11012/CCD/',
                               ['energy','polarization','exposure','filenumber'],
                               coords = {'filenumber':filenumber_coord},
                               md_filter={'sampleid':1,'CCD Shutter Inhibit':0}
                              )
    return res

@pytest.fixture(autouse=True,scope='module')
def b11012_single_scan_qxy(alsloader,filenumber_coord):
    return alsloader.loadFileSeries(
                                'Example/11012/CCD/',
                               ['energy','polarization','exposure','filenumber'],
                               coords = {'filenumber':filenumber_coord},
                               md_filter={'sampleid':1,'CCD Shutter Inhibit':0},
                              output_qxy=True)
@pytest.fixture()
def b11012_new_dark_load(alsloader_postmar21):
    return alsloader_postmar21.loadFileSeries('PSS300nm_C_ccd100/CCD/',['energy'],md_filter={'CCD Camera Shutter Inhibit': 0})

def test_new_dark_load_runs(b11012_new_dark_load):
    assert type(b11012_new_dark_load)==xr.DataArray
def test_11012_single_scan_import(b11012_single_scan):
    assert type(b11012_single_scan)==xr.DataArray


def test_11012_single_scan_qxy_import(b11012_single_scan_qxy):
    assert type(b11012_single_scan_qxy) == xr.DataArray
    
def test_load_insensitive_to_trailing_slash(alsloader,filenumber_coord):
    withslash = alsloader.loadFileSeries(
                                'Example/11012/CCD/',
                               ['energy','polarization','exposure','filenumber'],
                               coords = {'filenumber':filenumber_coord},
                               md_filter={'sampleid':1,'CCD Shutter Inhibit':0}
                              )
        
    withoutslash = alsloader.loadFileSeries(
                                'Example/11012/CCD',
                               ['energy','polarization','exposure','filenumber'],
                               coords = {'filenumber':filenumber_coord},
                               md_filter={'sampleid':1,'CCD Shutter Inhibit':0}
                              )
        
    assert np.allclose(withslash,withoutslash)
