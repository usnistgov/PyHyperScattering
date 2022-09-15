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
def filenumber_coord():
    files = os.listdir('Example/11012/CCD/')

    filenumber_coord = {}
    for file in files:
        if '.fits' in file:
            filenumber_coord.update({file:int(file[-10:-5])})

    return filenumber_coord

def test_11012_single_scan_import(alsloader,filenumber_coord):
    res = alsloader.loadFileSeries(
                                'Example/11012/CCD/',
                               ['energy','polarization','exposure','filenumber'],
                               coords = {'filenumber':filenumber_coord},
                               md_filter={'sampleid':1,'CCD Shutter Inhibit':0}
                              )
    assert type(res)==xr.DataArray
    return res


def test_11012_single_scan_qxy_import(alsloader,filenumber_coord):
    return alsloader.loadFileSeries(
                                'Example/11012/CCD/',
                               ['energy','polarization','exposure','filenumber'],
                               coords = {'filenumber':filenumber_coord},
                               md_filter={'sampleid':1,'CCD Shutter Inhibit':0},
                              output_qxy=True)

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