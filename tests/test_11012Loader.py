import sys,os
sys.path.append("src/")

from PyHyperScattering.load import ALS11012RSoXSLoader

from PyHyperScattering.integrate import PFEnergySeriesIntegrator

import xarray as xr

#import HDR

def test_loader_imports_cleanly():
	global loader
	loader = ALS11012RSoXSLoader(corr_mode='expt',dark_pedestal=200)
	loader.loadSampleSpecificDarks("Example/11012/CCD/",md_filter={'sampleid':1})
def test_custom_coord_creation():
	global filenumber_coord

	files = os.listdir('Example/11012/CCD/')

	filenumber_coord = {}
	for file in files:
 	   if '.fits' in file:
 	       filenumber_coord.update({file:int(file[-10:-5])})

	return filenumber_coord
def test_11012_single_scan_import():
	global loader 
	return loader.loadFileSeries(
                                'Example/11012/CCD/',
                               ['energy','polarization','exposure','filenumber'],
                               coords = {'filenumber':test_custom_coord_creation()},
                               md_filter={'sampleid':1,'CCD Shutter Inhibit':0}
                              )


def test_examine_single_scan():
	data = test_11012_single_scan_import()

	assert type(data)==xr.DataArray

