import sys
sys.path.append("src/")

from PyHyperScattering.load import SST1RSoXSLoader

from PyHyperScattering.integrate import PFEnergySeriesIntegrator
import numpy as np
import pandas as pd
import xarray as xr

def test_loader_imports_cleanly():
	global loader
	loader = SST1RSoXSLoader(corr_mode='none')

def test_SST1_single_scan_import():
	global loader
	return loader.loadFileSeries('Example/SST1/21792/',['energy','polarization'])

def test_SST1_single_scan_qxy_import():
		global loader
		return loader.loadFileSeries('Example/SST1/21792/',['energy','polarization'],output_qxy=True)

def test_load_insensitive_to_trailing_slash():
		withslash = loader.loadFileSeries('Example/SST1/21792/',['energy','polarization'])
        
		withoutslash = loader.loadFileSeries('Example/SST1/21792',['energy','polarization'])
        
		assert np.allclose(withslash,withoutslash)