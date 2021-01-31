import sys
sys.path.append("PyHyperScattering/")

from SST1RSoXSLoader import SST1RSoXSLoader

from PFEnergySeriesIntegrator import PFEnergySeriesIntegrator
import numpy as np
import pandas as pd
import xarray as xr

def test_loader_imports_cleanly():
	global loader
	loader = SST1RSoXSLoader(corr_mode='none')

def test_SST1_single_scan_import():
	global loader
	return loader.loadFileSeries('Example/SST1/21792/',['energy','polarization'])
