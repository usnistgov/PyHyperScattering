import sys,os
sys.path.append("src/")

from PyHyperScattering.load import cyrsoxsLoader

import xarray as xr
import pathlib
#import HDR

def test_cyrsoxs_loader_imports_cleanly():
	cyloader = cyrsoxsLoader()

def test_cyrsoxs_single_scan_import():
        load = cyrsoxsLoader()

        raw = load.loadDirectory(pathlib.Path('2021-09-03_pl-0.1_cyl10_core5_sp60'))
       
        assert type(raw) == xr.DataArray

        return raw
