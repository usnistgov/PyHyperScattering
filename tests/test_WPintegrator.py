import sys,os
sys.path.append("src/")

from PyHyperScattering.load import cyrsoxsLoader
from PyHyperScattering.integrate import WPIntegrator

import xarray as xr
import pathlib
#import HDR

def test_wp_integrator_imports_cleanly():
	integ = WPIntegrator()


def test_cyrsoxs_single_scan_import():
        load = cyrsoxsLoader()
        integ = WPIntegrator()

        raw = load.loadDirectory(pathlib.Path('2021-09-03_pl-0.1_cyl10_core5_sp60'))
        reduced = integ.integrateImageStack(raw)

        assert type(reduced)==xr.DataArray

        return raw
