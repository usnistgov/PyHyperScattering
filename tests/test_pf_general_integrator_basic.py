import xarray as xr
import numpy as np
from PyHyperScattering.PFGeneralIntegrator import PFGeneralIntegrator


def test_single_image_dataset():
    npts = 123
    img = xr.DataArray(np.ones((10, 10)), dims=["pix_y", "pix_x"])
    integ = PFGeneralIntegrator(
        maskmethod="none",
        geomethod="none",
        do_1d_integration=True,
        return_sigma=True,
        npts=npts,
        energy=1000,
    )
    integ.calibrationFromNikaParams(
        distance=1000,
        bcx=5,
        bcy=5,
        tiltx=0,
        tilty=0,
        pixsizex=100,
        pixsizey=100,
    )
    result = integ.integrateSingleImage(img)
    assert isinstance(result, xr.Dataset)
    assert set(result.data_vars) == {"I", "dI"}
    assert len(result.coords["q"]) == npts
