import numpy as np
import xarray as xr
import pytest
from PyHyperScattering import RSoXS

@pytest.fixture()
def simple_dataset():
    chi = np.arange(-180, 180)
    q = np.arange(5)
    pols = [0, 90]
    data = np.zeros((2, len(chi), len(q)))
    for i, p in enumerate(pols):
        data[i] = 10 + 5 * np.cos(np.deg2rad(chi - p))[:, None]
    return xr.DataArray(data, dims=("polarization", "chi", "q"),
                        coords={"polarization": pols, "chi": chi, "q": q})

def test_AR_matches_manual(simple_dataset):
    ds = simple_dataset
    ar_auto = ds.rsoxs.AR(calc2d=True)
    para = ds.sel(polarization=0)
    perp = ds.sel(polarization=90)
    para_para = para.rsoxs.slice_chi(0)
    para_perp = para.rsoxs.slice_chi(-90)
    perp_perp = perp.rsoxs.slice_chi(-90)
    perp_para = perp.rsoxs.slice_chi(0)
    ar_para = (para_para - para_perp) / (para_para + para_perp)
    ar_perp = (perp_perp - perp_para) / (perp_perp + perp_para)
    ar_manual = (ar_para + ar_perp) / 2
    assert np.allclose(ar_auto, ar_manual)

def test_AR_warning_on_offset():
    chi = np.arange(-180, 180)
    q = np.arange(5)
    pols = [0, 90]
    data = np.zeros((2, len(chi), len(q)))
    base = 10 + 5 * np.cos(np.deg2rad(chi))
    data[0] = base[:, None]
    data[1] = base[:, None] + 1
    ds = xr.DataArray(data, dims=("polarization", "chi", "q"),
                      coords={"polarization": pols, "chi": chi, "q": q})
    with pytest.warns(UserWarning):
        ds.rsoxs.AR(calc2d=True)
