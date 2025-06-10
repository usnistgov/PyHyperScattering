import sys
sys.path.append("src/")

import numpy as np
import xarray as xr
from PyHyperScattering import Fitting

q = np.linspace(0.005, 0.007, 50)
AMPLITUDE = 5.0
CENTER = 0.006
WIDTH = 0.0002
BACKGROUND = 0.1

lorentz_da = xr.DataArray(
    Fitting.lorentz(q, AMPLITUDE, CENTER, WIDTH),
    coords={"q": q},
    dims=["q"],
)
lorentz_bg_da = xr.DataArray(
    Fitting.lorentz_w_flat_bg(q, AMPLITUDE, CENTER, WIDTH, BACKGROUND),
    coords={"q": q},
    dims=["q"],
)

def test_fit_lorentz():
    res = Fitting.fit_lorentz(
        lorentz_da, guess=[AMPLITUDE * 0.9, CENTER * 1.01, WIDTH * 1.1], silent=True
    )
    assert np.isclose(res.intensity.mean(), AMPLITUDE, rtol=1e-5)
    assert np.isclose(res.pos.mean(), CENTER, rtol=1e-5)
    assert np.isclose(res.width.mean(), WIDTH, rtol=1e-5)

def test_fit_lorentz_bg():
    res = Fitting.fit_lorentz_bg(
        lorentz_bg_da,
        guess=[AMPLITUDE * 0.9, CENTER * 1.01, WIDTH * 1.1, BACKGROUND * 0.8],
        silent=True,
    )
    assert np.isclose(res.intensity.mean(), AMPLITUDE, rtol=1e-5)
    assert np.isclose(res.pos.mean(), CENTER, rtol=1e-5)
    assert np.isclose(res.width.mean(), WIDTH, rtol=1e-5)
    assert np.isclose(res.bg.mean(), BACKGROUND, rtol=1e-5)

def test_fitting_apply_with_stacked_dimension():
    stacked = xr.concat([lorentz_bg_da, lorentz_bg_da], dim="replicate").assign_coords(
        replicate=["r1", "r2"]
    )
    result = stacked.fit.apply(
        Fitting.fit_lorentz_bg,
        guess=[AMPLITUDE * 0.9, CENTER * 1.01, WIDTH * 1.1, BACKGROUND * 0.8],
        silent=True,
    )
    assert list(result.dims) == ["replicate"]
    assert list(result.coords["replicate"].values) == ["r1", "r2"]

