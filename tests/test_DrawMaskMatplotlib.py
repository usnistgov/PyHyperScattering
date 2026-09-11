import numpy as np
import pytest

pytest.importorskip("ipywidgets")

from PyHyperScattering.IntegrationUtils import DrawMaskMatplotlib
from PyHyperScattering.PFGeneralIntegrator import PFGeneralIntegrator


def _square_polygon():
    # a 5x5 square from (2,2) to (7,7) in (x, y), matching the fixture used in
    # test_PFGeneralIntegrator_mask_parsing.py
    return np.array([[2, 2], [7, 2], [7, 7], [2, 7]])


def test_mask_property_marks_polygon_interior():
    drawer = DrawMaskMatplotlib(np.zeros((10, 10)))
    drawer.polygons.append(_square_polygon())

    mask = drawer.mask

    assert mask.shape == (10, 10)
    assert mask.dtype == bool
    assert mask.any()
    assert mask[4, 4]  # inside the square
    assert not mask[0, 0]  # outside the square


def test_save_and_load_round_trips_polygons(tmp_path):
    drawer = DrawMaskMatplotlib(np.zeros((10, 10)))
    drawer.polygons.append(_square_polygon())

    mask_file = tmp_path / "mask.json"
    drawer.save(mask_file)

    reloaded = DrawMaskMatplotlib(np.zeros((10, 10)))
    reloaded.load(mask_file)

    assert len(reloaded.polygons) == 1
    np.testing.assert_allclose(reloaded.polygons[0], drawer.polygons[0])
    np.testing.assert_array_equal(reloaded.mask, drawer.mask)


def test_saved_mask_file_loads_in_pfgeneralintegrator(tmp_path):
    drawer = DrawMaskMatplotlib(np.zeros((10, 10)))
    drawer.polygons.append(_square_polygon())

    mask_file = tmp_path / "mask.json"
    drawer.save(mask_file)

    integrator = PFGeneralIntegrator(maskmethod="none", geomethod="none")
    integrator.loadPyHyperMask(maskpath=mask_file, maskshape=(10, 10))

    assert integrator.mask.shape == (10, 10)
    assert integrator.mask.dtype == bool
    # PFGeneralIntegrator.mask uses the opposite convention from DrawMaskMatplotlib.mask:
    # True = valid/unmasked pixel, vs. True = masked out. Boundary pixels can differ by one
    # pixel between skimage.draw.polygon2mask (used by PFGeneralIntegrator) and
    # matplotlib.path.Path.contains_points (used by DrawMaskMatplotlib), so only compare
    # points clearly inside/outside the polygon rather than exact edge-pixel agreement.
    assert not integrator.mask[4, 4]  # inside the square: masked out -> not valid
    assert integrator.mask[0, 0]  # outside the square: not masked -> valid
