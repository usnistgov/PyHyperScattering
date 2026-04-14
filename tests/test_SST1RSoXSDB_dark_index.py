import numpy as np
import pytest

from PyHyperScattering.SST1RSoXSDB import _scalarize_dark_index


def test_scalarize_dark_index_accepts_length_one_numpy_arrays():
    assert _scalarize_dark_index(np.array([3])) == 3


def test_scalarize_dark_index_accepts_numpy_scalars():
    assert _scalarize_dark_index(np.array(4)) == 4


def test_scalarize_dark_index_accepts_uniform_length_many_arrays():
    assert _scalarize_dark_index(np.array([5, 5, 5])) == 5


def test_scalarize_dark_index_rejects_nonuniform_arrays():
    with pytest.raises(ValueError, match="not uniform"):
        _scalarize_dark_index(np.array([1, 2]))
