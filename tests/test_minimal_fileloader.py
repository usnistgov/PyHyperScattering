import sys, os, tempfile, pathlib
sys.path.append('src/')

import numpy as np
import xarray as xr
from PyHyperScattering.FileLoader import FileLoader

class MinimalLoader(FileLoader):
    file_ext = r'.*\.txt$'
    md_loading_is_quick = True

    def peekAtMd(self, filepath):
        idx = int(pathlib.Path(filepath).stem.split('_')[-1])
        return {"index": idx}

    def loadSingleImage(self, filepath, coords=None, return_q=False, image_slice=None, **kwargs):
        qx = np.array([0.0, 1.0, 2.0])
        qy = np.array([0.0, 1.0])
        attrs = {"index": int(pathlib.Path(filepath).stem.split('_')[-1])}
        if coords:
            attrs.update(coords)
        data = np.zeros((len(qy), len(qx)))
        if return_q:
            return xr.DataArray(data, dims=["qy", "qx"], coords={"qx": qx, "qy": qy}, attrs=attrs)
        else:
            return xr.DataArray(data, dims=["pix_y", "pix_x"], attrs=attrs)

def test_loadfile_series_qxy():
    loader = MinimalLoader()
    with tempfile.TemporaryDirectory() as tmp:
        # create dummy files
        for i in range(3):
            open(os.path.join(tmp, f"image_{i}.txt"), "w").close()
        result = loader.loadFileSeries(tmp, ["index"], output_qxy=True)
    assert "qx" in result.coords and "qy" in result.coords
    assert len(result.qx) == 3
    assert len(result.qy) == 2
    assert result.attrs.get("dims_unpacked") == ["index"]
