import json
import pathlib

import pandas as pd

from PyHyperScattering.PFGeneralIntegrator import PFGeneralIntegrator


def test_load_pyhyper_mask_parses_json_strings_with_pandas3(tmp_path):
    polygon_df = pd.DataFrame(
        {
            "x": [2, 7, 7, 2],
            "y": [2, 2, 7, 7],
        }
    )
    mask_file = tmp_path / "mask.json"
    mask_file.write_text(json.dumps([polygon_df.to_json()]))

    integrator = PFGeneralIntegrator(maskmethod="none", geomethod="none")
    integrator.loadPyHyperMask(maskpath=pathlib.Path(mask_file), maskshape=(10, 10))

    assert integrator.mask.shape == (10, 10)
    assert integrator.mask.dtype == bool
    assert integrator.mask.any()
