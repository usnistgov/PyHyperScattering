"""
File that will contain functions to:
    1. Use pygix to apply the missing wedge Ewald's sphere correciton & convert to q-space
    2. Generate 2D plots of Qz vs Qxy corrected detector images
    3. Generate 2d plots of Q vs Chi images, with the option to apply the sin(chi) correction
    4. etc.
    
"""

import xarray as xr
import numpy as np
import pygix
from PyHyperScattering.IntegrationUtils import DrawMask

def pg_convert(da, poniPath, maskPath, inplane_config='q_xy'):
    """
    Converts raw GIWAXS detector image to q-space data. Returns two DataArrays, Qz vs Qxy & Q vs Chi
    
    Inputs: Raw GIWAXS DataArray
            Path to .poni file for converting to q-space & applying missing wedge correction
    Outputs: Cartesian & Polar DataArrays
    """

    # Initialize pygix transform object
    pg = pygix.Transform()
    pg.load(str(poniPath))
    pg.sample_orientation = 3
    pg.incident_angle = float(da.incident_angle[2:])

    # Load PyHyper-drawn mask
    draw = DrawMask(da)
    draw.load(maskPath)
    mask = draw.mask

    recip_data, qxy, qz = pg.transform_reciprocal(da.data,
                                                  method='bbox',
                                                  unit='A',
                                                  mask=np.flipud(mask),
                                                  correctSolidAngle=True)
    recip_data = np.reshape(recip_data, (recip_data.shape[0], recip_data.shape[1], 1))
    
    recip_da = xr.DataArray(data=recip_data,
                            dims=['q_z', inplane_config, 'time'],
                            coords={
                                'q_z': qz,
                                inplane_config: qxy,
                                'time': np.array([float(da.time)])
                            },
                            attrs=da.attrs)

    caked_data, qr, chi = pg.transform_image(da.data, 
                                             process='polar',
                                             method = 'bbox',
                                             unit='q_A^-1',
                                             mask=np.flipud(mask),
                                             correctSolidAngle=True)
    caked_data = np.reshape(caked_data, (caked_data.shape[0], caked_data.shape[1], 1))

    caked_da = xr.DataArray(data=caked_data,
                        dims=['chi', 'qr', 'time'],
                        coords={
                            'chi': chi,
                            'qr': qr,
                            'time': np.array([float(da.time)])
                        },
                        attrs=da.attrs)
    caked_da.attrs['inplane_config'] = inplane_config
    
    return recip_da, caked_da
    
# def pg_convert_series(da, poniPath, maskPath, inplane_config='q_xy'):
