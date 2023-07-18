"""
File that will contain functions to:
    1. Use pygix to apply the missing wedge Ewald's sphere correction & convert to q-space
    2. Generate 2D plots of Qz vs Qxy corrected detector images
    3. Generate 2d plots of Q vs Chi images, with the option to apply the sin(chi) correction
    4. etc.
"""

import xarray as xr
import numpy as np
import pygix  # type: ignore
import fabio # fabio package for .edf imports
import pathlib
from typing import Union, Tuple
from PyHyperScattering.IntegrationUtils import DrawMask
from tqdm.auto import tqdm 

class Transform:
    """
    Class for transforming GIWAXS data into different formats.
    
    Attributes:
        poniPath (str): Path to .poni file for converting to q-space & applying missing wedge correction
        maskPath (str): Path to the mask file to use for the conversion
        inplane_config (str): The configuration of the inplane. Default is 'q_xy'.
    """

    def __init__(self, poniPath, maskPath, inplane_config='q_xy'):
        self.poniPath = poniPath
        self.maskPath = maskPath
        self.inplane_config = inplane_config

    def load_mask(self, da):
        """Load the mask file based on its file type."""
        try:
            if self.maskPath.suffix == '.json':
                draw = DrawMask(da)  
                draw.load(self.maskPath)
                return draw.mask
            elif self.maskPath.suffix == '.edf':
                return fabio.open(self.maskPath).data
            else:
                raise ValueError(f"Unsupported file type: {self.maskPath.suffix}")
        except Exception as e:
            print(f"An error occurred while loading the mask file: {e}")

    def add_time_coord(self, da):
        """Add the 'time' coordinate if it exists in the original data array."""
        if 'time' in da.coords:
            da = da.assign_coords({'time': float(da.time)})
            da = da.expand_dims(dim={'time': 1})
        return da

    def pg_convert(self, da):
        """
        Converts raw GIWAXS detector image to q-space data. Returns two DataArrays, Qz vs Qxy & Q vs Chi
        
        Inputs: Raw GIWAXS DataArray
        Outputs: Cartesian & Polar DataArrays
        """

        # Initialize pygix transform object
        pg = pygix.Transform()
        pg.load(str(self.poniPath))
        pg.sample_orientation = 3
        pg.incident_angle = float(da.incident_angle[2:])

        # load mask
        mask = self.load_mask(da)

        recip_data, qxy, qz = pg.transform_reciprocal(da.data,
                                                      method='bbox',
                                                      unit='A',
                                                      mask=mask,
                                                      correctSolidAngle=True)
        
        recip_da = xr.DataArray(data=recip_data,
                                dims=['q_z', self.inplane_config],
                                coords={
                                    'q_z': ('q_z', qz, {'units': '1/Å'}),
                                    self.inplane_config: (self.inplane_config, qxy, {'units': '1/Å'})
                                },
                                attrs=da.attrs)

        caked_data, qr, chi = pg.transform_image(da.data, 
                                                 process='polar',
                                                 method = 'bbox',
                                                 unit='q_A^-1',
                                                 mask=mask,
                                                 correctSolidAngle=True)

        caked_da = xr.DataArray(data=caked_data,
                            dims=['chi', 'qr'],
                            coords={
                                'chi': ('chi', chi, {'units': '°'}),
                                'qr': ('qr', qr, {'units': '1/Å'})
                            },
                            attrs=da.attrs)
        caked_da.attrs['inplane_config'] = self.inplane_config

        recip_da = self.add_time_coord(recip_da)
        caked_da = self.add_time_coord(caked_da)

        return recip_da, caked_da

    def pg_convert_series(self, da):
        """
        Converts raw GIWAXS DataArray to q-space and returns Cartesian & Polar DataArrays

        Inputs: Raw GIWAXS DataArray with a time dimension
        Outputs: 2 DataArrays in q-space with dimensions (q_z, inplane_config (default is q_xy), time) and (chi, qr, time)
        """
        recip_das = []
        caked_das = []
        for time in tqdm(da.time):
            da_slice = da.sel(time=float(time))
            recip_da_slice, caked_da_slice = self.pg_convert(da=da_slice)
            recip_das.append(recip_da_slice)
            caked_das.append(caked_da_slice)
            
        recip_da_series = xr.concat(recip_das, 'time')
        caked_da_series = xr.concat(caked_das, 'time')
        
        return recip_da_series, caked_da_series

    # - This needs to be updated appropriately so that it can be called inline. Process(Transform) should be able
    # to import the attribute name of a newly generated .zarr by an active Transform() object instance.
    def save_as_zarr(self, da: xr.DataArray, base_path: Union[str, pathlib.Path], prefix: str, suffix: str, mode: str = 'w'):
        """
        Save the DataArray as a .zarr file in a specific path, with a file name constructed from a prefix and suffix.

        Parameters:
            da (xr.DataArray): The DataArray to be saved.
            base_path (Union[str, pathlib.Path]): The base path to save the .zarr file.
            prefix (str): The prefix to use for the file name.
            suffix (str): The suffix to use for the file name.
            mode (str): The mode to use when saving the file. Default is 'w'.
        """
        ds = da.to_dataset(name='DA')
        file_path = pathlib.Path(base_path).joinpath(f"{prefix}_{suffix}.zarr")
        ds.to_zarr(file_path, mode=mode)

class ProcessData:
    def __init__(self, raw_zarr_file_path: Union[str, pathlib.Path], pg_transformer: Transform):
        """
        Constructor of the ProcessData class.

        Parameters:
            raw_zarr_file_path (Union[str, pathlib.Path]): The path to the raw .zarr file.
            pg_transformer (Transform): An instance of the Transform class for performing the conversion.
        """
        self.raw_zarr_file_path = pathlib.Path(raw_zarr_file_path)
        self.pg_transformer = pg_transformer
        self.raw_da = self.load_zarr(self.raw_zarr_file_path)

        self.recip_zarr_file_path, self.caked_zarr_file_path = self.convert_raw_zarr_to_recip_and_caked()
        self.recip_da = self.load_zarr(self.recip_zarr_file_path)
        self.caked_da = self.load_zarr(self.caked_zarr_file_path)

    def load_zarr(self, file_path: Union[str, pathlib.Path]):
        """
        Load a .zarr file as an xarray DataArray.

        Parameters:
            file_path (Union[str, pathlib.Path]): The path to the .zarr file.

        Returns:
            xr.DataArray: The loaded xarray DataArray.
        """
        return xr.open_zarr(str(file_path)).DA

    def convert_raw_zarr_to_recip_and_caked(self):
        """
        Convert the raw DataArray to reciprocal and caked DataArrays, and save them as .zarr files.
        The paths to the new .zarr files are returned.

        Returns:
            Tuple[pathlib.Path, pathlib.Path]: The paths to the .zarr files for the reciprocal and caked DataArrays.
        """
        recip_da, caked_da = self.pg_transformer.pg_convert_series(self.raw_da)

        recip_zarr_file_path = self.raw_zarr_file_path.with_name(f"recip_{self.raw_zarr_file_path.stem}.zarr")
        recip_da.to_dataset(name='DA').to_zarr(recip_zarr_file_path, mode='w')

        caked_zarr_file_path = self.raw_zarr_file_path.with_name(f"caked_{self.raw_zarr_file_path.stem}.zarr")
        caked_da.to_dataset(name='DA').to_zarr(caked_zarr_file_path, mode='w')

        return recip_zarr_file_path, caked_zarr_file_path

    def azimuthal_integration(self, recip_da_series: xr.DataArray, caked_da_series: xr.DataArray, dim: Union[str, Tuple[str]]):
        """
        Perform 1D azimuthal integration using boxcuts.

        Parameters:
            caked_da_series (xr.DataArray): The caked DataArray series.
            dim (Union[str, Tuple[str]]): The dimension(s) over which to integrate.

        Returns:
            Tuple[xr.DataArray, xr.DataArray]: The integrated reciprocated and caked DataArray series.
        """
        caked_integrated = caked_da_series.integrate(dim)

        return caked_integrated

# -- old Tranform class, refactored: (07/17/23)
'''
class Transform:

    def __init__(self, poniPath, maskPath, inplane_config='q_xy'):
        self.poniPath = poniPath
        self.maskPath = maskPath
        self.inplane_config = inplane_config

    def pg_convert(self, da):
        """
        Converts raw GIWAXS detector image to q-space data. Returns two DataArrays, Qz vs Qxy & Q vs Chi
        
        Inputs: Raw GIWAXS DataArray
                Path to .poni file for converting to q-space & applying missing wedge correction
        Outputs: Cartesian & Polar DataArrays
        """

        # Initialize pygix transform object
        pg = pygix.Transform()
        pg.load(str(self.poniPath))
        pg.sample_orientation = 3
        pg.incident_angle = float(da.incident_angle[2:])

        # print (self.maskPath.suffix)
        if self.maskPath.suffix == '.json':
            # Load PyHyper-drawn mask
            draw = DrawMask(da)  # Assuming 'da' is defined somewhere in your code
            draw.load(self.maskPath)
            mask = draw.mask
        elif self.maskPath.suffix == '.edf':
            # Load EDF file using MNE
            # mask = mne.io.read_raw_edf(self.maskPath)
            mask = fabio.open(self.maskPath).data # load mask file
        else:
            print(f"Unsupported file type: {self.maskPath.suffix}")

        recip_data, qxy, qz = pg.transform_reciprocal(da.data,
                                                      method='bbox',
                                                      unit='A',
                                                      mask=mask,
                                                      correctSolidAngle=True)
        
        recip_da = xr.DataArray(data=recip_data,
                                dims=['q_z', self.inplane_config],
                                coords={
                                    'q_z': ('q_z', qz, {'units': '1/Å'}),
                                    self.inplane_config: (self.inplane_config, qxy, {'units': '1/Å'})
                                },
                                attrs=da.attrs)

        caked_data, qr, chi = pg.transform_image(da.data, 
                                                 process='polar',
                                                 method = 'bbox',
                                                 unit='q_A^-1',
                                                 mask=mask,
                                                 correctSolidAngle=True)

        caked_da = xr.DataArray(data=caked_data,
                            dims=['chi', 'qr'],
                            coords={
                                'chi': ('chi', chi, {'units': '°'}),
                                'qr': ('qr', qr, {'units': '1/Å'})
                            },
                            attrs=da.attrs)
        caked_da.attrs['inplane_config'] = self.inplane_config

        if 'time' in da.coords:
            recip_da = recip_da.assign_coords({'time': float(da.time)})
            recip_da = recip_da.expand_dims(dim={'time': 1})
            caked_da = caked_da.assign_coords({'time': float(da.time)})
            caked_da = caked_da.expand_dims(dim={'time': 1})
        
        return recip_da, caked_da
        
    def pg_convert_series(self, da):
        """
        Converts raw GIWAXS DataArray to q-space and returns Cartesian & Polar DataArrays

        Inputs: Raw GIWAXS DataArray with a time dimension
        Outputs: 2 DataArrays in q-space with dimensions (q_z, inplane_config (default is q_xy), time) and (chi, qr, time)
        """
        recip_das = []
        caked_das = []
        for time in tqdm(da.time):
            da_slice = da.sel(time=float(time))
            recip_da_slice, caked_da_slice = self.pg_convert(da=da_slice)
            recip_das.append(recip_da_slice)
            caked_das.append(caked_da_slice)
            
        recip_da_series = xr.concat(recip_das, 'time')
        caked_da_series = xr.concat(caked_das, 'time')
        
        return recip_da_series, caked_da_series
'''

# -- old pg_convert script (07/17/23)
'''
# def pg_convert(da, poniPath, maskPath, inplane_config='q_xy'):
#     """
#     Converts raw GIWAXS detector image to q-space data. Returns two DataArrays, Qz vs Qxy & Q vs Chi
    
#     Inputs: Raw GIWAXS DataArray
#             Path to .poni file for converting to q-space & applying missing wedge correction
#     Outputs: Cartesian & Polar DataArrays
#     """

#     # Initialize pygix transform object
#     pg = pygix.Transform()
#     pg.load(str(poniPath))
#     pg.sample_orientation = 3
#     pg.incident_angle = float(da.incident_angle[2:])

#     # Load PyHyper-drawn mask
#     draw = DrawMask(da)
#     draw.load(maskPath)
#     mask = draw.mask

#     recip_data, qxy, qz = pg.transform_reciprocal(da.data,
#                                                   method='bbox',
#                                                   unit='A',
#                                                   mask=mask,
#                                                   correctSolidAngle=True)
    
#     recip_da = xr.DataArray(data=recip_data,
#                             dims=['q_z', inplane_config],
#                             coords={
#                                 'q_z': ('q_z', qz, {'units': '1/Å'}),
#                                 inplane_config: (inplane_config, qxy, {'units': '1/Å'})
#                             },
#                             attrs=da.attrs)

#     caked_data, qr, chi = pg.transform_image(da.data, 
#                                              process='polar',
#                                              method = 'bbox',
#                                              unit='q_A^-1',
#                                              mask=mask,
#                                              correctSolidAngle=True)

#     caked_da = xr.DataArray(data=caked_data,
#                         dims=['chi', 'qr'],
#                         coords={
#                             'chi': ('chi', chi, {'units': '°'}),
#                             'qr': ('qr', qr, {'units': '1/Å'})
#                         },
#                         attrs=da.attrs)
#     caked_da.attrs['inplane_config'] = inplane_config

#     if 'time' in da.coords:
#         recip_da = recip_da.assign_coords({'time': float(da.time)})
#         recip_da = recip_da.expand_dims(dim={'time': 1})
#         caked_da = caked_da.assign_coords({'time': float(da.time)})
#         caked_da = caked_da.expand_dims(dim={'time': 1})
    
#     return recip_da, caked_da
    
# def pg_convert_series(da, poniPath, maskPath, inplane_config='q_xy'):
#     """
#     Converts raw GIWAXS DataArray to q-space and returns Cartesian & Polar DataArrays

#     Inputs: Raw GIWAXS DataArray with a time dimension
#     Outputs: 2 DataArrays in q-space with dimensions (q_z, inplane_config (default is q_xy), time) and (chi, qr, time)
#     """
#     recip_das = []
#     caked_das = []
#     for time in tqdm(da.time):
#         da_slice = da.sel(time=float(time))
#         recip_da_slice, caked_da_slice = pg_convert(da=da_slice, 
#                                                     poniPath=poniPath,
#                                                     maskPath=maskPath,
#                                                     inplane_config=inplane_config)
#         recip_das.append(recip_da_slice)
#         caked_das.append(caked_da_slice)
        
#     recip_da_series = xr.concat(recip_das, 'time')
#     caked_da_series = xr.concat(caked_das, 'time')
    
#     return recip_da_series, caked_da_series
'''