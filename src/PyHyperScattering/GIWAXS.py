"""
File to:
    1. Use pygix to apply the missing wedge Ewald's sphere correction & convert to q-space
    2. Generate 2D plots of Qz vs Qxy corrected detector images
    3. Generate 2d plots of Q vs Chi images, with the option to apply the sin(chi) correction
    4. etc.
"""

# Imports
import xarray as xr
import numpy as np
import pygix  # type: ignore
import fabio # fabio package for .edf imports
import pathlib
from typing import Union, Tuple
from PyHyperScattering.IntegrationUtils import DrawMask
from tqdm.auto import tqdm 

class Transform:
    """ Class for transforming GIWAXS data into different formats. """
    def __init__(self, 
                 poniPath: Union[str, pathlib.Path], 
                 maskPath: Union[str, pathlib.Path, np.ndarray], 
                 inplane_config: str = 'q_xy', 
                 energy: float = None):
        """
        Attributes:
        poniPath (pathlib Path or str): Path to .poni file for converting to q-space 
                                        & applying missing wedge correction
        maskPath (pathlib Path or str or np.array): Path to the mask file to use 
                                for the conversion, or a numpy array
        inplane_config (str): The configuration of the inplane. Default is 'q_xy'.
        energy (optional, float): Set energy if default energy in poni file is invalid
        """

        self.poniPath = pathlib.Path(poniPath)
        try:
            self.maskPath = pathlib.Path(maskPath)
        except TypeError:
            self.maskPath = maskPath
            
        self.inplane_config = inplane_config
        if energy:
            self.energy = energy
            self.wavelength = np.round((4.1357e-15*2.99792458e8)/(energy*1000), 13)
        else:
            self.energy = None
            self.wavelength = None

    def load_mask(self, da):
        """Load the mask file based on its file type."""

        if isinstance(self.maskPath, np.ndarray):
            return self.maskPath

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
        if self.wavelength:
            pg.wavelength = self.wavelength

        # Load mask
        mask = self.load_mask(da)

        # Cartesian 2D plot transformation
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

        # Polar 2D plot transformation
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

        # Preseve time dimension if it is in the dataarray, for stacking purposes
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
        for time in tqdm(da.time, desc='Transforming raw GIWAXS time slices'):
            da_slice = da.sel(time=float(time))
            recip_da_slice, caked_da_slice = self.pg_convert(da=da_slice)
            recip_das.append(recip_da_slice)
            caked_das.append(caked_da_slice)
            
        recip_da_series = xr.concat(recip_das, 'time')
        caked_da_series = xr.concat(caked_das, 'time')
        
        return recip_da_series, caked_da_series

def single_images_to_dataset(files, loader, transformer, savePath=None, savename=None):
    """
    Function that takes a subscriptable object of filepaths corresponding to raw GIWAXS
    beamline data, loads the raw data into an xarray DataArray, generates pygix-transformed 
    cartesian and polar DataArrays, and creates 3 corresponding xarray Datasets 
    containing a DataArray per sample. The raw dataarrays must contain the attribute 'scan_id'

    Inputs: files: indexable object containing pathlib.Path filepaths to raw GIWAXS data
            loader: custom PyHyperScattering GIWAXSLoader object, must return DataArray
            transformer: instance of Transform object defined above, takes raw 
                         dataarray and returns two processed dataarrays
            savePath: optional, required to save zarrs, choose savePath for zarr store
                      pathlib.Path or absolute path as a string
            savename: optional, required to save zarrs, string for name of zarr store 
                      to be saved inside savePath, add 'raw_', 'recip_', or 'caked_' 
                      file prefix and '.zarr' suffix

    Outputs: 3 Datasets: raw, recip (cartesian), and caked (polar)
             optionally also saved zarr stores
    """
    # Select the first element of the sorted set outside of the for loop to initialize the xr.DataSet
    DA = loader.loadSingleImage(files[0])
    recip_DA, caked_DA = transformer.pg_convert(DA)

    # Save coordinates for interpolating other dataarrays 
    recip_coords = recip_DA.coords
    caked_coords = caked_DA.coords

    # Create a DataSet, each DataArray will be named according to it's scan id
    raw_DS = DA.to_dataset(name=DA.scan_id)
    recip_DS = recip_DA.to_dataset(name=DA.scan_id)
    caked_DS = caked_DA.to_dataset(name=DA.scan_id)

    # Populate the DataSet with 
    for filepath in tqdm(files[1:], desc=f'Transforming Raw Data'):
        DA = loader.loadSingleImage(filepath)
        recip_DA, caked_DA = transformer.pg_convert(DA)
        
        recip_DA = recip_DA.interp(recip_coords)
        caked_DA = caked_DA.interp(caked_coords)    

        raw_DS[f'{DA.scan_id}'] = DA
        recip_DS[f'{DA.scan_id}'] = recip_DA    
        caked_DS[f'{DA.scan_id}'] = caked_DA

    # Save zarr stores if selected
    if savePath and savename:
        print('Saving zarrs...')
        savePath = pathlib.Path(savePath)
        raw_DS.to_zarr(savePath.joinpath(f'raw_{savename}.zarr'), mode='w')
        recip_DS.to_zarr(savePath.joinpath(f'recip_{savename}.zarr'), mode='w')
        caked_DS.to_zarr(savePath.joinpath(f'caked_{savename}.zarr'), mode='w')
        print('Saved!')
    else:
        print('No save path or no filename specified, not saving zarrs... ')

    return raw_DS, recip_DS, caked_DS


# save_as_zarr method from the Transform class, commented (07/25/23)
'''
    # # - This needs to be updated appropriately so that it can be called inline. Process(Transform) should be able
    # # to import the attribute name of a newly generated .zarr by an active Transform() object instance.
    # def save_as_zarr(self, da: xr.DataArray, base_path: Union[str, pathlib.Path], prefix: str, suffix: str, mode: str = 'w'):
    #     """
    #     Save the DataArray as a .zarr file in a specific path, with a file name constructed from a prefix and suffix.

    #     Parameters:
    #         da (xr.DataArray): The DataArray to be saved.
    #         base_path (Union[str, pathlib.Path]): The base path to save the .zarr file.
    #         prefix (str): The prefix to use for the file name.
    #         suffix (str): The suffix to use for the file name.
    #         mode (str): The mode to use when saving the file. Default is 'w'.
    #     """
    #     ds = da.to_dataset(name='DA')
    #     file_path = pathlib.Path(base_path).joinpath(f"{prefix}_{suffix}.zarr")
    #     ds.to_zarr(file_path, mode=mode)
'''

# Keith's ProcessData class, commented out temporarily (07/25/23)
'''
class ProcessData:
    def __init__(self, 
                 raw_zarr_file_path: Union[str, pathlib.Path],
            pg_transformer: Transform):
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
'''

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