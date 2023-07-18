from PIL import Image
from PyHyperScattering.FileLoader import FileLoader
import os
import pathlib
import xarray as xr
import pandas as pd
import datetime
import warnings
import json
#from pyFAI import azimuthalIntegrator
import numpy as np
from tqdm.auto import tqdm 


class CMSGIWAXSLoader(FileLoader):
    """
    GIXS Data Loader Class | NSLS-II 11-BM (CMS)
    Used to load single TIFF time-series TIFF GIWAXS images.
    """
    def __init__(self, md_naming_scheme=[], root_folder=None):
        self.md_naming_scheme = md_naming_scheme
        self.root_folder = root_folder
        self.sample_dict = None

    def loadSingleImage(self, filepath):
        """
        Loads a single xarray DataArray from a filepath to a raw TIFF
        """

        # Check that the path exists before continuing.
        if not pathlib.Path(filepath).is_file():
            raise ValueError(f"File {filepath} does not exist.")
        
        # Open the image from the filepath
        image = Image.open(filepath)

        # Create a numpy array from the image
        image_data = np.array(image)

        # Run the loadMetaData method to construct the attribute dictionary for the filePath.
        attr_dict = self.loadMd(filepath)

        # Convert the image numpy array into an xarray DataArray object.
        image_da = xr.DataArray(data = image_data, 
                                dims=['pix_y', 'pix_x'],
                                attrs=attr_dict)
        
        image_da = image_da.assign_coords({
            'pix_x': image_da.pix_x.data,
            'pix_y': image_da.pix_y.data
        })
        return image_da
    
    def loadMd(self, filepath, delim = '_'):
        """
        Description: Uses metadata_keylist to generate attribute dictionary of metadata based on filename.
        Handle Variables
            filepath : string
                Filepath passed to the loadMetaData method that is used to extract metadata relevant to the TIFF image.
            delim : string
                String used as a delimiter in the filename. Defaults to an underscore '_' if no other delimiter is passed.
        
        Method Variables
            attr_dict : dictionary
                Attributes ictionary of metadata attributes created using the filename and metadata list passed during initialization.
            md_list : list
                Metadata list - list of metadata keys used to segment the filename into a dictionary corresponding to said keys.
        """

        attr_dict = {} # Initialize the dictionary.
        name = filepath.name # # strip the filename from the filePath
        md_list = name.split(delim) # splits the filename based on the delimter passed to the loadMetaData method.

        for i, md_item in enumerate(self.md_naming_scheme):
            attr_dict[md_item] = md_list[i]
        return attr_dict
    
    def loadSeries(self, files, filter='', time_start=0):
        """
        Load many raw TIFFs into an xarray DataArray

        Input: files: Either a pathlib.Path object that can be filtered with a 
                      glob filter or an iterable that contains the filepaths
        Output: xr.DataArray with appropriate dimensions & coordinates
        """

        data_rows = []
        if issubclass(type(files), pathlib.Path):
            for filepath in tqdm(files.glob(f'*{filter}*')):
                image_da = self.loadSingleImage(filepath)
                image_da = image_da.assign_coords({'series_number': int(image_da.series_number)})
                image_da = image_da.expand_dims(dim={'series_number': 1})
                data_rows.append(image_da)
        else:
            try:
                for filepath in tqdm(files):
                    image_da = self.loadSingleImage(filepath)
                    image_da = image_da.assign_coords({'series_number': int(image_da.series_number)})
                    image_da = image_da.expand_dims(dim={'series_number': 1})
                    data_rows.append(image_da)  
            except TypeError:
                warnings.warn('"files" needs to be a pathlib.Path or iterable')  
                return None      

        out = xr.concat(data_rows, 'series_number')
        out = out.sortby('series_number')
        out = out.assign_coords({
            'series_number': out.series_number.data,
            'time': ('series_number', 
                     out.series_number.data*np.round(float(out.exposure_time[:-1]),
                                                     1)+np.round(float(out.exposure_time[:-1]),1)+time_start)
        })
        out = out.swap_dims({'series_number': 'time'})
        out = out.sortby('time')
        del out.attrs['series_number']

        return out

    def createSampleDictionary(self, root_folder):
        """
        Loads and creates a sample dictionary from a root folder path.
        The dictionary will contain: sample name, scanID list, series scanID list, 
        and a pathlib object variable for each sample's data folder (which contains the /maxs/raw/ subfolders).
        """

        # Ensure the root_folder is a pathlib.Path object
        self.root_folder = pathlib.Path(self.root_folder)
        if not self.root_folder.is_dir():
            raise ValueError(f"Directory {self.root_folder} does not exist.")
        
        # Initialize the sample dictionary
        sample_dict = {}

        # Find the index of 'scan_id' and 'series_number' in the md_naming_scheme list
        scan_id_index = None
        series_number_index = None
        scan_id_aliases = ['scanID', 'ID', 'scannum', 'scan', 'SCAN', 'Scan', 'scanid', 'id', 'ScanNum', 'scan_id', 'scan_ID']
        series_number_aliases = ['seriesnum', 'seriesid', 'series_id', 'series_ID', 'series', 'SERIES', 'Series', 'series_number', 'series_num']

        for index, name in enumerate(self.md_naming_scheme):
            if name.lower() in [alias.lower() for alias in scan_id_aliases]:
                scan_id_index = index
            elif name.lower() in [alias.lower() for alias in series_number_aliases]:
                series_number_index = index

        if scan_id_index is None or series_number_index is None:
            raise ValueError('md_naming_scheme does not contain keys for scan_id or series_number.')
        
        # Iterate through all subdirectories in the root folder
        for subdir in root_folder.iterdir():
            if subdir.is_dir():
                # Check if /maxs/raw/ subdirectory exists
                maxs_raw_dir = subdir / "maxs" / "raw"
                if maxs_raw_dir.is_dir():
                    # The name of the subdirectory is considered as the sample name
                    sample_name = subdir.name
                    sample_dict[sample_name] = {
                        "scanlist": [],
                        "serieslist": {},
                        "sample_path": subdir
                    }
                    
                    # Iterate through the files in the /maxs/raw/ subdirectory
                    for filename in maxs_raw_dir.glob('*'):
                        metadata = filename.stem.split('_')
                        scan_id = metadata[scan_id_index]
                        series_number = metadata[series_number_index]

                        # Update serieslist
                        if scan_id in sample_dict[sample_name]["serieslist"]:
                            sample_dict[sample_name]["serieslist"][scan_id] += 1
                        else:
                            sample_dict[sample_name]["serieslist"][scan_id] = 1

                        # Append the series_number to the scanlist in the dictionary
                        if series_number not in sample_dict[sample_name]["scanlist"]:
                            sample_dict[sample_name]["scanlist"].append(series_number)

        self.sample_dict = sample_dict
        return sample_dict