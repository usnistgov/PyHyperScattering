from PIL import Image
from PyHyperScattering.FileLoader import FileLoader
import os, pathlib, datetime, warnings, json, corrections
import xarray as xr
import pandas as pd
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
        a pathlib object variable for each sample's data folder (which contains the /maxs/raw/ subfolders),
        and time_start and exposure_time for each series of scans.
        """

        # Ensure the root_folder is a pathlib.Path object
        self.root_folder = pathlib.Path(root_folder)
        if not self.root_folder.is_dir():
            raise ValueError(f"Directory {self.root_folder} does not exist.")
        
        # Initialize the sample dictionary
        sample_dict = {}
        
        # Alias mappings for scan_id, series_number, time_start, and exposure_time
        scan_id_aliases = ['scanid', 'id', 'scannum', 'scan', 'scan_id', 'scan_ID']
        series_number_aliases = ['seriesnum', 'seriesid', 'series_id', 'series_ID', 'series', 'series_number', 'series_num']
        time_start_aliases = ['start_time', 'starttime', 'start', 'clocktime', 'clock', 'clockpos', 'clock_time', 'time', 'time_start']
        exposure_time_aliases = ['exptime', 'exp_time', 'exposuretime', 'etime', 'exp', 'expt', 'exposure_time']

        # Identify the indices of the required metadata in the naming scheme
        for idx, alias in enumerate(self.md_naming_scheme):
            if alias.lower() in [alias.lower() for alias in scan_id_aliases]:
                self.scan_id_index = idx
            if alias.lower() in [alias.lower() for alias in series_number_aliases]:
                self.series_number_index = idx

        if self.scan_id_index is None or self.series_number_index is None:
            raise ValueError('md_naming_scheme does not contain keys for scan_id or series_number.')

        # Update sample_dict with new information
        for sample_folder in self.root_folder.iterdir():
            if sample_folder.is_dir():
                # Confirm that this is a sample folder by checking for /maxs/raw/ subfolder
                maxs_raw_dir = sample_folder / 'maxs' / 'raw'
                if maxs_raw_dir.is_dir():
                    # Sample folder checks out, extract scan_id, series_number, time_start, and exposure_time
                    sample_name = sample_folder.name
                    scan_list = []
                    series_list = collections.defaultdict(list)
                    
                    for image_file in maxs_raw_dir.glob('*'):
                        # Load metadata from image
                        metadata = self.loadMd(image_file)
                        
                        # Lowercase all metadata keys for case insensitivity
                        metadata_lower = {k.lower(): v for k, v in metadata.items()}
                        
                        # Find and store scan_id, series_number, time_start, and exposure_time
                        scan_id = metadata_lower.get(self.md_naming_scheme[self.scan_id_index].lower())
                        series_number = metadata_lower.get(self.md_naming_scheme[self.series_number_index].lower())
                        time_start = next((metadata_lower[key] for key in metadata_lower if key in time_start_aliases), None)
                        exposure_time = next((metadata_lower[key] for key in metadata_lower if key in exposure_time_aliases), None)

                        # Add them to our lists
                        scan_list.append(scan_id)
                        series_list[scan_id].append((series_number, time_start, exposure_time))
                    
                    # Store data in dictionary
                    sample_dict[sample_name] = {
                        'scanlist': scan_list,
                        'serieslist': series_list,
                        'path': sample_folder
                    }

        self.sample_dict = sample_dict
        return sample_dict

    def selectSampleAndSeries(self):
        """
        Prompts the user to select a sample and one or more series of scans from that sample.
        The user can choose to select all series of scans.
        The selections will be stored as the 'selected_series' attribute.
        """
        # Show the user a list of sample names and get their selection
        print("Please select a sample:")
        sample_names = list(self.sample_dict.keys())
        for i, sample_name in enumerate(sample_names, 1):
            print(f"[{i}] {sample_name}")
        sample_index = int(input("Enter the number of your choice: ")) - 1
        selected_sample = sample_names[sample_index]

        # Show the user a choice between single image or image series and get their selection
        print("\nWould you like to choose a single image or an image series?")
        print("[1] Single Image")
        print("[2] Image Series")
        choice = int(input("Enter the number of your choice: "))
        
        # Get the selected sample's scan list and series list
        scan_list = self.sample_dict[selected_sample]['scanlist']
        series_list = self.sample_dict[selected_sample]['serieslist']

        # Identify series scan IDs and single image scan IDs
        series_scan_ids = {series[1] for series in series_list}
        single_image_scan_ids = [scan_id for scan_id in scan_list if scan_id not in series_scan_ids]

        if choice == 1:
            # The user has chosen to select a single image
            print("\nPlease select a scan ID:")
            for i, scan_id in enumerate(single_image_scan_ids, 1):
                print(f"[{i}] {scan_id}")
            scan_id_index = int(input("Enter the number of your choice: ")) - 1
            self.selected_series = [(selected_sample, single_image_scan_ids[scan_id_index])]
        else:
            # The user has chosen to select an image series
            print("\nPlease select one or more series (Enter 'a' to select all series, 'q' to finish selection):")
            selected_series = []
            while True:
                for i, series in enumerate(series_list, 1):
                    print(f"[{i}] Series {series[1]} (start time: {series[2]}, exposure time: {series[3]})")
                print("[a] All series")
                print("[q] Finish selection")
                selection = input("Enter the number(s) of your choice (comma-separated), 'a', or 'q': ")
                if selection.lower() == 'q':
                    break
                elif selection.lower() == 'a':
                    selected_series = series_list
                    break
                else:
                    # Get the series indices from the user's input
                    series_indices = list(map(int, selection.split(',')))
                    selected_series += [series_list[i-1] for i in series_indices]
            self.selected_series = [(selected_sample, series[1]) for series in selected_series]
        
        print("\nSelection completed.")

# -- originally pushed createSampleDictionary() method: (07/18/2023)
"""    def createSampleDictionary(self, root_folder):
        
        # Loads and creates a sample dictionary from a root folder path.
        # The dictionary will contain: sample name, scanID list, series scanID list, 
        # and a pathlib object variable for each sample's data folder (which contains the /maxs/raw/ subfolders).
        

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
"""