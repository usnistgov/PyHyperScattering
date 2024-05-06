import pathlib
import warnings
import fabio
from PIL import Image
from PyHyperScattering.FileLoader import FileLoader
import xarray as xr
import pandas as pd
import numpy as np
from tqdm.auto import tqdm 

class CMSGIWAXSLoader(FileLoader):
    """
    GIXS Data Loader Class | NSLS-II 11-BM (CMS)
    Used to load single TIFF time-series TIFF GIWAXS images.
    """
    def __init__(self, md_naming_scheme=[], root_folder=None, delim='_'):
        """
        Inputs: md_naming_scheme: A list of attribute names to load from 
                filename, which is split by the delimeter argument (default is 
                '_'). If no list is provided, the filename will be loaded an 
                an attribute. 

                root_folder: not implemented yet, to use with future helper functions

                delim: delimeter value to split filename (default is underscore)
        """

        self.md_naming_scheme = md_naming_scheme
        if len(md_naming_scheme) == 0:
            warnings.warn('Provided an empty md_naming_scheme. This will just load the filename as an attribute.',stacklevel=2)
        self.root_folder = root_folder
        self.delim = delim
        self.sample_dict = None
        self.selected_series = []
        
    def loadSingleImage(self, filepath,coords=None,return_q=False,image_slice=None,use_cached_md=False,**kwargs):
        """
        Loads a single xarray DataArray from a filepath to a raw TIFF

        Inputs: 
        - filepath: str or pathlib.Path to image to load

        Unused inputs (for loadFileSeries compatability): 
        - coords 
        - return_q 
        - image_slice 
        - use_cached_md

        """

        # Ensure that the path exists before continuing.
        filepath = pathlib.Path(filepath)
        
        # Open the image from the filepath
        # Get the file extension using pathlib
        file_extension = filepath.suffix.lower()

        # Choose the appropriate method based on the file extension
        if file_extension in ['.tiff', '.tif']:
            # Load using PIL for tiffs, just to avoid a warning message
            image_data = np.array(Image.open(filepath))
        else:
            # Default for Load using Fabio
            image_data = fabio.open(filepath).data

        # Run the loadMetaData method to construct the attribute dictionary for the filePath.
        attr_dict = self.loadMd(filepath)

        # Convert the image numpy array into an xarray DataArray object.
        image_da = xr.DataArray(data = image_data, 
                                dims = ['pix_y', 'pix_x'],
                                attrs = attr_dict)
        
        image_da = image_da.assign_coords({
            'pix_x': image_da.pix_x.data,
            'pix_y': image_da.pix_y.data
        })
        return image_da
    
    def loadMd(self, filepath):
        """
        Description: Uses metadata_keylist to generate attribute dictionary of metadata based on filename.
        Input Variables
            filepath : string
                Filepath passed to the loadMetaData method that is used to extract metadata relevant to the TIFF image.
            
        
        """
        delim = self.delim
        attr_dict = {} # Attributes dictionary of metadata attributes created using the filename and metadata list passed during initialization.
        name = filepath.name # # strip the filename from the filePath
        md_list = name.split(delim) # splits the filename based on the delimiter passed to the loadMd method.
                                                    # Metadata list - list of metadata keys used to segment the filename into a dictionary corresponding to said keys.

        if len(self.md_naming_scheme) == 0:
            attr_dict['filename'] = name
        elif len(self.md_naming_scheme) >= 1:
            for i, md_item in enumerate(self.md_naming_scheme):
                attr_dict[md_item] = md_list[i]
        return attr_dict
    
    def loadSeries(self, files, filter='', time_start=0):
        """
        LEGACY METHOD: PyHyperScattering.load.loadFileSeries() will be the updated method moving forward
        
        Load many raw TIFFs into an xarray DataArray

        Input: files: Either a pathlib.Path object that can be filtered with a 
                      glob filter or an iterable that contains the filepaths
        Output: xr.DataArray with appropriate dimensions & coordinates
        """

        data_rows = []
        if issubclass(type(files), pathlib.Path):
            for filepath in tqdm(files.glob(f'*{filter}*'), desc='Loading raw GIWAXS time slices'):
                image_da = self.loadSingleImage(filepath)
                image_da = image_da.assign_coords({'series_number': int(image_da.series_number)})
                image_da = image_da.expand_dims(dim={'series_number': 1})
                data_rows.append(image_da)
        else:
            try:
                for filepath in tqdm(files, desc='Loading raw GIWAXS time slices'):
                    image_da = self.loadSingleImage(filepath)
                    image_da = image_da.assign_coords({'series_number': int(image_da.series_number)})
                    image_da = image_da.expand_dims(dim={'series_number': 1})
                    data_rows.append(image_da)  
            except TypeError as e:
                raise TypeError('"files" needs to be a pathlib.Path or iterable')  from e   

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
        NOT FULLY IMPLEMENTED YET
        
        Loads and creates a sample dictionary from a root folder path.
        The dictionary will contain: sample name, scanID list, series scanID list, 
        a pathlib object variable for each sample's data folder (which contains the /maxs/raw/ subfolders),
        and time_start and exposure_time for each series of scans.
        
        The method uses alias mappings to identify important metadata from the filenames:
        SCAN ID : Defines the scan ID number in the convention used at 11-BM (CMS), specific to a single shot exposure or time series.
            aliases : scan_id: 'scanid', 'id', 'scannum', 'scan', 'scan_id', 'scan_ID'
        SERIES NUMBER : Within a series (fixed SCAN ID), the exposure number in the series with respect to the starting TIME START (clocktime)
            aliases : series_number: 'seriesnum', 'seriesid', 'series_id', 'series_ID', 'series', 'series_number', 'series_num'
        TIME START : Also generically referred to as CLOCK TIME, logs the start of the exposure or series acquisition. This time is constant for all exposures within a series.
            aliases : time_start: 'start_time', 'starttime', 'start', 'clocktime', 'clock', 'clockpos', 'clock_time', 'time', 'time_start'
        EXPOSURE TIME : The duration of a single shot or exposure, either in a single image or within a series.
            aliases : 'exptime', 'exp_time', 'exposuretime', 'etime', 'exp', 'expt', 'exposure_time'
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
                    series_list = {}  # Initialize series_list as an empty dictionary
                    
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
                        
                        # Check if scan_id is in series_list, if not, create a new list
                        if scan_id not in series_list:
                            series_list[scan_id] = []

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
            NOT FULLY IMPLEMENTED YET

            Prompts the user to select a sample and one or more series of scans from that sample.
            The user can choose to select all series of scans.
            The selections will be stored as the 'selected_series' attribute and returned.
            """
            # Check if sample_dict has been generated
            if not self.sample_dict:
                print("Error: Sample dictionary has not been generated. Please run createSampleDictionary() first.")
                return

            while True:
                # Show the user a list of sample names and get their selection
                print("Please select a sample (or 'q' to exit):")
                sample_names = list(self.sample_dict.keys())
                for i, sample_name in enumerate(sample_names, 1):
                    print(f"[{i}] {sample_name}")
                print("[q] Exit")
                selection = input("Enter the number of your choice: ")
                if selection.lower() == 'q':
                    print("Exiting selection.")
                    return self.selected_series
                else:
                    sample_index = int(selection) - 1
                    selected_sample = sample_names[sample_index]

                # Show the user a choice between single image or image series and get their selection
                print("\nWould you like to choose a single image or an image series? (or 'q' to exit)")
                print("[1] Single Image")
                print("[2] Image Series")
                print("[q] Exit")
                choice = input("Enter the number of your choice: ")
                if choice.lower() == 'q':
                    print("Exiting selection.")
                    return self.selected_series
                choice = int(choice)

                # Get the selected sample's scan list and series list
                scan_list = self.sample_dict[selected_sample]['scanlist']
                series_list = self.sample_dict[selected_sample]['serieslist']

                # Identify series scan IDs and single image scan IDs
                series_scan_ids = set(series_list.keys())
                single_image_scan_ids = [scan_id for scan_id in scan_list if scan_id not in series_scan_ids]

                if choice == 1:
                    # The user has chosen to select a single image
                    print("\nPlease select a scan ID (or 'q' to exit):")
                    for i, scan_id in enumerate(single_image_scan_ids, 1):
                        print(f"[{i}] {scan_id}")
                    print("[q] Exit")
                    selection = input("Enter the number of your choice: ")
                    if selection.lower() == 'q':
                        print("Exiting selection.")
                        return self.selected_series
                    else:
                        scan_id_index = int(selection) - 1
                        selected_scan = single_image_scan_ids[scan_id_index]
                        self.selected_series.append((selected_sample, selected_scan))
                else:
                    # The user has chosen to select an image series
                    print("\nPlease select one or more series (Enter 'a' to select all series, 'q' to finish selection):")
                    selected_series = []
                    while True:
                        for i, series_scan_id in enumerate(series_scan_ids, 1):
                            series_data = series_list[series_scan_id]
                            print(f"[{i}] Series {series_scan_id} (start time: {series_data[0][1]}, exposure time: {series_data[0][2]})")
                        print("[a] All series")
                        print("[q] Finish selection")
                        selection = input("Enter the number(s) of your choice (comma-separated), 'a', or 'q': ")
                        if selection.lower() == 'q':
                            if selected_series:
                                break
                            else:
                                print("Exiting selection.")
                                return self.selected_series
                        elif selection.lower() == 'a':
                            selected_series = list(series_scan_ids)
                            break
                        else:
                            # Get the series indices from the user's input
                            series_indices = list(map(int, selection.split(',')))
                            selected_series += [list(series_scan_ids)[i-1] for i in series_indices]
                    self.selected_series.extend([(selected_sample, series) for series in selected_series])

                print("\nSelection completed.")
            return self.selected_series    
