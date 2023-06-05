from PyHyperScattering.FileLoader import FileLoader
from scipy.interpolate import RectBivariateSpline
import os
import xarray as xr
import pandas as pd
import numpy as np
import warnings
import os
import datetime
import fabio
try:
    import dask.array as da
    import dask
except ImportError:
    warnings.warn('Failed to import Dask, if Dask reduction desired install pyhyperscattering[performance]',stacklevel=2)

class SMIRSoXSLoader(FileLoader):
    '''
    Loader for reduced data (processed by Patryk Wasik) from NSLS-II SMI 

    This code is modified from cyrsoxsLoader.py
    '''
    
    def __init__(self, profile_time = True):
        '''
        Args:
            profile_time (bool, default True): print time/profiling data to console
        '''
        self.profile_time = profile_time
        
    def list_files(self,file_path,include_str):
        files = []
        new_files = []

        new_files = [os.path.join(file_path, file) for file in os.listdir(file_path) if include_str in file]
        files = files + new_files

        files.sort(key = lambda x: os.path.getmtime(x), reverse = False)
        files = pd.Series([os.path.split(path_to_file)[-1] for path_to_file in files])

        return files
    
    # Helper function that finds all substrings in a list of strings (data)
    def all_substr(self, data):
        iteration = 0
        substr = []
        temp = data[:]
        # Iterate until no substrings of length > 1 are left
        while (len(temp) > 0) and len(self.long_substr(temp)) > 1:
            iteration += 1
            # For the first iteration, find the longest common substring
            if iteration <= 1:
                substr += self.long_substr(data)
                # Remove the longest common substring from the temp list
                temp = [x.replace(substr[-1], '') for x in data]
            else:
                # For subsequent iterations, remove previously found substrings from the temp list
                temp = [x.replace(substr[-1], '') for x in temp]
                # Append new longest common substrings to the substr list
                substr += self.long_substr(temp)
        # return pd.Series(substr).unique().tolist()
        return substr

    # Helper function that finds the longest common substring in a list of strings (data)
    def long_substr(self, data):
        substr = []
        # Check if data has at least one string and that the first string has at least one character
        if len(data) > 0 and len(data[0]) > 0:
            # Iterate over each character in the first string
            for i in range(len(data[0])):
                # Iterate over each substring starting from the i-th character
                for j in range(len(data[0]) - i + 1):
                    # Check if the substring is longer than the current longest substring and is a substring of all strings in the data list
                    if j > len(substr) and self.is_substr(data[0][i:i + j], data):
                        # If so, append it to the substr list
                        substr.append(data[0][i:i + j])   
        # Return the substr list
        return substr

    # Helper function that checks if a string (find) is a substring of all strings in a list (data)
    def is_substr(self, find, data):
        # Return False if either the find string or the data list is empty
        if len(data) < 1 or len(find) < 1:
            return False
        # Iterate over each string in the data list
        for i in range(len(data)):
            # Check if the find string is not a substring of the i-th string
            if find not in data[i]:
                # If so, return False
                return False
        # If the find string is a substring of all strings in the data list, return True
        return True

    # Method that finds which parts of the filenames contain a given substring
    def find_substring_in_filenames(self, list_of_uniques, substring):
        str_pos_in_filenames = []
        for i, list_of_unique in enumerate(list_of_uniques):
            for item in self.all_substr(list_of_unique):
                    if substring in item:
                        str_pos_in_filenames.append(i)
        # If the substring was found in at least one filename, print the positions and return the unique positions
        if len(str_pos_in_filenames) >= 1:
            print(f'Field(s) {str_pos_in_filenames} in the filenames contain {substring}.')
            return pd.Series(str_pos_in_filenames).unique().tolist()
        else:
            # If the substring was not found in any filename, print all possible substrings that could be used instead
            print(f'No part of the file names contain {substring}. Here are all the substrings that will work. Substrings of these substrings also work:')
            list_of_strings = []
            for list_of_unique in list_of_uniques:
                list_of_strings.append(self.all_substr(list_of_unique))
            print(pd.Series(list_of_strings))
            
    def loadDirectory(self, directory, pol_strs = [], pols = [], remove_tail = '_xxxx.xxeV_qmap_Intensity.tif', remove_strs = []):
        '''
        Loads a processed SMI output directory into a Dask-backed qx/qy xarray.
        The function goes through several steps, including reading the filenames, 
        extracting the energy information from the filenames, loading the image data, 
        and interpolating the images onto a common grid. At the end, it returns an xarray DataArray object.
        Args:
            directory  (string or Path): folder which contains subfolders with data analysed by Patryk Wasik (NSLS-II SMI)
            pol_strs (list of strings): list of substrings in file name which correspond to provided X-ray beam polarizations (provided in pols)
            pols (list of values): list of X-ray beam polarization values which correspond to provided pol_strs
            remove_tail (string): string that corresponds to the length to be trimmed from the end of the file name. File name is inherited from first file in directory.
            remove_strs (list of strings): list of substrings to be removed from file name which is then pushed to sample_name and sampleid attributes returned.
        '''

        # If profiling is enabled, record the start time
        if self.profile_time:
            start = datetime.datetime.now()

        # Initialize a configuration dictionary
        config = {}

        # Identify files that contain intensity and Qx, Qy values
        files = self.list_files(directory, include_str = '_qmap_Intensity')
        Qx_files = self.list_files(directory, include_str = '_qmap_qhor')
        Qy_files = self.list_files(directory, include_str = '_qmap_qver')

        # Split the first filename by underscores to determine how many strings it contains
        file_name_strs = [ [] for _ in files[0].split('_')]

        # Parse all the files to extract strings from filenames
        for file in files:
            for i, item in enumerate(file.split('_')):
                file_name_strs[i].append(item)

        # Extract all unique energies from the filenames and convert them to floats
        all_energies = np.sort(file_name_strs[self.find_substring_in_filenames(file_name_strs, 'eV')[0]])
        energies = [float(''.join(c for c in energy if (c.isdigit() or c == '.'))) for energy in all_energies]

        # Save the unique energies into the config dictionary
        config['energy'] = pd.Series(energies).unique()
        elist = config['energy']
        num_energies = len(elist)

        # Identify Qx and Qy ranges for the file with maximum energy
        for i, e in enumerate(elist):
            if e == np.max(elist):
                max_range_Qx = np.loadtxt((f'{directory}' + Qx_files[i]), comments='#', delimiter=None, skiprows=1)
                max_range_Qy = np.loadtxt((f'{directory}' + Qy_files[i]), comments='#', delimiter=None, skiprows=1)

        # Create an empty list to hold interpolated image data
        outlist = []

        # Loop through each energy, load corresponding image and interpolate it onto a common grid
        for i, e in enumerate(elist):
            image = np.fliplr(np.flipud(np.rot90(fabio.open(f'{directory}' + files[i]).data)))
            NumX, NumY = image.shape
            Qx = np.loadtxt((f'{directory}' + Qx_files[i]), comments='#', delimiter=None, skiprows=1)
            Qy = np.loadtxt((f'{directory}' + Qy_files[i]), comments='#', delimiter=None, skiprows=1)
            qx_new = np.linspace(np.nanmin(max_range_Qx), np.nanmax(max_range_Qx), num=NumX)
            qy_new = np.linspace(np.nanmin(max_range_Qy), np.nanmax(max_range_Qy), num=NumY)
            interpolator = RectBivariateSpline(Qx, Qy, image)
            img = interpolator(qx_new, qy_new)
            outlist.append(img)

        # Stack the interpolated images into a 3D array
        data = da.stack(outlist,axis=2)

        # Extract the sample name and id from the first filename
        config['sample_name'] = files[0][:-len(remove_tail)]
        config['sampleid'] = config['sample_name']

        # Identify the polarization of the X-ray beam based on the filename
        for i, pol_str in enumerate(pol_strs):
            if pol_str in config['sample_name']:
                remove_strs.append(pol_str)
                config['polarization'] = pols[i]
            else:
                config['polarization'] = 0

        # Remove specified substrings from the sample name
        for remove_str in remove_strs:
            config['sample_name'] = config['sample_name'].replace(remove_str, '')

        # Identify the scattering configuration (WAXS or SAXS) based on Qx and Qy ranges
        if np.nanmax(Qx) > 1 or np.nanmax(Qy) > 1:
            config['rsoxs_config'] = 'waxs'
        else:
            config['rsoxs_config'] = 'saxs'

        # If profiling is enabled, print the time taken
        if self.profile_time: 
             print(f'Finished reading ' + str(num_energies) + ' energies. Time required: ' + str(datetime.datetime.now()-start))

        # Return the data as an xarray DataArray object with appropriate dimensions, coordinates and attributes
        return xr.DataArray(data, dims=("qx", "qy", "energy"), coords={"qy":max_range_Qy, "qx":max_range_Qx, "energy":elist},attrs=config).rename(config['sampleid'])