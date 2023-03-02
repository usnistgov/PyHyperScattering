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
    
    # Functions that finds common substrings in list/array of strings
    def all_substr(self, data):
        iteration = 0
        substr = []
        temp = data[:]
        while (len(temp) > 0) and len(self.long_substr(temp)) > 1:
            iteration += 1
            if iteration <= 1:
                substr += self.long_substr(data)
                temp = [x.replace(substr[-1], '') for x in data]
            else:
                temp = [x.replace(substr[-1], '') for x in temp]
                substr += self.long_substr(temp)
        # return pd.Series(substr).unique().tolist()
        return substr

    def long_substr(self, data):
        substr = []
        if len(data) > 0 and len(data[0]) > 0:
            for i in range(len(data[0])):
                for j in range(len(data[0]) - i + 1):
                    if j >= len(substr) and self.is_substr(data[0][i:i + j], data):
                        substr.append(data[0][i:i + j])   
        return substr

    def is_substr(self, find, data):
        if len(data) < 1 and len(find) < 1:
            return False
        for i in range(len(data)):
            if find not in data[i]:
                return False
        return True
    # end of functions that finds common substrings in list/array of strings

    def find_substring_in_filenames(self, list_of_uniques, substring):
        str_pos_in_filenames = []
        for i, list_of_unique in enumerate(list_of_uniques):
            for item in self.all_substr(list_of_unique):
                    if substring in item:
                        str_pos_in_filenames.append(i)
        if len(str_pos_in_filenames) >= 1:
            print(f'Field(s) {str_pos_in_filenames} in the filenames contain {substring}.')
            return pd.Series(str_pos_in_filenames).unique().tolist()
        else:
            print(f'No part of the file names contain {substring}. Here are all the substrings that will work. Substrings of these substrings also work:')
            list_of_strings = []
            for list_of_unique in list_of_uniques:
                list_of_strings.append(self.all_substr(list_of_unique))
            print(pd.Series(list_of_strings))
            
    def loadDirectory(self, directory, pol_strs = [], pols = [], remove_tail = '_xxxx.xxeV_qmap_Intensity.tif', remove_strs = []):
        '''
        Loads a processed SMI output directory into a Dask-backed qx/qy xarray.
        
        Args:
            directory  (string or Path): folder which contains subfolders with data analysed by Patryk Wasik (NSLS-II SMI)
            pol_strs (list of strings): list of substrings in file name which correspond to provided X-ray beam polarizations (provided in pols)
            pols (list of values): list of X-ray beam polarization values which correspond to provided pol_strs
            remove_tail (string): string that corresponds to the length to be trimmed from the end of the file name. File name is inherited from first file in directory.
            remove_strs (list of strings): list of substrings to be removed from file name which is then pushed to sample_name and sampleid attributes returned.
        '''
        if self.profile_time:
            start = datetime.datetime.now()
            
        config = {}

        files = self.list_files(directory, include_str = '_qmap_Intensity')
        Qx_files = self.list_files(directory, include_str = '_qmap_qhor')
        Qy_files = self.list_files(directory, include_str = '_qmap_qver')

        # Make a list of an arbitrary number of lists (determined by the number of strings in the file name separated by '_')
        file_name_strs = []
        del file_name_strs
        file_name_strs = []

        for item in files[0].split('_'):
            file_name_strs.append([])

        # Go through all the files and append entries from different filenames into the lists generated above
        # assumes that the info in each string separated by '_' is consistent in placement in the file name for all files
        for file in files:
            for i, item in enumerate(file.split('_')):
                file_name_strs[i].append(item)

        #define all the all_energies available from the files loaded: 
        all_energies = np.sort(file_name_strs[self.find_substring_in_filenames(file_name_strs, 'eV')[0]])

        energies = []
        for i, energy in enumerate(all_energies):
            energies.append(float(''.join(c for c in energy if (c.isdigit() or c == '.'))))

        config['energy'] = pd.Series(energies).unique()

        elist = config['energy']
        num_energies = len(elist)
        
        max_range_Qx = None
        max_range_Qy = None

        for i, e in enumerate(elist):
            if e == np.max(elist):
                max_range_Qx = np.loadtxt((f'{directory}' + Qx_files[i]), comments='#', delimiter=None, skiprows=1)
                max_range_Qy = np.loadtxt((f'{directory}' + Qy_files[i]), comments='#', delimiter=None, skiprows=1)

        outlist = []
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
        data = da.stack(outlist,axis=2)

        config['sample_name'] = files[0][:-len(remove_tail)]
        for remove_str in remove_strs:
            config['sample_name'] = config['sample_name'].replace(remove_str, '')

        config['sampleid'] = config['sample_name']

        for i, pol_str in enumerate(pol_strs):
            if pol_str in config['sample_name']:
                config['polarization'] = pols[i]
            else:
                config['polarization'] = 0

        if np.nanmax(Qx) > 1 or np.nanmax(Qy) > 1:
            config['rsoxs_config'] = 'waxs'
        else:
            config['rsoxs_config'] = 'saxs'
        
        if self.profile_time: 
             print(f'Finished reading ' + str(num_energies) + ' energies. Time required: ' + str(datetime.datetime.now()-start))

        return xr.DataArray(data, dims=("qx", "qy", "energy"), coords={"qy":max_range_Qy, "qx":max_range_Qx, "energy":elist},attrs=config).rename(config['sample_name'])
