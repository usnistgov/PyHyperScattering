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
import h5py

import re

class ESRFID2Loader(FileLoader):
    '''
    Loader for NEXUS files from the ID2 beamline at the ESRF 

    '''
    file_ext = '(.*)eiger2(.*)azim.h5'
    md_loading_is_quick = True
    
    def __init__(self,md_parse_dict=None):
        if md_parse_dict is None:
            self.md_regex = None
            self.md_keys=None
        else:
            regex_list = []
            md_keys = []
            for key,value in md_parse_dict.items():
                md_keys.append(key)
                regex_list.append(value)
            self.md_regex = re.compile('_'.join(regex_list))
            self.md_keys = md_keys

    def loadMd(self,filepath,split_on='_',keys=None):
        return self.peekAtMd(filepath,split_on='_')

    def peekAtMd(self,filepath,split_on='_',keys=None):
        ## Open h5 file and grab attributes
        with h5py.File(str(filepath),'r') as h5:
            title = h5['entry_0000/title'][()].decode('utf8')
            parameters = h5['entry_0000/PyFAI/parameters']
            
            PyFAI_params = {}
            for key,value in parameters.items():
                try: #decode bytestringes if bytestring
                    value = value[()].decode('utf8')
                except AttributeError: #dont decode if not bytestring
                    value = value[()]
                PyFAI_params[key] = value
                
        ## Parse Title String
        if self.md_regex is None:
            title_params = title.split(split_on)
        else:
            result = self.md_regex.findall(title)
            if not result or (len(result)>1):
                raise ValueError(f'Regex parser failed!\ntitle={title}\nregex={self.md_regex.pattern}')
            title_params = result[0]
            
        ## Add keys to title string
        if self.md_keys is None: #use numerical keys
            title_params = {str(i):v for i,v in enumerate(title_params)}
        else:
            assert len(self.md_keys)==len(title_params), 'Number of provided keys and found params don"t match. Check keys and split_on'
            title_params = {k:v for k,v in zip(self.md_keys,title_params)}
            
        ## Construct final params array
        params = {}
        params.update(title_params)
        params.update(PyFAI_params)
        return params
            
        
    def loadSingleImage(self,filepath,coords=None,return_q=True):
        '''
        HELPER FUNCTION that loads a single image and returns an xarray with either pix_x / pix_y dimensions (if return_q == False) or qx / qy (if return_q == True)


        Args:
            filepath (Pathlib.path): path of the file to load
            coords (dict-like): coordinate values to inject into the metadata
            return_q (bool): return qx / qy coords.  If false, returns pixel coords.

        '''
        headerdict = self.loadMd(filepath)
        if coords is not None:
            headerdict.update(coords)
        with h5py.File(filepath,'r') as h5:
            title = h5['entry_0000/title'][()].decode('utf8')
            start_time = h5['entry_0000/start_time'][()].decode('utf8')
            
            default_path = h5['entry_0000'].attrs['default']
            default_group = h5[default_path]
            
            signal = default_group.attrs['signal']
            data = default_group[signal][()]
            
            try:
                axes = default_group.attrs['axes']
            except KeyError:
                axes=None
                coords=None
            else:
                axes = axes[1:]#throw out first entry (always '.')
                coords = {}
                for ax in axes:
                    coords[ax] = default_group[ax][()]
                    # headerdict[ax] = default_group[ax][()]
            
            da = xr.DataArray(
                data=data.squeeze(),
                dims=axes,
                coords=coords,
                name=title,
                attrs=headerdict
            )
            da = da.where(da>-10)# need to change negative default values to NaN
        return da
                                         
