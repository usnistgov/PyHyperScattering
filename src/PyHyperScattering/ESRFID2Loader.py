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
import copy

import re

class ESRFID2Loader(FileLoader):
    '''
    Loader for NEXUS files from the ID2 beamline at the ESRF 

    '''
    file_ext = '(.*)eiger2(.*).h5'
    md_loading_is_quick = True
    
    def __init__(self,md_parse_dict=None,pedestal_value=1e-6,masked_pixel_fill=np.nan):
        '''
            Args:
                md_parse_dict (dict): keys should be names of underscore separated paramters in title. values should be regex to parse values
                pedestal_value: value to add to image in order to deal with zero_counts
                masked_pixel_fill: If None, pixels with value -10 will be converted to NaN. Otherwise, will be converted to this value

        '''
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
        self.pedestal_value=pedestal_value
        self.masked_pixel_fill = masked_pixel_fill
        self.cached_md = None
        

    def loadMd(self,filepath,split_on='_',keys=None):
        return self.peekAtMd(filepath,split_on='_')

    def peekAtMd(self,filepath,split_on='_',keys=None):
        ## Open h5 file and grab attributes
        with h5py.File(str(filepath),'r') as h5:
            title = h5['entry_0000/PyFAI/parameters/Title'][()].decode('utf8')
            parameters = h5['entry_0000/PyFAI/parameters']
            
            transmission = h5['entry_0000/PyFAI/MCS/Intensity1ShutCor'][()]/h5['entry_0000/PyFAI/MCS/Intensity0ShutCor'][()]
            transmission_mean = np.mean(transmission)
            
            PyFAI_params = {}
            for key,value in parameters.items():
                value = value[()]
                try: #decode bytestringes if bytestring
                    value = value.decode('utf8')
                except AttributeError: #dont decode if not bytestring
                    pass
                try:#try to convert strings to floats
                    value = float(value)
                except ValueError:# bail on conversion
                    pass
                PyFAI_params[key] = value
                
        ## Parse Title String
        if self.md_regex is None:
            title_params = title.split(split_on)
        else:
            result = self.md_regex.findall(title)
            if not result or (len(result)>1):
                # peter ignored - should this actually be a warning, not an error? was ValueError
                warnings.warn(f'Regex parser failed!\ntitle={title}\nregex={self.md_regex.pattern}',stacklevel=2)
                title_params=[]
            else:
                title_params = result[0]
            
        ## Add keys to title string
        if self.md_keys is None: #use numerical keys
            title_params = {str(i):v for i,v in enumerate(title_params)}
        else:
            assert len(self.md_keys)==len(title_params), 'Number of provided keys and found params don"t match. Check keys and split_on'
            #warnings.warn(f'Regex parser failed!\ntitle={title}\nregex={self.md_regex.pattern}',stacklevel=2)
            title_params = {k:v for k,v in zip(self.md_keys,title_params)}
        
        for key,value in title_params.items():
            try:#try to convert strings to floats
                value = float(value)
            except ValueError:# bail on conversion
                pass
            title_params[key] = value            
        ## Construct final params array
        params = {'trans':transmission,'trans_mean':transmission_mean}
        params.update(title_params)
        params.update(PyFAI_params)
        self.cached_md = params
        return params
            
        
    def loadSingleImage(self,filepath,coords=None,return_q=True,image_slice=None,use_cached_md=False,**kwargs):
        '''
        HELPER FUNCTION that loads a single image and returns an xarray with either pix_x / pix_y dimensions (if return_q == False) or qx / qy (if return_q == True)


        Args:
            filepath (Pathlib.path): path of the file to load
            coords (dict-like): coordinate values to inject into the metadata
            return_q (bool): return qx / qy coords.  If false, returns pixel coords.

        '''
        if len(kwargs.keys())>0:
            warnings.warn(f'Loader does not support features for args: {kwargs.keys()}',stacklevel=2)
        if use_cached_md and (self.cached_md is not None):
            headerdict = copy.deepcopy(self.cached_md)
        else:
            headerdict = self.loadMd(filepath)
            
        if coords is not None:
            headerdict.update(coords)
            
        if image_slice is None:
            image_slice = ()
            
        with h5py.File(filepath,'r') as h5:           
            default_path = h5['entry_0000'].attrs['default']
            default_group = h5[default_path]
            
            signal = default_group.attrs['signal']
            if image_slice is None:
                data = default_group[signal][()]
            else:
                data = default_group[signal][tuple(image_slice)]

            
            try:
                axes = default_group.attrs['axes']
            except KeyError:
                axes=None
                coords=None
            else:
                axes[0] = 't'
                #axes = axes:[1:]#throw out first entry (always '.')
                coords = {}
                for i,ax in enumerate(axes):
                    try:
                        sl = image_slice[i]
                    except IndexError:
                        sl = ()
                    coords[ax] = default_group[ax][sl]
                    # headerdict[ax] = default_group[ax][()]
            
        img = xr.DataArray(
            data=data,
            dims=axes,
            coords=coords,
            name=headerdict['Title'],
            attrs=headerdict
        )

        img = img.where(img>-10,other=self.masked_pixel_fill)# need to change negative default values to NaN

        #apply pedestal
        img += self.pedestal_value

        return img
                                         
