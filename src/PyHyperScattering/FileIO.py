import warnings
import xarray as xr
import numpy as np
import pickle
import math

@xr.register_dataset_accessor('fileio')
@xr.register_dataarray_accessor('fileio')
class FileIO:
    def __init__(self,xr_obj):
        self._obj=xr_obj
        
        self._pyhyper_type = 'reduced'
        try:
            self._chi_min = np.min(xr_obj.chi)
            self._chi_max = np.max(xr_obj.chi)
            self._chi_range = [self._chi_min,self._chi_max]
        except AttributeError:
            self._pyhyper_type = 'raw'
        
    def savePickle(self,filename):
        with open(filename, 'wb') as file:
            pickle.dump(object, file)
    
    def loadPickle(self,filename):
        return pickle.load( open( filename, "rb" ) )