import xarray as xr
import numpy as np

@xr.register_dataset_accessor('pt')
@xr.register_dataarray_accessor('pt')
class PlotTools():
    def __init__(self,xr_obj):
        self._obj=xr_obj
        
        self._pyhyper_type = 'reduced'
        try:
            self._chi_min = np.min(xr_obj.chi)
            self._chi_max = np.max(xr_obj.chi)
            self._chi_range = [self._chi_min,self._chi_max]
        except AttributeError:
            self._pyhyper_type = 'raw'
    
    def plot_loglog(self):
        '''
        
        '''
        return self._obj.plot(xscale='log',yscale='log')