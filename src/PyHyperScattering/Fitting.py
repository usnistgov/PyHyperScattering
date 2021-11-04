import scipy.optimize
import xarray as xr
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
#tqdm.pandas()

# the following block monkey-patches xarray to add tqdm support.  This will not be needed once tqdm v5 releases.
from xarray.core.groupby import DataArrayGroupBy,DatasetGroupBy

def inner_generator(df_function='apply'):
    def inner(df,func,*args,**kwargs):
        t = tqdm(total=len(df))
        def wrapper(*args,**kwargs):
            t.update( n=1 if not t.total or t.n < t.total else 0)
            return func(*args,**kwargs)
        result = getattr(df,df_function)(wrapper, **kwargs)
    
        t.close()
        return result
    return inner

DataArrayGroupBy.progress_apply = inner_generator()
DatasetGroupBy.progress_apply = inner_generator()

DataArrayGroupBy.progress_map = inner_generator(df_function='map')
DatasetGroupBy.progress_map = inner_generator(df_function='map')
#end monkey patch

@xr.register_dataset_accessor('fit')
@xr.register_dataarray_accessor('fit')
class Fitting:
    def __init__(self,xr_obj):
        self._obj=xr_obj
    def apply(self,fit_func,fit_axis = 'q',**kwargs):
        '''
        Apply a fit function to this PyHyperScattering dataset.
        
        This is intended to smooth over some of the gory xarray details of fitting.
        
        Args:
            fit_func (callable): a function that takes any arguments passed as kwargs and returns an xarray Dataset or DataArray in the same coordinate space with the fit results.  See examples in Fitting.py.
            fit_axis (str, default 'q'): the "special axis" along which fits should be applied, i.e, you wish to fit in intensity vs fit_axis space.
            
            kwargs (anything): passed through to fit_func
            
        
        Example:
            data.fit.apply(PyHyperScattering.Fitting.fit_lorentz_bg,silent=True)
        '''
        df = self._obj    
        for name,idx in df.indexes.items():
            if type(idx)==pd.core.indexes.multi.MultiIndex:
                df = df.unstack(name)

        dims_to_stack = []

        for name in df.indexes.keys():
            if name != fit_axis:
                dims_to_stack.append(name)

        df = df.stack(temp_fit_axis = dims_to_stack)
        df = df.groupby('temp_fit_axis')
        df = df.progress_map(fit_func,**kwargs)
        df = df.unstack('temp_fit_axis')
        df = df.mean('q')
        return df

def fit_lorentz(x,guess=None,pos_int_override=False,silent=False):
    '''
    Fit a lorentzian, constructed as a lambda function compatible with xarray.groupby([...]).apply().

    Uses the scipy.optimize.curve_fit NLS solver.
        
    Args:
        x: input xarray groupby entry
        guess (list): [intensity, q, width] tuple to start NLS fitting from.
                        If guess is none, pos_int_override will be set to True, and width will be 0.0002.
        pos_int_override (bool): if True, overrides the peak center as the median q-value of the array, and intensity as the intensity at that q.
    '''
    # example guess: [500.,0.00665,0.0002] [int, q, width]
    x = x.dropna('q')
    if guess == None:
        guess = [500.,0.00665,0.0002]
        pos_int_override=True
    if pos_int_override:
        guess[1] = np.median(x.coords['q'])
        guess[0] = x.sel(q=guess[1],method='nearest')
    if not silent: 
        print(f"Starting fit on {x.coords}")
    try:
        coeff, var_matrix = scipy.optimize.curve_fit(lorentz,x.coords['q'].data,x.data,p0=guess)
    except RuntimeError:
        if not silent:
            print("Fit failed to converge")
        retval = xr.DataArray(data=np.nan,coords=x.coords).to_dataset(name='intensity')
        retval['pos'] = xr.DataArray(data=np.nan,coords=x.coords)
        retval['width'] = xr.DataArray(data=np.nan,coords=x.coords)
        return retval
    if not silent:
        print(f"Fit completed, coeff = {coeff}")
    retval = xr.DataArray(data=coeff[0],coords=x.coords).to_dataset(name='intensity')
    retval['pos'] = xr.DataArray(data=coeff[1],coords=x.coords)
    retval['width'] = xr.DataArray(data=coeff[2],coords=x.coords)
    return retval
def fit_lorentz_bg(x,guess=None,pos_int_override=False,silent=False):
    '''
    Fit a lorentzian, constructed as a lambda function compatible with xarray.groupby([...]).apply().

    Uses the scipy.optimize.curve_fit NLS solver.
        
    Args:
        x: input xarray groupby entry
        guess (list): [intensity, q, width, background] tuple to start NLS fitting from.
                        If guess is none, pos_int_override will be set to True, bg will be zero, and width will be 0.0002.
        pos_int_override (bool): if True, overrides the peak center as the median q-value of the array, and intensity as the intensity at that q.
    '''
    # example guess: [500.,0.00665,0.0002,0] [int, q, width, bg]
    x = x.dropna('q')
    if guess == None:
        guess = [500.,0.00665,0.0002,0]
        pos_int_override=True
    if pos_int_override:
        guess[1] = np.median(x.coords['q'])
        guess[0] = x.sel(q=guess[1],method='nearest')
    if not silent: 
        print(f"Starting fit on {x.coords}")
    try:        
        coeff, var_matrix = scipy.optimize.curve_fit(lorentz_w_flat_bg,x.coords['q'].data,x.data,p0=guess)
    except RuntimeError:
        if not silent:
            print("Fit failed to converge")
        retval = xr.DataArray(data=np.nan,coords=x.coords).to_dataset(name='intensity')
        retval['pos'] = xr.DataArray(data=np.nan,coords=x.coords)
        retval['width'] = xr.DataArray(data=np.nan,coords=x.coords)
        retval['bg'] = xr.DataArray(data=np.nan,coords=x.coords)
        return retval
    if not silent:
        print(f"Fit completed, coeff = {coeff}")
    retval = xr.DataArray(data=coeff[0],coords=x.coords).to_dataset(name='intensity')
    retval['pos'] = xr.DataArray(data=coeff[1],coords=x.coords)
    retval['width'] = xr.DataArray(data=coeff[2],coords=x.coords)
    retval['bg'] = xr.DataArray(data=coeff[3],coords=x.coords)
    return retval

def dummy_fit(x,guess=None,pos_int_override=False):
    '''
    For testing purposes, a fit function constructed as a lambda function compatible with xarray.groupby([...]).apply(), but which just returns zero.
    '''
    return xr.DataArray(data=0,coords=x.coords)

def gauss(x, *p):
    '''
    Helper function - Gaussian peak

    Args:
        x (numeric): the input
        p (list): parameters [A, mu, sigma] where the intensity will be: A*numpy.exp(-(x-mu)**2/(2.*sigma**2))
    '''
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))
def lorentz( x, *p ):
    '''
    Helper function - Lorentzian peak

    Args:
        x (numeric): the input
        p (list): parameters [A, x0, gamma] where the intensity will be: a * gam**2 / ( gam**2 + ( x - x0 )**2)
    '''
    a, x0, gam = p
    return a * gam**2 / ( gam**2 + ( x - x0 )**2)
def lorentz_w_flat_bg( x, *p ):
    '''
    Helper function - Lorentzian peak with flat background

    Args:
        x (numeric): the input
        p (list): parameters [A, x0, gamma, bg] where the intensity will be: (a * gam**2 / ( gam**2 + ( x - x0 )**2)) + bg
    '''
    a, x0, gam, bg = p
    return bg+(a * gam**2 / ( gam**2 + ( x - x0 )**2))