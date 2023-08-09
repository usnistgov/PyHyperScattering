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
    
def fit_cos_anisotropy(data,qL,qU,qspacing,Enlist,ChiL,ChiU,binnumber,Chilim):
    '''
    Will determine ansiotropy as a function of the q slicing and spacing given
    data= Xarray input with dimensions of Chi, q, Energy, Polarization
    qL, qU = upper and lower bounds for the q range
    qspacing = distance between q points 
    Enlist = list of energies that will be fit
    ChiL, ChiU = allows you to cut out any problematic points around the mask
    binnumber = gives you the option of binning the data in chi - 0 keeps it as is
    Chilim = an upper limit for the goodness of fit beyond which all results will be set to 0, implies that the data does not reflect a cos function for whatever reason
    '''
    
    #generates arrays to store the final results
    qarray=np.arange(qL,qU,qspacing)
    anisotropy=np.zeros([len(Enlist),len(qarray)])
    chisq=np.zeros([len(Enlist),len(qarray)])
    anisotropyU=np.zeros([len(Enlist),len(qarray)])
    
    for Ecount,En in enumerate(Enlist):
        for i in range(len(qarray)):
            
            qmin=qarray[i]-qspacing/2
            qmax=qarray[i]+qspacing/2
            r=data.sel(energy=En,q=slice(qmin,qmax)).mean('q')
            
            if binnumber >0:
                rc=r.coarsen(chi=binnumber,boundary='trim').mean()
                intensity=rc.values
                chi=rc['chi'].values
            else:
                intensity=r.values
                chi=r['chi'].values
            # removes any nans    
            a_infs = np.where(np.isnan(intensity))
            intensity=np.delete(intensity,a_infs)
            chi=np.delete(chi,a_infs)
            
            #removes any problematic points around the edge of the mask
            check=np.where(np.logical_and(chi>=ChiU, chi<=ChiL))
            chi=np.delete(chi,check)
            intensity=np.delete(intensity,check)
            
            chi=np.radians(chi) # switches chi from degrees to radians for the fitting function
            
            params, ani, ani_unc, gf=fit_cos(chi, intensity) # calls the fitting function on the selected I vs Chi cut
            
            # Check to compare the Chi value to a upper limit where the fit is assumed to ha
            if gf > Chilim:
                anisotropy[Ecount,i]=0
                anisotropyU[Ecount,i]=ani_unc
                chisq[Ecount,i]=gf
            else: 
                anisotropy[Ecount,i]=ani
                anisotropyU[Ecount,i]=ani_unc
                chisq[Ecount,i]=gf
    if data['polarization'].values[0]==0:
        anisotropy=-1*anisotropy
    return qarray, anisotropy, anisotropyU, chisq
    
    
def fit_cos_anisotropy_single(data,q,qspacing,En,ChiL,ChiU,binnumber,Chilim):
    '''
    fits the anisotropy with a cos function for a single energy/q position
    q = qposition that will be fit
    qspacing = distance between q points 
    En = energy that will be fit
    ChiL, ChiU = allows you to cut out any problematic points around the mask
    binnumber = gives you the option of binning the data in chi - 0 keeps it as is
    Chilim = an upper limit for the goodness of fit beyond which all results will be set to 0, implies that the data does not reflect a cos function for whatever reason
    
    '''
    
    #qarray=np.arange(qL,qU,qspacing)
    #anisotropy=np.zeros([len(Enlist),len(qarray)])
    #chisq=np.zeros([len(Enlist),len(qarray)])
    #anisotropyU=np.zeros([len(Enlist),len(qarray)])
    
    
            
    qmin=q-qspacing/2
    qmax=q+qspacing/2
    r=data.sel(energy=En,q=slice(qmin,qmax)).mean('q')
        
    if binnumber >0:
        rc=r.coarsen(chi=binnumber,boundary='trim').mean()
        intensity=rc.values
        chi=rc['chi'].values
    else:
        intensity=r.values
        chi=r['chi'].values
    a_infs = np.where(np.isnan(intensity))
    intensity=np.delete(intensity,a_infs)
    chi=np.delete(chi,a_infs)
    check=np.where(np.logical_and(chi>=ChiU, chi<=ChiL))
    chi=np.delete(chi,check)
    intensity=np.delete(intensity,check)
    chi=np.radians(chi)
    #print(intensity)
    params, ani, ani_unc, gf=fit_cos(chi, intensity)
    
    return chi,intensity,params, ani, ani_unc, gf

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
    
def sin_func(x,a,c):
    '''
    Helper function - sin function

    Args:
        x (numeric): the input in radians
        a,c: parameters [a,c] where the sin function is = a * np.sin(2*x)+c
    '''
    
    return a* np.sin(2*x)+c
    
def cos_func(x,a,c):
    '''
    Helper function - cos function
    Args:
        x (numeric): the input in radians
        a,c: parameters where the sin function is = a * np.cos(2*x)+c
    '''
    
    
    return a* np.cos(2*x)+c
    
def fit_cos(x_data, y_data):
    '''
    Fits the Intensity vs chi plot with a cosine finction using the scipy.optimize.curve_fit module
    automatically estimates the initial values for the amplitude(a) and offset(c)
    
    Calculates the anisotropy as the ratio between the fitted values for a and the offset
    Ani_unc is the propogated uncertainty
    '''
    a_init=np.max(y_data)-np.min(y_data)
    c_init=np.mean(y_data)
    try:
        params, params_covariance = scipy.optimize.curve_fit(cos_func, x_data, y_data,p0=[a_init, c_init])
        std=np.sqrt(params_covariance)
        Ani=params[0]/params[1]
        Ani_unc=np.sqrt((std[0][0]/params[0])**2 +(std[1][1]/params[1])**2 )*Ani
        if np.abs(Ani) >1:
            Ani=0
        Chisq=np.sum((y_data- cos_func(x_data, params[0], params[1]))**2)
    except ValueError:
        params=0
        Ani=0
        Ani_unc=0
        Chisq=100
    
    return params, Ani, Ani_unc, Chisq