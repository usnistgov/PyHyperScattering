import scipy.optimize
import xarray as xr
import numpy as np


def fit_lorentz(x,guess=None,pos_int_override=False):
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
    if guess == None:
        guess = [500.,0.00665,0.0002]
        pos_int_override=True
    if pos_int_override:
        guess[1] = np.median(x.coords['q'])
        guess[0] = x.sel(q=guess[1],method='nearest')
    print(f"Starting fit on {x.coords}")
    try:
        coeff, var_matrix = scipy.optimize.curve_fit(lorentz,x.coords['q'],x.data,p0=guess)
    except RuntimeError:
        print("Fit failed to converge")
        return xr.DataArray(data=np.nan,coords=x.coords)
    print(f"Fit completed, coeff = {coeff}")
    return xr.DataArray(data=coeff[0],coords=x.coords)

def fit_lorentz_bg(x,guess=None,pos_int_override=False):
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
    if guess == None:
        guess = [500.,0.00665,0.0002,0]
        pos_int_override=True
    if pos_int_override:
        guess[1] = np.median(x.coords['q'])
        guess[0] = x.sel(q=guess[1],method='nearest')
    print(f"Starting fit on {x.coords}")
    try:
        coeff, var_matrix = scipy.optimize.curve_fit(lorentz_w_flat_bg,x.coords['q'],x.data,p0=guess)
    except RuntimeError:
        print("Fit failed to converge")
        return xr.DataArray(data=np.nan,coords=x.coords)
    print(f"Fit completed, coeff = {coeff}")
    return xr.DataArray(data=coeff[0],coords=x.coords)


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