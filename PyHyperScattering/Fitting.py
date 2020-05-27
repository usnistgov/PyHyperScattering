import scipy.optimize
import xarray as xr
import numpy as np


def fit_lorentz(x,guess=None,pos_int_override=False):
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
    return xr.DataArray(data=0,coords=x.coords)

def gauss(x, *p):
    A, mu, sigma = p
    return A*numpy.exp(-(x-mu)**2/(2.*sigma**2))
def lorentz( x, *p ):
    a, x0, gam = p
    return a * gam**2 / ( gam**2 + ( x - x0 )**2)
def lorentz_w_flat_bg( x, *p ):
    a, x0, gam, bg = p
    return bg+(a * gam**2 / ( gam**2 + ( x - x0 )**2))