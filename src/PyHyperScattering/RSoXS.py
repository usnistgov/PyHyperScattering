import warnings
import xarray as xr
import numpy as np
import math

def slice_chi(img,chi,chi_width=5):
    return img.sel(chi=slice(chi-chi_width,chi+chi_width)).mean('chi')

def slice_q(img,q,q_width=None):
    '''
    slice an xarray in q
    
    @param img: xarray to work on
    
    @param q: q about which slice should be centered
    
    @param q_width (default 10%): width of slice in each direction.
    '''
    if q_width==None:
        q_width = 0.1*q
    return img.sel(q=slice(q-q_width,q+q_width)).mean('q')

def select_chi(img,chi,method='interp'):
    return img.sel(chi=chi,method=method)

def select_q(img,q,method='interp'):
    return img.sel(q=q,method=method)
def select_pol(img,pol,method='nearest'):
    return img.sel(pol=pol,method=method)
    
    
def AR(img,two_AR=False,chi_width=5):
    '''
    img can either be a single dataarray (in which case we'll compute AR using 0 and 90 deg chi slices), or a list of two dataarrays with polarization 0 and 90 (in which case we'll use the more rigorous approach decoupling the polarization-induced anisotropy)
    '''
    
    if(type(img)==xr.DataArray):
        para = slice_chi(img,0,chi_width=chi_width)
        perp = slice_chi(img,-90,chi_width=chi_width)
        return ((para - perp) / (para+perp))
    elif(len(img)==2):
        para_pol = select_pol(img,0)
        perp_pol = select_pol(img,-90)
        
        para_para = slice_chi(para_pol,0)
        para_perp = slice_chi(para_pol,-90)
        
        perp_perp = slice_chi(perp_pol,-90)
        perp_para = slice_chi(perp_pol,0)
        
        AR_para = ((para_para - para_perp)/(para_para+para_perp))
        AR_perp = ((perp_perp - perp_para)/(perp_perp+perp_para))
        
        if two_AR:
            return (AR_para,AR_perp)
        else:
            return (AR_para+AR_perp)/2
    else:
        raise NotImplementedError('Need either a single DataArray or a list of 2 dataarrays')
    
def collate_AR_stack(sample,energy):
    raise NotImplementedError('This is a stub function. Should return tuple of the two polarizations, but it does not yet.')
    '''for sam in data_idx.groupby('sample'):
        print(f'Processing for {sam[0]}')
        for enset in sam[1].groupby('energy'):
            print(f'    Processing energy group {enset[0]}')
            pol90 = enset[1][enset[1]['pol']==90.0].num
            pol0 = enset[1][enset[1]['pol']==0.0].num
            print(f'        Pol 0: {pol0}')
            print(f'        Pol 90: {pol90}')'''
    
'''
    
for img in int_stack:
    f = plt.figure()

    img.sel(chi=slice(-5,5)).unstack('system').mean('chi').plot(label='0 deg ± 5 deg',norm=LogNorm(1e1,1e5))
    plt.title(f'{img.sample_name} @ pol = {float(img.polarization[0])}, chi = 0 deg ± 5 deg')
    plt.legend()
    plt.show()
    plt.savefig(f'2D_chi0_{img.sample_name}_pol{float(img.polarization[0])}.png')
    plt.close()
    img.sel(chi=slice(-95,-85)).unstack('system').mean('chi').plot(label='90 deg ± 5 deg',norm=LogNorm(1e1,1e5))
    plt.title(f'{img.sample_name} @ pol = {float(img.polarization[0])}, chi = 90 deg ± 5 deg')
    plt.legend()
    plt.show()
    plt.savefig(f'2D_chi90_{img.sample_name}_pol{float(img.polarization[0])}.png')
    plt.close()
    '''