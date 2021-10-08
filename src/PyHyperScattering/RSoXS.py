import warnings
import xarray as xr
import numpy as np
import math

def slice_chi(img,chi,chi_width=5):
    '''
    slice an xarray in chi
    
    Args:
        img (xarray): xarray to work on
        chi (numeric): q about which slice should be centered, in deg
        chi_width (numeric): width of slice in each direction, in deg
    '''
    return img.sel(chi=slice(chi-chi_width,chi+chi_width)).mean('chi')

def slice_q(img,q,q_width=None):
    '''
    slice an xarray in q
    
    Args:
        img (xarray): xarray to work on
        q (numeric): q about which slice should be centered
        q_width (numeric): width of slice in each direction, in q units
    '''
    if q_width==None:
        q_width = 0.1*q
    return img.sel(q=slice(q-q_width,q+q_width)).mean('q')

def select_chi(img,chi,method='interp'):
    return img.sel(chi=chi,method=method)

def select_q(img,q,method='interp'):
    return img.sel(q=q,method=method)
def select_pol(img,pol,method='nearest'):
    return img.sel(polarization=pol,method=method)
    
    
def AR(img,calc2d=False,two_AR=False,chi_width=5,calc2d_norm_energy=None):
    '''
    Calculate the RSoXS Anisotropic Ratio (AR) of either a single RSoXS scan or a polarized pair of scans.

    AR is defined as (para-perp)/(para+perp) where para is the chi slice parallel to the polarization direction, and perp is the chi slice 90 deg offset from the polarization direction.
    
    Args:
        img (xarray): image to plot
        calc2d (bool): calculate the AR using both polarizations
        two_AR (bool): return both polarizations if calc2d = True.  If two_AR = False, return the average AR between the two polarizations.
        calc2d_norm_energy (numeric): if set, normalizes each polarization's AR at a given energy.  THIS EFFECTIVELY FORCES THE AR TO 0 AT THIS ENERGY.
    '''
    
    if(not calc2d):
        para = slice_chi(img,0,chi_width=chi_width)
        perp = slice_chi(img,-90,chi_width=chi_width)
        return ((para - perp) / (para+perp))
    elif(calc2d):
        para_pol = select_pol(img,0)
        perp_pol = select_pol(img,90)
        
        para_para = slice_chi(para_pol,0)
        para_perp = slice_chi(para_pol,-90)
        
        perp_perp = slice_chi(perp_pol,-90)
        perp_para = slice_chi(perp_pol,0)
        
        AR_para = ((para_para - para_perp)/(para_para+para_perp))
        AR_perp = ((perp_perp - perp_para)/(perp_perp+perp_para))
        
        if calc2d_norm_energy is not None:
            AR_para = AR_para / AR_para.sel(energy=calc2d_norm_energy)
            AR_perp = AR_perp / AR_perp.sel(energy=calc2d_norm_energy)

        if AR_para < AR_perp or AR_perp < AR_para:
            warnings.warn('One polarization has a systematically higher/lower AR than the other.  Typically this indicates bad intensity values.',stacklevel=2)

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