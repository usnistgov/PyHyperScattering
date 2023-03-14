import warnings
import xarray as xr
import numpy as np
import math

#@xr.register_dataset_accessor('rsoxs')

@xr.register_dataarray_accessor('rsoxs')
class RSoXS:
    def __init__(self,xr_obj):
        self._obj=xr_obj
        
        self._pyhyper_type = 'reduced'
        try:
            self._chi_min = np.min(xr_obj.chi)
            self._chi_max = np.max(xr_obj.chi)
            self._chi_range = [self._chi_min,self._chi_max]
        except AttributeError:
            self._pyhyper_type = 'raw'
        
    def slice_chi(self,chi,chi_width=5):
        '''
        slice an xarray in chi

        Args:
            img (xarray): xarray to work on
            chi (numeric): q about which slice should be centered, in deg
            chi_width (numeric): width of slice in each direction, in deg
        '''
        
        slice_begin =  chi-chi_width
        slice_end = chi+chi_width
        '''
            cases to handle:
            1) wrap-around slice.  begins before we start and ends after we end.  return whole array and warn.
            2) wraps under, ends inside
            3) wraps under, ends under --> translate both coords
            4) wraps over, ends inside
            5) wraps over, ends over --> translate both coords.
            6) begins inside, ends inside
        '''
        
        if slice_begin < self._chi_min and slice_end < self._chi_min:
            #case 3
            nshift = math.floor((self._chi_min-slice_end)/360) +1
            slice_begin += 360*nshift
            slice_end += 360*nshift
        elif slice_begin > self._chi_max and slice_end > self._chi_max:
            #case 5
            nshift = math.floor((slice_begin - self._chi_max)/360) +1
            slice_begin -= 360*nshift
            slice_end -= 360*nshift
        
        
        if slice_begin<self._chi_min and slice_end > self._chi_max:
            #case 1
            warnings.warn(f'Chi slice specified from {slice_begin} to {slice_end}, which exceeds range of {self._chi_min} to {self._chi_max}.  Returning mean across all values of chi.',stacklevel=2)
            selector = np.ones_like(self._obj.chi,dtype=bool)
        elif slice_begin<self._chi_min and slice_end < self._chi_max :
            #wrap-around _chi_min: case 2
            selector = np.logical_and(self._obj.chi>=self._chi_min,
                                      self._obj.chi <= slice_end)
            selector = np.logical_or(selector,
                                     np.logical_and(self._obj.chi<=self._chi_max,
                                       self._obj.chi >= (self._chi_max - (self._chi_min - slice_begin) +1)))
        elif slice_end > self._chi_max and slice_begin > self._chi_min:
            #wrap-around _chi_max: case 4
            selector = np.logical_and(self._obj.chi<= self._chi_max,
                                      self._obj.chi >= slice_begin)
            selector = np.logical_or(selector,
                                    np.logical_and(self._obj.chi>= self._chi_min,
                                           self._obj.chi <= (self._chi_min + (slice_end - self._chi_max)-1)))
        else:
            #simple slice, case 6, hooray
            selector = np.logical_and(self._obj.chi>=slice_begin,
                                      self._obj.chi<=slice_end)
            
        return self._obj.isel({'chi':selector}).mean('chi')
            
    def slice_q(self,q,q_width=None):
        '''
        slice an xarray in q

        Args:
            img (xarray): xarray to work on
            q (numeric): q about which slice should be centered
            q_width (numeric): width of slice in each direction, in q units
        '''
        img = self._obj
        if q_width==None:
            q_width = 0.1*q
        return img.sel(q=slice(q-q_width,q+q_width)).mean('q')

    def select_chi(self,chi,method='nearest'):
        if chi < self._chi_min:
            nshift = math.floor((self._chi_min-chi)/360) +1
            chi += 360*nshift
        elif chi > self._chi_max:
            nshift = math.floor((chi - self._chi_max)/360) +1
            chi -= 360*nshift
        return self._obj.sel(chi=chi,method=method)

    def select_q(self,q,method='interp'):
        return self._obj.sel(q=q,method=method)
    def select_pol(self,pol,method='nearest'):
        return self._obj.sel(polarization=pol,method=method)


    def AR(self,calc2d=False,two_AR=False,chi_width=5,calc2d_norm_energy=None):
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
            para = self.slice_chi(0,chi_width=chi_width)
            perp = self.slice_chi(-90,chi_width=chi_width)
            return ((para - perp) / (para+perp))
        elif(calc2d):
            para_pol = self.select_pol(0)
            perp_pol = self.select_pol(90)

            para_para = para_pol.slice_chi(0,chi_width=chi_width)
            para_perp = para_pol.slice_chi(-90,chi_width=chi_width)

            perp_perp = perp_pol.slice_chi(-90,chi_width=chi_width)
            perp_para = perp_pol.slice_chi(0,chi_width=chi_width)

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