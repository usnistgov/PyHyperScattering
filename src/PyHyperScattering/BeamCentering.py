import warnings
import numpy as np
import xarray as xr
from pyFAI import azimuthalIntegrator
from scipy.optimize import minimize
from tqdm.auto import tqdm


@xr.register_dataarray_accessor('beamcenter')
class CenteringAccessor:

    def __init__(self, xr_obj):
        self._obj = xr_obj
        self._pyhyper_type = 'reduced'
        try:
            self._chi_min = np.min(xr_obj.chi)
            self._chi_max = np.max(xr_obj.chi)
            self._chi_range = [self._chi_min, self._chi_max]
        except AttributeError:
            self._pyhyper_type = 'raw'
        self.integrator = None
        self.centering_energy = None

    def create_integrator(self, energy,poni1=None,poni2=None):
        if poni1 == None:
            poni1 = self._obj.poni1
        if poni2 == None:
            poni2 = self._obj.poni2
        return azimuthalIntegrator.AzimuthalIntegrator(
            self._obj.dist, self._obj.poni1, self._obj.poni2,
            self._obj.rot1, self._obj.rot2, self._obj.rot3,
            pixel1=self._obj.pixel1, pixel2=self._obj.pixel2,
            wavelength=1.239842e-6/energy
        )

    def optimization_func(self, x, image, obj, energy,
                          q_min, q_max, pbar=None,
                          chi_min=-179, chi_max=179,
                          num_points=150):
        #print(f'evaluating objective at {x}')
        integrator = obj.create_integrator(energy,poni1=x[0],poni2=x[1])
        _, image_int = integrator.integrate_radial(image, num_points,
                                                   radial_range=(q_min, q_max),
                                                   azimuth_range=(chi_min, chi_max),
                                                   radial_unit='q_A^-1')
        if pbar is not None:
            pbar.update(1)
        return np.var(image_int[image_int != 0])

    def refine_geometry(self, energy, q_min, q_max,
                        chi_min=-179, chi_max=179,
                        poni1_guess=None, poni2_guess=None,
                        bounds=None, method='Nelder-Mead',
                        num_points=150, max_iter = 150, 
                        verbose=False, ):
        if not poni1_guess:
            poni1_guess = self._obj.poni1
            poni2_guess = self._obj.poni2
        if (not self.integrator) | (self.centering_energy != energy):
            self.integrator = self.create_integrator(energy=energy)
            self.centering_energy = energy
        if not bounds:
            bounds = [(poni1_guess*0.9, poni1_guess*1.1),
                      (poni2_guess*0.9, poni2_guess*1.1)]
        options = {}
        options['maxiter'] = max_iter
        if verbose:
            options['disp'] = True
        image = self._obj.sel(energy=energy).values
        with tqdm(total=1,desc='Optimizing Beamcenter') as pbar:
            res = minimize(self.optimization_func, (poni1_guess, poni2_guess),
                           args=(image, self, energy,
                                 q_min, q_max, pbar, chi_min, chi_max,
                                 num_points),
                           bounds=bounds, method=method, options=options)

        if res.success:
            if (res.x == (self._obj.poni1,self._obj.poni2)).all():
                print(f'Optimization successful, already at optimum values.  Nothing changed.')
                
            else:
                print(f'Optimization successful. Updating beamcenter to ({res.x[0]},{res.x[1]}), old values were ({self._obj.poni1},{self._obj.poni2})')
                self._obj.attrs['poni1'] = res.x[0]
                self._obj.attrs['poni2'] = res.x[1]
                
            return res
        else:
            warnings.warn('Optimization was unsuccessful. Try new guesses and start again')
