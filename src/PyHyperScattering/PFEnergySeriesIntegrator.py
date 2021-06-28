from pyFAI import azimuthalIntegrator
from PyHyperScattering.PFGeneralIntegrator import PFGeneralIntegrator
import h5py
import warnings
import xarray as xr
import numpy as np
import math

class PFEnergySeriesIntegrator(PFGeneralIntegrator):

    def integrateSingleImage(self,img):
        # for each image: 
        #    get the energy and locate the matching integrator
        #    use that integrator to reduce
        #    return single reduced frame
        if type(img.energy) != float:
            en = img.energy.values[0]
        else:
            en = img.energy
        try:
            self.integrator = self.integrator_stack[en]
        except KeyError:
            self.integrator = self.createIntegrator(en)
        res = super().integrateSingleImage(img)
        try:
            if len(self.dest_q)>0:
                return res.interp(q=self.dest_q)
            else:
                return res
        except TypeError:
            return res
    def integrateImageStack(self,img_stack):
        
        if img_stack.system.to_pandas().drop_duplicates().shape[0] != img_stack.system.shape[0]:
            warnings.warn('Your system contains duplicate conditions.  This is not supported and may not work.  Try adding additional coords to separate image conditions')
        
        # get just the energies of the image stack
        energies = img_stack.energy.to_dataframe()
        
        energies = energies['energy'].drop_duplicates()
        
        #create an integrator for each energy
        for en in energies:
            self.createIntegrator(en)
        self.createIntegrator(np.median(energies))
        # find the output q for the midpoint and set the final q binning
        self.dest_q = self.integrator_stack[np.median(energies)].integrate2d(np.zeros_like(self.mask).astype(int), self.npts, 
                                               unit='q_A^-1',
                                               method=self.integration_method).radial
        
        # single image reduce each entry in the stack
        # + 
        # restack the reduced data

        return img_stack.groupby('system',squeeze=False).map(self.integrateSingleImage)
    
    def createIntegrator(self,en):
        self.integrator_stack[en] = azimuthalIntegrator.AzimuthalIntegrator(
            self.dist, self.poni1, self.poni2, self.rot1, self.rot2, self.rot3 ,pixel1=self.pixel1,pixel2=self.pixel2, wavelength = 1.239842e-6/en)
        return self.integrator_stack[en]
    def __init__(self,**kwargs):
        #@todo: how much of this can be in a super.init call?
        self.integrator_stack = {}
        
        super().__init__(**kwargs)
    def recreateIntegrator(self):
        pass
    
    def __str__(self):
        return f"PyFAI energy-series integrator  SDD = {self.dist} m, poni1 = {self.poni1} m, poni2 = {self.poni2} m, rot1 = {self.rot1} rad, rot2 = {self.rot2} rad"