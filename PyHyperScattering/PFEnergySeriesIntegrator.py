from pyFAI import azimuthalIntegrator
from PFGeneralIntegrator import PFGeneralIntegrator
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
            
        # find the output q for the midpoint and set the final q binning
        self.dest_q = self.integrator_stack[np.median(energies)].integrate2d(np.zeros_like(self.mask).astype(int), self.npts, 
                                               unit='q_A^-1',
                                               method=self.integration_method).radial
        
        # single image reduce each entry in the stack
        # + 
        # restack the reduced data

        return img_stack.groupby('system').map(self.integrateSingleImage)
    
    def createIntegrator(self,en):
        self.integrator_stack[en] = azimuthalIntegrator.AzimuthalIntegrator(
            self.dist, self.poni1, self.poni2, self.rot1, self.rot2, self.rot3 ,pixel1=self.pixel1,pixel2=self.pixel2, wavelength = 1.239842e-6/en)
        return self.integrator_stack[en]
    def __init__(self,maskmethod = "none",maskpath = "",
                 geomethod = "none",
                 NIdistance=0, NIbcx=0, NIbcy=0, NItiltx=0, NItilty=0,
                 NIpixsizex = 0.027, NIpixsizey = 0.027,
                 template_xr = None,
                 integration_method='csr_ocl',
                 correctSolidAngle=True,
                 npts = 500):
        #@todo: how much of this can be in a super.init call?
        self.integrator_stack = {}
        
        if(maskmethod == "nika"):
            self.loadNikaMask(maskpath)
        elif(maskmethod == "none"):
            self.mask = None

        self.integration_method = integration_method
        self.npts = npts
        self.dest_q = None
        
        if geomethod == "nika":
            self.calibrationFromNikaParams(NIdistance, NIbcx, NIbcy, NItiltx, NItilty,pixsizex = NIpixsizex, pixsizey = NIpixsizey)
        if geomethod == 'template_xr':
            self.calibrationFromTemplateXRParams(template_xr)
        elif geomethod == "none":
            self.dist = 0.1
            self.poni1 = 0
            self.poni2 = 0
            self.rot1 = 0
            self.rot2 = 0
            self.rot3 = 0
            self.pixel1 = 0.027
            self.pixel2 = 0.027
            warnings.warn('Initializing geometry with default values.  This is probably NOT what you want.')
        
    def recreateIntegrator(self):
        pass
    
    def __str__(self):
        return f"PyFAI energy-series integrator  SDD = {self.dist} m, poni1 = {self.poni1} m, poni2 = {self.poni2} m, rot1 = {self.rot1} rad, rot2 = {self.rot2} rad"