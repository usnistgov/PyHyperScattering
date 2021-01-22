from pyFAI import azimuthalIntegrator
import h5py
import warnings
import xarray as xr
import numpy as np
import math

class PFGeneralIntegrator():

    def integrateSingleImage(self,img):
        if(img.ndim>2):
            img_to_integ = img[0]
            img_to_integ = img[0].values
        else:
            img_to_integ = img

            img_to_integ = img.values
            
        if(img.system.shape[0]>1):
            system_to_integ = [img[0].system]
            warnings.warn(f'There are two images for {img.system}, I am ONLY INTEGRATING THE FIRST.  This may cause the labels to be dropped and the result to need manual re-tagging in the index.')
        else:
            system_to_integ = img.system
        TwoD = self.integrator.integrate2d(img_to_integ,
                                               self.npts,
                                               filename=None,
                                               correctSolidAngle=True,
                                               error_model="azimuthal",
                                               mask=self.mask,
                                               unit='q_A^-1',
                                               method=self.integration_method
                                              )

        try:
            return xr.DataArray([TwoD.intensity],dims=['system','chi','q'],coords={'q':TwoD.radial,'chi':TwoD.azimuthal,'system':img.system},attrs=img.attrs)
            return xr.DataArray([TwoD.intensity],dims=['system','chi','q'],coords={'q':TwoD.radial,'chi':TwoD.azimuthal,'system':system_to_integ},attrs=img.attrs)
        except AttributeError:
            return xr.DataArray(TwoD.intensity,dims=['chi','q'],coords={'q':TwoD.radial,'chi':TwoD.azimuthal},attrs=img.attrs)


    def integrateImageStack(self,img_stack):
        int_stack = img_stack.groupby('system').map(self.integrateSingleImage)
        #PRSUtils.fix_unstacked_dims(int_stack,img_stack,'system',img_stack.attrs['dims_unpacked'])
        return int_stack
    def __init__(self,maskmethod = "none",maskpath = "",
                 geomethod = "none",
                 NIdistance=0, NIbcx=0, NIbcy=0, NItiltx=0, NItilty=0,
                 NIpixsizex = 0.027, NIpixsizey = 0.027,
                 energy = 2000,
                 integration_method='csr_ocl',
                 npts = 500):
        #energy units eV
        if(maskmethod == "nika"):
            self.loadNikaMask(maskpath)
        elif(maskmethod == "none"):
            self.mask = None

        self.integration_method = integration_method
        self.wavelength = 1.239842e-6/energy
        self.npts = npts

        if geomethod == "nika":
            self.calibrationFromNikaParams(NIdistance, NIbcx, NIbcy, NItiltx, NItilty,pixsizex = NIpixsizex, pixsizey = NIpixsizey)
            self.wavelength = 1.239842e-6/energy
        elif geomethod == "none":
            self.dist = 0.1
            self.poni1 = 0
            self.poni2 = 0
            self.rot1 = 0
            self.rot2 = 0
            self.rot3 = 0
            self.pixel1 = 0.027/1e3
            self.pixel2 = 0.027/1e3
            warnings.warn('Initializing geometry with default values.  This is probably NOT what you want.')


        self.recreateIntegrator()

    def __str__(self):
        return f"PyFAI general integrator wrapper SDD = {self.dist} m, poni1 = {self.poni1} m, poni2 = {self.poni2} m, rot1 = {self.rot1} rad, rot2 = {self.rot2} rad"


    def loadNikaMask(self,filetoload):
        #Loads a Nika-generated HDF5 mask and converts it to an array that matches the local conventions.
        maskhdf = h5py.File(filetoload,'r')
        convertedmask = np.flipud(np.rot90(maskhdf['M_ROIMask']))
        boolmask = np.invert(convertedmask.astype(bool))
        print("Imported Nika mask, dimensions " + str(np.shape(boolmask)))
        self.mask = boolmask

    def calibrationFromNikaParams(self,distance, bcx, bcy, tiltx, tilty,pixsizex, pixsizey):
        #Return a calibration array [dist,poni1,poni2,rot1,rot2,rot3] from a Nika detector geometry
        # if you change the CCD binning, pixsizexy params need to be given.  Default is for 4x4 binning which results in effective size of 27 um.
        #this will probably only support rotations in the SAXS limit (i.e., where sin(x) ~ x, i.e., a couple degrees)
        # since it assumes the PyFAI and Nika rotations are about the same origin point (which I think isn't true).

        self.dist = distance / 1000 # mm in Nika, m in pyFAI
        self.poni1 = bcy * pixsizey / 1000#pyFAI uses the same 0,0 definition, so just pixel to m.  y = poni1, x = poni2
        self.poni2 = bcx * pixsizex / 1000

        self.rot1 = tiltx * (math.pi/180)
        self.rot2 = tilty * (math.pi/180) #degree to radian and flip x/y
        self.rot3 = 0 #don't support this, it's only relevant for multi-detector geometries

        self.pixel1 = pixsizey/1e3
        self.pixel2 = pixsizex/1e3
        self.recreateIntegrator()
    def recreateIntegrator(self):
        #loads an image file, spins up a pyFAI integrator with the right params, and integrates it.

        self.integrator = azimuthalIntegrator.AzimuthalIntegrator(
            self.dist, self.poni1, self.poni2, self.rot1, self.rot2, self.rot3 ,pixel1=self.pixel1,pixel2=self.pixel2, wavelength = self.wavelength)
