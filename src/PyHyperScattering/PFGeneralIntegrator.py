from pyFAI import azimuthalIntegrator
from pyFAI.units import eq_q, formula_q, register_radial_unit
import h5py
import warnings
import xarray as xr
import numpy as np
import math
import matplotlib.pyplot as plt
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
#end monkey patch

class PFGeneralIntegrator():

    def integrateSingleImage(self,img):
        if type(img) == xr.Dataset:
            for key in img.keys():
                target_key=key
            img=img[key]
        if(img.ndim>2):
            img_to_integ = img[0].values
        else:
            img_to_integ = img.values
        
        if(img.system.shape[0]>1):
            system_to_integ = [img[0].system]
            warnings.warn(f'There are two images for {img.system}, I am ONLY INTEGRATING THE FIRST.  This may cause the labels to be dropped and the result to need manual re-tagging in the index.',stacklevel=2)
        else:
            system_to_integ = img.system
        if self.do_1d_integration:
            integ_func = self.integrator.integrate1d  
        else:
            integ_func = self.integrator.integrate2d
            
        frame = integ_func(img_to_integ,
                           self.npts,
                           filename=None,
                           correctSolidAngle=self.correctSolidAngle,
                           error_model="azimuthal",
                           dummy=-8675309 if self.maskToNan else 0,
                           mask=self.mask,
                           unit='arcsinh(q.µm)' if self.use_log_ish_binning else 'q_A^-1',
                           method=self.integration_method
                          )
        
        if self.maskToNan:
                        #preexisting_nans = np.isnan(TwoD.intensity).sum()
                        frame.intensity[frame.intensity==-8675309] = np.nan
                        #print(f'Patched dummy flag to NaN, number of NaNs = {np.isnan(TwoD.intensity).sum()}, preexisting {preexisting_nans}')
        if self.use_log_ish_binning:
            radial_to_save = np.sinh(frame.radial)/10000 #was 1000 for inverse nm
        else:
            radial_to_save = frame.radial
        if self.do_1d_integration:
            try:
                res = xr.DataArray([frame.intensity],dims=['system','q'],coords={'q':radial_to_save,'system':system_to_integ},attrs=img.attrs)
                if self.return_sigma:
                    sigma = xr.DataArray([frame.sigma],dims=['system','q'],coords={'q':radial_to_save,'system':system_to_integ},attrs=img.attrs)
            except AttributeError:
                res = xr.DataArray(frame.intensity,dims=['q'],coords={'q':radial_to_save},attrs=img.attrs)
                if self.return_sigma:
                    sigma = xr.DataArray(frame.sigma,dims=['q'],coords={'q':radial_to_save},attrs=img.attrs)
        else:
            try:
                res = xr.DataArray([frame.intensity],dims=['system','chi','q'],coords={'q':radial_to_save,'chi':frame.azimuthal,'system':system_to_integ},attrs=img.attrs)
                if self.return_sigma:
                    sigma = xr.DataArray([frame.sigma],dims=['system','chi','q'],coords={'q':radial_to_save,'chi':frame.azimuthal,'system':system_to_integ},attrs=img.attrs)
            except AttributeError:
                res = xr.DataArray(frame.intensity,dims=['chi','q'],coords={'q':radial_to_save,'chi':frame.azimuthal},attrs=img.attrs)
                if self.return_sigma:
                    sigma = xr.DataArray(frame.sigma,dims=['chi','q'],coords={'q':radial_to_save,'chi':frame.azimuthal},attrs=img.attrs)
        if self.return_sigma:
            res = res.to_dataset(name='I')
            res['dI'] = sigma
        return res

    def integrateImageStack(self,img_stack):
        int_stack = img_stack.groupby('system').map_progress(self.integrateSingleImage)
        #PRSUtils.fix_unstacked_dims(int_stack,img_stack,'system',img_stack.attrs['dims_unpacked'])
        return int_stack
    def __init__(self,maskmethod = "none",maskpath = "",
                 geomethod = "none",
                 NIdistance=0, NIbcx=0, NIbcy=0, NItiltx=0, NItilty=0,
                 NIpixsizex = 0.027, NIpixsizey = 0.027,
                 template_xr = None,
                 energy = 2000,
                 integration_method='csr_ocl',
                 correctSolidAngle=True,
                 maskToNan = True,
                 npts = 500,
                 use_log_ish_binning=False,
                 do_1d_integration=False,
                 return_sigma=False):
        #energy units eV
        if(maskmethod == "nika"):
            self.loadNikaMask(maskpath)
        elif(maskmethod == "none"):
            self.mask = None
        self.correctSolidAngle = correctSolidAngle
        self.integration_method = integration_method
        self.wavelength = 1.239842e-6/energy
        self.npts = npts
        self.use_log_ish_binning = use_log_ish_binning
        self.do_1d_integration = do_1d_integration
        if self.use_log_ish_binning:
            register_radial_unit("arcsinh(q.µm)",
                     scale=1.0,
                     label=r"arcsinh($q$.µm)",
                     formula="arcsinh(4.0e-6*π/λ*sin(arctan2(sqrt(x**2 + y**2), z)/2.0))")
    
        self.maskToNan = maskToNan
        self.return_sigma = return_sigma
        
        if geomethod == "nika":
            self.calibrationFromNikaParams(NIdistance, NIbcx, NIbcy, NItiltx, NItilty,pixsizex = NIpixsizex, pixsizey = NIpixsizey)
            self.wavelength = 1.239842e-6/energy
        elif geomethod == 'template_xr':
            self.calibrationFromTemplateXRParams(template_xr)
        elif geomethod == "none":
            self.dist = 0.1
            self.poni1 = 0
            self.poni2 = 0
            self.rot1 = 0
            self.rot2 = 0
            self.rot3 = 0
            self.pixel1 = 0.027/1e3
            self.pixel2 = 0.027/1e3
            warnings.warn('Initializing geometry with default values.  This is probably NOT what you want.',stacklevel=2)


        self.recreateIntegrator()

    def __str__(self):
        return f"PyFAI general integrator wrapper SDD = {self.dist} m, poni1 = {self.poni1} m, poni2 = {self.poni2} m, rot1 = {self.rot1} rad, rot2 = {self.rot2} rad"

    def loadNikaMask(self,filetoload,rotate_image=True):
        '''
        Loads a Nika-generated HDF5 or tiff mask and converts it to an array that matches the local conventions.
        
        Args:
            filetoload (pathlib.Path or string): path to hdf5/tiff format mask from Nika.
            rotate_image (bool, default True): rotate image as should work
        '''
        if 'h5' in str(filetoload) or 'hdf' in str(filetoload):
            type='h5'
            maskhdf = h5py.File(filetoload,'r')
            mask = maskhdf['M_ROIMask']

        elif 'tif' in str(filetoload):
            type='tif'
            mask = plt.imread(filetoload)
        else:
            warnings.warn('Unsupported mask type...',stacklevel=2)

            
        if rotate_image: 
            convertedmask = np.flipud(np.rot90(mask))
        else:
            convertedmask = mask
        boolmask = np.invert(convertedmask.astype(bool))
        print(f"Imported Nika mask of type {type}, dimensions {str(np.shape(boolmask))}")
        self.mask = boolmask

    def calibrationFromNikaParams(self,distance, bcx, bcy, tiltx, tilty,pixsizex, pixsizey):
        '''
        Return a calibration array [dist,poni1,poni2,rot1,rot2,rot3] from a Nika detector geometry.
           
           this will probably only support rotations in the SAXS limit (i.e., where sin(x) ~ x, i.e., a couple degrees)
           since it assumes the PyFAI and Nika rotations are about the same origin point (which I think isn't true).
        
        Args:
            distance: sample-detector distance in mm
            bcx: beam center x in pixels
            bcy: beam center y in pixels
            tiltx: detector x tilt in deg, see note above
            tilty: detector y tilt in deg, see note above
            pixsizex: pixel size in x, microns
            pixsizey: pixel size in y, microns
        '''
        self.dist = distance / 1000 # mm in Nika, m in pyFAI
        self.poni1 = bcy * pixsizey / 1000#pyFAI uses the same 0,0 definition, so just pixel to m.  y = poni1, x = poni2
        self.poni2 = bcx * pixsizex / 1000

        self.rot1 = tiltx * (math.pi/180)
        self.rot2 = tilty * (math.pi/180) #degree to radian and flip x/y
        self.rot3 = 0 #don't support this, it's only relevant for multi-detector geometries

        self.pixel1 = pixsizey/1e3
        self.pixel2 = pixsizex/1e3
        self.recreateIntegrator()
        
    def calibrationFromTemplateXRParams(self,raw_xr):
        '''
        Return a calibration array [dist,poni1,poni2,rot1,rot2,rot3] from a pyFAI values in a template xarray
        
        Args:
            raw_xr (raw format xarray): a raw_xr bearing the metadata in members
        
        '''
        self.dist = raw_xr.dist
        self.poni1 = raw_xr.poni1
        self.poni2 = raw_xr.poni2

        self.rot1 = raw_xr.rot1
        self.rot2 = raw_xr.rot2
        self.rot3 = raw_xr.rot3

        self.pixel1 = raw_xr.pixel1
        self.pixel2 = raw_xr.pixel2
        
        self.recreateIntegrator()
    def recreateIntegrator(self):
        '''
        recreate the integrator, after geometry change
        '''
        self.integrator = azimuthalIntegrator.AzimuthalIntegrator(
            self.dist, self.poni1, self.poni2, self.rot1, self.rot2, self.rot3 ,pixel1=self.pixel1,pixel2=self.pixel2, wavelength = self.wavelength)
