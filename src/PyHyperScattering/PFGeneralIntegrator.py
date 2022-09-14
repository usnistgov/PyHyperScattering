from pyFAI import azimuthalIntegrator
from pyFAI.units import eq_q, formula_q, register_radial_unit
import h5py
import warnings
import xarray as xr
import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from PIL import Image
from skimage import draw

# tqdm.pandas()
# the following block monkey-patches xarray to add tqdm support.  This will not be needed once tqdm v5 releases.
from xarray.core.groupby import DataArrayGroupBy, DatasetGroupBy


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


# end monkey patch

class PFGeneralIntegrator():

    def integrateSingleImage(self, img):
        if type(img) == xr.Dataset:
            for key in img.keys():
                target_key=key
            img=img[key]
        if(img.ndim>2):
            
            img_to_integ = np.squeeze(img.values)
        else:
            img_to_integ = img.values
        
        assert np.shape(self.mask)==np.shape(img_to_integ),f'Error!  Mask has shape {np.shape(self.mask)} but you are attempting to integrate data with shape {np.shape(img_to_integ)}.  Try changing mask orientation or updating mask.'
        stacked_axis = list(img.indexes.keys())
        stacked_axis.remove('pix_x')
        stacked_axis.remove('pix_y')
        assert len(stacked_axis)==1, "More than one axis left after removing pix_x and pix_y, not sure how to handle"
        stacked_axis = stacked_axis[0]
        if(img.__getattr__(stacked_axis).shape[0]>1):
            system_to_integ = [img[0].__getattr__(stacked_axis)]
            warnings.warn(f'There are two images for {img.__getattr__(stacked_axis)}, I am ONLY INTEGRATING THE FIRST.  This may cause the labels to be dropped and the result to need manual re-tagging in the index.',stacklevel=2)
        else:
            system_to_integ = img.__getattr__(stacked_axis)
        if self.do_1d_integration:
            integ_func = self.integrator.integrate1d
        else:
            integ_func = self.integrator.integrate2d

        try:
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
        except TypeError as e:
            if 'diffSolidAngle() missing 2 required positional arguments: ' in str(e):
                raise TypeError(
                    'Geometry is incorrect, cannot integrate.\n \n - Do your mask dimensions match your image dimensions? \n - Do you have pixel sizes set that are not zero?\n - Is SDD, beamcenter/poni, and tilt set correctly?') from e
            else:
                raise e

        if self.maskToNan:
            # preexisting_nans = np.isnan(TwoD.intensity).sum()
            frame.intensity[frame.intensity == -8675309] = np.nan
            # print(f'Patched dummy flag to NaN, number of NaNs = {np.isnan(TwoD.intensity).sum()}, preexisting {preexisting_nans}')
        if self.use_log_ish_binning:
            radial_to_save = np.sinh(frame.radial) / 10000  # was 1000 for inverse nm
        else:
            radial_to_save = frame.radial
        if self.do_1d_integration:
            try:
                res = xr.DataArray([frame.intensity],dims=[stacked_axis,'q'],coords={'q':radial_to_save,stacked_axis:system_to_integ},attrs=img.attrs)
                if self.return_sigma:
                    sigma = xr.DataArray([frame.sigma],dims=[stacked_axis,'q'],coords={'q':radial_to_save,stacked_axis:system_to_integ},attrs=img.attrs)
            except AttributeError:
                res = xr.DataArray(frame.intensity, dims=['q'], coords={'q': radial_to_save}, attrs=img.attrs)
                if self.return_sigma:
                    sigma = xr.DataArray(frame.sigma, dims=['q'], coords={'q': radial_to_save}, attrs=img.attrs)
        else:
            try:
                res = xr.DataArray([frame.intensity],dims=[stacked_axis,'chi','q'],coords={'q':radial_to_save,'chi':frame.azimuthal,stacked_axis:system_to_integ},attrs=img.attrs)
                if self.return_sigma:
                    sigma = xr.DataArray([frame.sigma],dims=[stacked_axis,'chi','q'],coords={'q':radial_to_save,'chi':frame.azimuthal,stacked_axis:system_to_integ},attrs=img.attrs)
            except AttributeError:
                res = xr.DataArray(frame.intensity, dims=['chi', 'q'],
                                   coords={'q': radial_to_save, 'chi': frame.azimuthal}, attrs=img.attrs)
                if self.return_sigma:
                    sigma = xr.DataArray(frame.sigma, dims=['chi', 'q'],
                                         coords={'q': radial_to_save, 'chi': frame.azimuthal}, attrs=img.attrs)
        if self.return_sigma:
            res = res.to_dataset(name='I')
            res['dI'] = sigma
        return res

    def integrateImageStack(self,data):
        indexes = list(data.indexes.keys())
        indexes.remove('pix_x')
        indexes.remove('pix_y')
        if len(indexes) == 1:
            data_int = data.groupby(indexes[0],squeeze=False).progress_apply(self.integrateSingleImage)
        else:
            #some kinda logic to check for existing multiindexes and stack into them appropriately maybe
            data = data.stack({'pyhyper_internal_multiindex':indexes})
            data_int = data.groupby('pyhyper_internal_multiindex',squeeze=False).progress_apply(self.integrateSingleImage).unstack('pyhyper_internal_multiindex')
        return data_int
        #int_stack = img_stack.groupby('system').map_progress(self.integrateSingleImage)
        #PRSUtils.fix_unstacked_dims(int_stack,img_stack,'system',img_stack.attrs['dims_unpacked'])
        #return int_stack
    def __init__(self,maskmethod = "none",maskpath = "",
                 geomethod = "none",
                 NIdistance=0, NIbcx=0, NIbcy=0, NItiltx=0, NItilty=0,
                 NIpixsizex=0, NIpixsizey=0,
                 template_xr=None,
                 energy=2000,
                 integration_method='csr_ocl',
                 correctSolidAngle=True,
                 maskToNan=True,
                 npts=500,
                 use_log_ish_binning=False,
                 do_1d_integration=False,
                 return_sigma=False):
        # energy units eV
        if isinstance(mask, str):
            self.mask = self.load_mask(path=mask, rotate_image=rotate_image)
        else:
            if maskmethod == 'nika':
                if not isinstance(mask, dict):
                    mask = {}
                self.mask = self.load_mask(*mask.update({'nika': maskpath, 'rotate_image': rotate_image}))
            else:
                if not isinstance(mask, dict):
                    self.mask = None
                else:
                    self.mask = self.load_mask(*mask.update({'rotate_image': rotate_image}))
        self.dist = 0.1
        self.poni1 = 0
        self.poni2 = 0
        self.rot1 = 0
        self.rot2 = 0
        self.rot3 = 0
        self.pixel1 = 0 / 1e3
        self.pixel2 = 0 / 1e3
        self.correctSolidAngle = correctSolidAngle
        self.integration_method = integration_method
        self._energy = energy
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
        # self._energy = 0
        if geomethod == "nika":
            self.ni_pixel_x = NIpixsizex
            self.ni_pixel_y = NIpixsizey
            self.ni_distance = NIdistance
            self.ni_beamcenter_x = NIbcx
            self.ni_beamcenter_y = NIbcy
            self.ni_tilt_x = NItiltx
            self.ni_tilt_y = NItilty
        elif geomethod == 'template_xr':
            self.calibrationFromTemplateXRParams(template_xr)
        elif geomethod == "none":
            warnings.warn('Initializing geometry with default values.  This is probably NOT what you want.',
                          stacklevel=2)

        self.recreateIntegrator()

    def __str__(self):
        return f"PyFAI general integrator wrapper SDD = {self.dist} m, poni1 = {self.poni1} m, poni2 = {self.poni2} m, rot1 = {self.rot1} rad, rot2 = {self.rot2} rad"

    def load_mask(self, **kwargs):
        '''
        loads a mask either from a path, from a NIKA file, or from a list of polygon points
        the mask dictionary should have keys indicating which method to use and the necessary information
        for a path, mask itself can be a string, or a dictionary with the key path whose value is the string
        for NIKA, there should be a key "nika" with the path to that file
        for a polygon, there should be a key "points" with a list of lists of points i.e.
            [[[1050,480],[500,480],[500,520],[1050,520]],[[1050,80],[500,80],[500,120],[1050,120]]]

        '''

        if 'points' in kwargs:
            points = kwargs['points']
            xs = []
            ys = []
            for polygon in points:
                x, y = zip(*polygon)
                xs += x
                ys += y
            if 'shape' in kwargs:
                shape = kwargs['shape']
            else:
                shape = (max(xs), max(ys))
            image = np.zeros(shape)
            for polygon in points:
                image += draw.polygon2mask(shape, polygon)
            image[image > 1] = 1
        elif 'image' in kwargs:
            path = kwargs['image']
            im = Image.open(path)
            image = np.array(im)
        elif 'nika' in kwargs:
            image = self.loadNikaMask(kwargs['nika'])
        else:
            warnings.warn("no valid inputs to load or create a mask", stacklevel=2)
            image = None
        if 'rotate_image' in kwargs:
            if kwargs['rotate_image']:
                image = np.flipud(np.rot90(image))
        boolmask = np.invert(image.astype(bool))
        print(f"Imported or created mask with dimensions {str(np.shape(boolmask))}")
        self.mask = boolmask

    def loadNikaMask(self, filetoload):

        '''
        Loads a Nika-generated HDF5 or tiff mask and converts it to an array that matches the local conventions.

        Args:
            filetoload (pathlib.Path or string): path to hdf5/tiff format mask from Nika.
            rotate_image (bool, default True): rotate image as should work
        '''
        mask = None

        if 'h5' in str(filetoload) or 'hdf' in str(filetoload):
            type = 'h5'
            maskhdf = h5py.File(filetoload, 'r')
            mask = maskhdf['M_ROIMask']

        elif 'tif' in str(filetoload):
            type = 'tif'
            mask = plt.imread(filetoload)
        else:
            warnings.warn('Unsupported mask type...', stacklevel=2)
        return mask

    def calibrationFromTemplateXRParams(self, raw_xr):

        '''
        Sets calibration from a pyFAI values in a template xarray

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

    @property
    def wavelength(self):
        return 1.239842e-6 / self._energy  # = wl ; energy = 1.239842e-6 / wl

    @wavelength.setter
    def wavelength(self, value):
        self._energy = 1.239842e-6 / value
        self.recreateIntegrator()

    @property
    def energy(self):
        return self._energy

    @energy.setter
    def energy(self, value):
        self._energy = value
        self.recreateIntegrator()

    @property
    def ni_beamcenter_x(self):
        try:
            return self.poni2 / self.ni_pixel_x * 1000
        except ZeroDivisionError:
            warnings.warn('x pixel size is 0, cannot set beam center, fix pixel size first', stacklevel=2)
            return 0

    @ni_beamcenter_x.setter
    def ni_beamcenter_x(self, value):
        self.poni2 = self.ni_pixel_x * value / 1000
        self.recreateIntegrator()

    @property
    def ni_beamcenter_y(self):
        try:
            return self.poni1 / self.ni_pixel_y * 1000
        except ZeroDivisionError:
            warnings.warn('y pixel size is 0, cannot set beam center, fix pixel size first', stacklevel=2)
            return 0

    @ni_beamcenter_y.setter
    def ni_beamcenter_y(self, value):
        self.poni1 = self.ni_pixel_y * value / 1000
        self.recreateIntegrator()

    @property
    def ni_distance(self):
        return self.dist * 1000

    @ni_distance.setter
    def ni_distance(self, value):
        self.dist = value / 1000
        self.recreateIntegrator()

    @property
    def ni_tilt_x(self):
        return self.rot1 / (math.pi / 180)

    @ni_tilt_x.setter
    def ni_tilt_x(self, value):
        self.rot1 = value * (math.pi / 180)
        self.recreateIntegrator()

    @property
    def ni_tilt_y(self):
        return self.rot2 / (math.pi / 180)  # tilt = rot / const, rot = tilt * const

    @ni_tilt_y.setter
    def ni_tilt_y(self, value):
        self.rot2 = value * (math.pi / 180)
        self.recreateIntegrator()

    @property
    def ni_pixel_x(self):
        return self.pixel2 * 1e3

    @ni_pixel_x.setter
    def ni_pixel_x(self, value):
        self.pixel2 = value / 1e3
        self.ni_beamcenter_x = self.ni_beamcenter_x
        self.recreateIntegrator()

    @property
    def ni_pixel_y(self):
        return self.pixel1 * 1e3

    @ni_pixel_y.setter
    def ni_pixel_y(self, value):
        self.pixel1 = value / 1e3
        self.ni_beamcenter_y = self.ni_beamcenter_y
        self.recreateIntegrator()

    def recreateIntegrator(self):
        '''
        recreate the integrator, after geometry change
        '''
        self.integrator = azimuthalIntegrator.AzimuthalIntegrator(
            self.dist, self.poni1, self.poni2, self.rot1, self.rot2, self.rot3, pixel1=self.pixel1, pixel2=self.pixel2,
            wavelength=self.wavelength)

    def calibrationFromNikaParams(self, distance, bcx, bcy, tiltx, tilty, pixsizex, pixsizey):
        '''
        DEPRECATED as of 0.2

       Set the local calibrations using Nika parameters.
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

        self.ni_pixel_x = pixsizex
        self.ni_pixel_y = pixsizey
        self.ni_distance = distance
        self.ni_beamcenter_x = bcx
        self.ni_beamcenter_y = bcy
        self.ni_tilt_x = tiltx
        self.ni_tilt_y = tilty

        ''' preserved for reference
        self.dist = distance / 1000 # mm in Nika, m in pyFAI
        self.poni1 = bcy * pixsizey / 1000#pyFAI uses the same 0,0 definition, so just pixel to m.  y = poni1, x = poni2
        self.poni2 = bcx * pixsizex / 1000

        self.rot1 = tiltx * (math.pi/180)
        self.rot2 = tilty * (math.pi/180) #degree to radian and flip x/y
        self.rot3 = 0 #don't support this, it's only relevant for multi-detector geometries

        self.pixel1 = pixsizey/1e3
        self.pixel2 = pixsizex/1e3
        self.recreateIntegrator()'''
