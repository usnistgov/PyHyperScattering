from pyFAI import azimuthalIntegrator
from pyFAI.units import eq_q, formula_q, register_radial_unit
from pyFAI.io.ponifile import PoniFile
import h5py
import warnings
import xarray as xr
import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from PIL import Image
from skimage import draw
import json
import pandas as pd
import fabio

# tqdm.pandas()
# the following block monkey-patches xarray to add tqdm support.  This will not be needed once tqdm v5 releases.
from xarray.core.groupby import DataArrayGroupBy, DatasetGroupBy


def inner_generator(df_function='apply'):
    def inner(df, func, *args, **kwargs):
        t = tqdm(total=len(df))

        def wrapper(*args, **kwargs):
            t.update(n=1 if not t.total or t.n < t.total else 0)
            return func(*args, **kwargs)

        result = getattr(df, df_function)(wrapper, **kwargs)

        t.close()
        return result

    return inner


DataArrayGroupBy.progress_apply = inner_generator(df_function='apply')
DatasetGroupBy.progress_apply = inner_generator(df_function='apply')

DataArrayGroupBy.progress_map = inner_generator(df_function='map')
DatasetGroupBy.progress_map = inner_generator(df_function='map')

DataArrayGroupBy.progress_map_blocks = inner_generator(df_function='map_blocks')
DatasetGroupBy.progress_map_blocks = inner_generator(df_function='map_blocks')

# end monkey patch


class PFGeneralIntegrator:
    """PyFAI general integrator wrapper"""

    def __init__(self,
                 maskmethod = 'none',
                 maskrotate = True,
                 geomethod = 'none',
                 NIdistance = 0, NIbcx = 0, NIbcy = 0, NItiltx = 0, NItilty = 0,
                 NIpixsizex = 0, NIpixsizey = 0,
                 template_xr = None,
                 ponifile = None,
                 energy = 2000,
                 integration_method = 'csr_ocl',
                 correctSolidAngle = True,
                 maskToNan = True,
                 npts = 500,
                 use_log_ish_binning = False,
                 do_1d_integration = False,
                 return_sigma = False,
                 use_chunked_processing = False,
                 **kwargs):
        
        """
        General pyFAI-wrapped integrator class

        Some Inputs:
        maskmethod (str, default = 'none'): What type of mask to load 
                   options: [nika, polygon, image, pyhyper, edf, numpy, none]

        geomethod (str, default = 'none'): where to get calibration information 
                  from for integrators
                  options: ['nika', 'template_xr', 'ponifile', 'none']

        template_xr (xr.DataArray): xarray for example shape for empty masks, 
                    and attributes for calibration if geomethod='template_xr'

        ponifile (str or pathlib.Path):

        Important keyword arguments:
        maskpath (str or pathlib.Path): path to mask, if specifed a method that 
                                        requires a file     
        mask (numpy.ndarray): if maskmethod is 'numpy', supply an array mask                        
        """     

        if maskmethod == 'nika':
            self.loadNikaMask(rotate_image=maskrotate, **kwargs)
        elif maskmethod == 'polygon':
            self.loadPolyMask(**kwargs)
        elif maskmethod == 'image':
            self.loadImageMask(maskrotate=maskrotate, **kwargs)
        elif maskmethod == 'pyhyper':
            self.loadPyHyperMask(**kwargs)
        elif maskmethod == 'edf':
            self.loadEdfMask(**kwargs)
        elif maskmethod == 'numpy':
            self.mask = kwargs['mask']
        elif maskmethod == 'none':
            self.mask = None
        else:
            raise ValueError(f'Invalid or unsupported maskmethod {maskmethod}.')
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
            register_radial_unit(
                "arcsinh(q.µm)",
                scale=1.0,
                label=r"arcsinh($q$.µm)",
                formula="arcsinh(4.0e-6*π/λ*sin(arctan2(sqrt(x**2 + y**2), z)/2.0))",
            )

        self.maskToNan = maskToNan
        self.return_sigma = return_sigma
        self.use_chunked_processing = use_chunked_processing
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
        elif geomethod == 'ponifile':
            self.calibrationFromPoniFile(ponifile)
        elif geomethod == "none":
            warnings.warn(
                'Initializing geometry with default values.  This is probably NOT what you want.',
                stacklevel=2,
            )

        self.recreateIntegrator()
    
    def __str__(self):
        return f"PyFAI general integrator wrapper SDD = {self.dist} m, poni1 = {self.poni1} m, poni2 = {self.poni2} m, rot1 = {self.rot1} rad, rot2 = {self.rot2} rad"

    def integrateSingleImage(self, img):
        if type(img) == xr.Dataset:
            for key in img.keys():
                target_key = key
            img = img[key]
        if img.ndim > 2:

            img_to_integ = np.squeeze(img.values)
        else:
            img_to_integ = img.values

        if self.mask is None:
            warnings.warn(
                'No mask defined.  Creating an empty mask with dimensions {img.shape}.',
                stacklevel=2,
            )
            self.mask = np.zeros_like(img).squeeze()
        assert np.shape(self.mask) == np.shape(
            img_to_integ
        ), f'Error!  Mask has shape {np.shape(self.mask)} but you are attempting to integrate data with shape {np.shape(img_to_integ)}.  Try changing mask orientation or updating mask.'
        stacked_axis = list(img.dims)
        stacked_axis.remove('pix_x')
        stacked_axis.remove('pix_y')
        if len(stacked_axis) > 0:
            assert (
                len(stacked_axis) == 1
            ), f"More than one dimension left after removing pix_x and pix_y, I see {stacked_axis}, not sure how to handle"
            stacked_axis = stacked_axis[0]
            # print(f'looking for {stacked_axis} in {img[0].indexes} (indexes), it has dims {img[0].dims} and looks like {img[0]}')
            if img.__getattr__(stacked_axis).shape[0] > 1:
                system_to_integ = img[0].indexes[stacked_axis]
                warnings.warn(
                    f'There are two images for {img.__getattr__(stacked_axis)}, I am ONLY INTEGRATING THE FIRST.  This may cause the labels to be dropped and the result to need manual re-tagging in the index.',
                    stacklevel=2,
                )
            else:
                system_to_integ = img.indexes[stacked_axis]

        else:
            stacked_axis = 'image_num'
            system_to_integ = [0]
        if self.do_1d_integration:
            integ_func = self.integrator.integrate1d
        else:
            integ_func = self.integrator.integrate2d

        try:
            frame = integ_func(
                img_to_integ,
                self.npts,
                filename=None,
                correctSolidAngle=self.correctSolidAngle,
                error_model="azimuthal",
                dummy=-8675309 if self.maskToNan else 0,
                mask=self.mask,
                unit='arcsinh(q.µm)' if self.use_log_ish_binning else 'q_A^-1',
                method=self.integration_method,
            )
        except TypeError as e:
            if 'diffSolidAngle() missing 2 required positional arguments: ' in str(e):
                raise TypeError(
                    'Geometry is incorrect, cannot integrate.\n \n - Do your mask dimensions match your image dimensions? \n - Do you have pixel sizes set that are not zero?\n - Is SDD, beamcenter/poni, and tilt set correctly?'
                ) from e
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
                res = xr.DataArray(
                    [frame.intensity],
                    dims=[stacked_axis, 'q'],
                    coords={
                        'q': ('q', radial_to_save),
                        stacked_axis: (stacked_axis, system_to_integ),
                    },
                    attrs=img.attrs,
                )
                if self.return_sigma:
                    sigma = xr.DataArray(
                        [frame.sigma],
                        dims=[stacked_axis, 'q'],
                        coords={
                            'q': ('q', radial_to_save),
                            stacked_axis: (stacked_axis, system_to_integ),
                        },
                        attrs=img.attrs,
                    )
            except AttributeError:
                res = xr.DataArray(
                    frame.intensity, dims=['q'], coords={'q': radial_to_save}, attrs=img.attrs
                )
                if self.return_sigma:
                    sigma = xr.DataArray(
                        frame.sigma, dims=['q'], coords={'q': radial_to_save}, attrs=img.attrs
                    )
        else:
            try:
                res = xr.DataArray(
                    [frame.intensity],
                    dims=[stacked_axis, 'chi', 'q'],
                    coords={
                        'q': ('q', radial_to_save),
                        'chi': ('chi', frame.azimuthal),
                        stacked_axis: (stacked_axis, system_to_integ),
                    },
                    attrs=img.attrs,
                )  # .transpose(['chi','q',stacked_axis])
                if self.return_sigma:
                    sigma = xr.DataArray(
                        [frame.sigma],
                        dims=[stacked_axis, 'chi', 'q'],
                        coords={
                            'q': ('q', radial_to_save),
                            'chi': ('chi', frame.azimuthal),
                            stacked_axis: (stacked_axis, system_to_integ),
                        },
                        attrs=img.attrs,
                    )  # .transpose(['chi','q',stacked_axis])
            except AttributeError:
                res = xr.DataArray(
                    frame.intensity,
                    dims=['chi', 'q'],
                    coords={'q': radial_to_save, 'chi': frame.azimuthal},
                    attrs=img.attrs,
                )
                if self.return_sigma:
                    sigma = xr.DataArray(
                        frame.sigma,
                        dims=['chi', 'q'],
                        coords={'q': radial_to_save, 'chi': frame.azimuthal},
                        attrs=img.attrs,
                    )
        if self.return_sigma:
            sigma = xr.ones_like(res)
            sigma.data = np.broadcast_to(frame.sigma, sigma.shape)
            res = res.to_dataset(name='I')
            res['dI'] = sigma
        return res

    '''
    legacy index ident code:
     indexes = list(data.indexes.keys())
        indexes.remove('pix_x')
        indexes.remove('pix_y')
        real_indexes = indexes
        for idx in indexes:
            if type(data.indexes[idx]) == pd.core.indexes.multi.MultiIndex:
                for level in data.indexes[idx].names:
                    try:
                        real_indexes.remove(level)
                    except ValueError:
                        pass
        indexes = real_indexes
    '''

    def integrateImageStack_legacy(self, data):
        indexes = list(data.dims)
        indexes.remove('pix_x')
        indexes.remove('pix_y')

        if len(indexes) == 1:
            data_int = data.groupby(indexes[0],squeeze=False).progress_map(self.integrateSingleImage)
        elif len(indexes) == 0:
            data_int = self.integrateSingleImage(data).isel(image_num=0)
        else:
            # some kinda logic to check for existing multiindexes and stack into them appropriately maybe
            data = data.stack({'pyhyper_internal_multiindex': indexes})
            data_int = data.groupby('pyhyper_internal_multiindex', squeeze=False)
            data_int = data_int.progress_map(self.integrateSingleImage)
            data_int = data_int.unstack('pyhyper_internal_multiindex')
            # this is a hack to fix the dimension order in case we are being called as an inner function of a Dask reduction
            if getattr(self, 'expected_dim_order', None) is not None:
                orig_order = data_int.dims
                data_int = data_int.transpose(*self.expected_dim_order)

        return data_int
        # int_stack = img_stack.groupby('system').map_progress(self.integrateSingleImage)
        # PRSUtils.fix_unstacked_dims(int_stack,img_stack,'system',img_stack.attrs['dims_unpacked'])
        # return int_stack

    def integrateImageStack_dask(self, data, chunksize=5):
        # int_stack = img_stack.groupby('system').map(self.integrateSingleImage)
        # return int_stack
        indexes = list(data.dims)
        try:
            indexes.remove('pix_x')
            indexes.remove('pix_y')
        except ValueError:
            pass
        try:
            indexes.remove('qx')
            indexes.remove('qy')
        except ValueError:
            pass

        if len(indexes) == 1:
            if (
                data.__getattr__(indexes[0]).to_pandas().drop_duplicates().shape[0]
                != data.__getattr__(indexes[0]).shape[0]
            ):
                warnings.warn(
                    f'Axis {indexes[0]} contains duplicate conditions.  This is not supported and may not work.  Try adding additional coords to separate image conditions',
                    stacklevel=2,
                )

            dim_to_chunk = indexes[0]
        else:
            # some kinda logic to check for existing multiindexes and stack into them appropriately maybe
            if 'energy' in indexes:
                dim_to_chunk = 'energy'
            else:
                dim_to_chunk = indexes[0]
            # this probably should check which is the longest?  Shortest?  and chunk that.
        print(f'chunking on {dim_to_chunk}')

        '''
            data = data.stack({'pyhyper_internal_multiindex':indexes})
            if data.pyhyper_internal_multiindex.to_pandas().drop_duplicates().shape[0] != data.pyhyper_internal_multiindex.shape[0]:
                warnings.warn('Your index set contains duplicate conditions.  This is not supported and may not work.  Try adding additional coords to separate image conditions',stacklevel=2)
                
            fake_image_to_process = data.isel(**{'pyhyper_internal_multiindex':0},drop=False)
            indexes=['pyhyper_internal_multiindex']
            dim_to_chunk = 'pyhyper_internal_multiindex'
        '''

        fake_image_to_process = data.isel(**{dim_to_chunk: 0}, drop=False)
        data = data.chunk({dim_to_chunk: chunksize})
        coord_dict = {}
        shape = tuple([])
        demo_integration = self.integrateSingleImage(fake_image_to_process)
        coord_dict.update({'chi': demo_integration.chi, 'q': demo_integration.q})
        npts_q = len(demo_integration.q)

        order_list = []
        for idx in indexes:
            order_list.append(idx)
            coord_dict[idx] = data.indexes[idx]
            shape = shape + tuple([len(data.indexes[idx])])
        shape = shape + (360, npts_q)

        desired_order_list = order_list + ['chi', 'q']
        coord_dict_sorted = {k: coord_dict[k] for k in desired_order_list}

        template = xr.DataArray(np.empty(shape), coords=coord_dict_sorted)

        print(demo_integration.dims)
        if 'image_num' in demo_integration.dims:
            template = template.transpose(
                *[item if item != 'image_num' else dim_to_chunk for item in demo_integration.dims]
            )
        elif dim_to_chunk not in demo_integration.dims:
            template = template.transpose(dim_to_chunk, *demo_integration.dims)
        template = template.chunk({indexes[0]: chunksize})
        '''
        try:
            print(template)
            print(template.indexes)
            print(template.pyhyper_internal_multiindex)
            print(data)
            print(data.indexes)
            print(data.pyhyper_internal_multiindex)
            print(data.pyhyper_internal_multiindex==template.pyhyper_internal_multiindex)
        except AttributeError:
            pass
            
        '''
        self.expected_dim_order = template.dims
        print(f'set expected dim order to {self.expected_dim_order}')
        integ_fly = data.map_blocks(self.integrateImageStack_legacy, template=template)
        if dim_to_chunk == 'pyhyper_internal_multiindex':
            integ_fly = integ_fly.unstack('pyhyper_internal_multiindex')
        return integ_fly 

    def integrateImageStack(self, img_stack, method=None, chunksize=None):
        ''' '''

        if (self.use_chunked_processing and method is None) or method == 'dask':
            func_args = {}
            if chunksize is not None:
                func_args['chunksize'] = chunksize
            return self.integrateImageStack_dask(img_stack, **func_args)
        elif (method is None) or method == 'legacy':
            return self.integrateImageStack_legacy(img_stack)
        else:
            raise NotImplementedError(f'unsupported integration method {method}')

    def loadPolyMask(self, maskpoints=[], **kwargs):
        '''
        loads a polygon mask from a list of polygon points

        Args:
        (list) maskpoints: a list of lists of points, e.g.
                [
                    [ #begin polygon 1
                        [0,0],[0,10],[10,10],[10,0]
                    ],
                    [ #later polygons]
                ]
        (tuple) maskshape: (x,y) dimensions of mask to create
                if not passed, will assume that the maximum point is included in the mask
        '''
        points = maskpoints
        xs = []
        ys = []
        for polygon in points:
            x, y = zip(*polygon)
            xs += x
            ys += y
            if 'maskshape' in kwargs:
                shape = kwargs['maskshape']
            else:
                shape = (math.ceil(max(xs)), math.ceil(max(ys)))
            image = np.zeros(shape)
            for polygon in points:
                image += draw.polygon2mask(shape, polygon)
            image[image > 1] = 1
        boolmask = np.invert(image.astype(bool))
        print(f"Created mask with dimensions {str(np.shape(boolmask))}")
        self.mask = boolmask

    def loadImageMask(self, **kwargs):
        '''
        loads a mask from a generic image

        Args:
            (pathlib.Path or String) maskpath: path to load
            (bool) maskrotate: rotate mask using np.flipud(np.rot90(mask))
        '''

        im = Image.open(kwargs['maskpath'])
        image = np.array(im)
        if 'maskrotate' in kwargs:
            if kwargs['maskrotate']:
                image = np.flipud(np.rot90(image))
        boolmask = np.invert(image.astype(bool))
        print(f"Imported mask with dimensions {str(np.shape(boolmask))}")
        self.mask = boolmask

    def loadEdfMask(self, **kwargs):
        '''
        Loads an edf-format mask (probably from pyFAI.calib2?).

        Args:
            filetoload (pathlib.Path or string): path to edf format mask
        '''
        filetoload = kwargs['maskpath']
        self.mask = fabio.open(filetoload).data

    def loadNikaMask(self, rotate_image = True, **kwargs):

        '''
        Loads a Nika-generated HDF5 or tiff mask and converts it to an array that matches the local conventions.

        Args:
            filetoload (pathlib.Path or string): path to hdf5/tiff format mask from Nika.
            rotate_image (bool, default True): rotate image as should work
        '''
        mask = None
        filetoload = kwargs['maskpath']

        if 'h5' in str(filetoload) or 'hdf' in str(filetoload):
            type = 'h5'
            maskhdf = h5py.File(filetoload, 'r')
            mask = maskhdf['M_ROIMask']

        elif 'tif' in str(filetoload):
            type = 'tif'
            mask = plt.imread(filetoload)
        else:
            warnings.warn('Unsupported mask type...', stacklevel=2)
        if 'rotate_image' in kwargs:
            if kwargs['rotate_image']:
                mask = np.flipud(np.rot90(mask))
        boolmask = np.invert(mask.astype(bool)[:])
        print(f"Imported or created mask with dimensions {str(np.shape(boolmask))}")
        self.mask = boolmask

    def loadPyHyperMask(self, **kwargs):
        '''
        Loads a mask json file saved by PyHyper's drawMask routines.

        Args:
            (pathlib.Path or string) maskpath: path to load json file from


        '''
        with open(kwargs['maskpath'], 'r') as f:
            strlist = json.load(f)
        # print(strlist)
        dflist = []
        for item in strlist:
            dflist.append(pd.read_json(item))
        # print(dflist)
        pyhyperlist = []
        for shape in dflist:
            pyhyper_shape = []
            for index, xval in enumerate(shape.x):
                yval = shape.y[index]
                pyhyper_shape.append([xval, yval])
            pyhyperlist.append(pyhyper_shape)
        self.loadPolyMask(maskpoints=pyhyperlist,**kwargs)

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
        try:
            self.energy = float(raw_xr.energy)
        except TypeError:
            pass

        if self.mask is None:
            self.mask = np.zeros((len(raw_xr.pix_y), len(raw_xr.pix_x)))
            warnings.warn(
                f'Since mask was none, creating an empty mask with shape {self.mask.shape}',
                stacklevel=2,
            )
            
        if hasattr(raw_xr.energy, '__iter__'):  # this is an iterable, not a single number
            self.energy = raw_xr.energy[0]
        else:
            self.energy = raw_xr.energy

        self.recreateIntegrator()

    def calibrationFromPoniFile(self, ponifile):

        '''
        Sets calibration from a pyFAI poni-file

        Args:
            ponifile (str or Pathlib.path): a pyFAI poni file containing the geometry
            raw_xr (raw format xarray): optional, raw xr with correct pixel dimensions 
                                        for creating an empty mask if necessary
        '''
        ponifile = PoniFile(data=str(ponifile))
        self.dist = ponifile._dist
        self.poni1 = ponifile._poni1
        self.poni2 = ponifile._poni2
        self.rot1 = ponifile._rot1
        self.rot2 = ponifile._rot2
        self.rot3 = ponifile._rot3
        self.wavelength = ponifile._wavelength

        self.pixel1 = ponifile.detector.pixel1
        self.pixel2 = ponifile.detector.pixel2

        self.recreateIntegrator()

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

    def recreateIntegrator(self):
        '''
        recreate the integrator, after geometry change
        '''
        self.integrator = azimuthalIntegrator.AzimuthalIntegrator(
            self.dist,
            self.poni1,
            self.poni2,
            self.rot1,
            self.rot2,
            self.rot3,
            pixel1=self.pixel1,
            pixel2=self.pixel2,
            wavelength=self.wavelength,
        )


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
            warnings.warn(
                'x pixel size is 0, cannot set beam center, fix pixel size first', stacklevel=2
            )
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
            warnings.warn(
                'y pixel size is 0, cannot set beam center, fix pixel size first', stacklevel=2
            )
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
