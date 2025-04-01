"""
File to:
    1. Use pygix to apply the missing wedge Ewald's sphere correction & convert to q-space
    2. Generate 2D plots of Qz vs Qxy corrected detector images
    3. Generate 2d plots of Q vs Chi images, with the option to apply the sin(chi) correction
    4. etc.
"""

# Imports
import xarray as xr
import numpy as np
import pathlib
from typing import Union, Tuple
from tqdm.auto import tqdm 
import warnings
from PyHyperScattering.PFGeneralIntegrator import PFGeneralIntegrator
try:
    import pygix  # type: ignore
except ImportError:
    warnings.warn('Unable to load optional dependency pygix, needed for grazing.  Install with pyhyperscattering[grazing] if needed.',stacklevel=2)

class PGGeneralIntegrator(PFGeneralIntegrator):
    """ 
    Integrator for GIWAXS data based on pygix.

    Inherits from PFGeneralIntegrator so as to benefit from its utility methods for mask loading, etc.


    """
    def __init__(self, 
                inplane_config: str = 'q_xy',
                sample_orientation: int = 3,
                incident_angle = 0.12,
                tilt_angle = 0.0,
                output_space = 'recip',
                 **kwargs):
        """
        PyGIX-backed Grazing Incidence Integrator

        Inputs:
        inplane_config (str, default 'q_xy'): The q axis to be considered in-plane.
        sample_orientation (int, default 3): the sample orientation relative to the detector.  see PyGIX docs.
        incident_angle (float, default 0.12): the incident angle, can also be set with .incident_angle = x.xx
        tilt_angle (float, default 0): sample tilt angle, can also be set with .tilt_angle = x.xx
        output_space (str, 'recip' or 'caked'): whether to produce reciprocal space (q_xy vs q_z, e.g.) data
            or 'caked' style data as with PF series integrators (|q| vs chi)

        See docstring for PFGeneralIntegrator for all geometry kwargs, which work here.

        """
        self.inplane_config = inplane_config
        self.sample_orientation = sample_orientation
        self.incident_angle = incident_angle
        self.tilt_angle = tilt_angle
        self.output_space = output_space
        super().__init__(**kwargs) # all other setup is done by the recreateIntegrator() function and superclass

    # def load_mask(self, da): has been superseded by PFGeneralIntegrator's methods
    
    # need to override this to make the PyGIX object
    def recreateIntegrator(self):
        '''
        recreate the integrator, after geometry change
        '''
        self.integrator = pygix.Transform(dist= self.dist,
                        poni1 = self.poni1, 
                        poni2 = self.poni2, 
                        rot1 = self.rot1,
                        rot2 = self.rot2, 
                        rot3 = self.rot3,
                        pixel1 = self.pixel1, 
                        pixel2 = self.pixel2, 
                        wavelength = self.wavelength,
                        useqx=True, 
                        sample_orientation = self.sample_orientation, 
                        incident_angle = self.incident_angle,
                        tilt_angle = self.tilt_angle)


    def integrateSingleImage(self, da):
        """
        Converts raw GIWAXS detector image to q-space data. Returns two DataArrays, Qz vs Qxy & Q vs Chi
        
        Inputs: Raw GIWAXS DataArray
        Outputs: Cartesian & Polar DataArrays
        """

        # Initialize pygix transform object - moved to recreateIntegrator
        
        # the following index stack/unstack code copied from PFGeneralIntegrator
        if(da.ndim>2):
            img_to_integ = np.squeeze(da.values)
        else:
            img_to_integ = da.values
        
        if self.mask is None:
            warnings.warn(f'No mask defined. Creating an empty mask with dimensions {img_to_integ.shape}.',stacklevel=2)
            self.mask = np.zeros_like(img_to_integ)
        assert np.shape(self.mask)==np.shape(img_to_integ),f'Error!  Mask has shape {np.shape(self.mask)} but you are attempting to integrate data with shape {np.shape(img_to_integ)}.  Try changing mask orientation or updating mask.'
        stacked_axis = list(da.dims)
        stacked_axis.remove('pix_x')
        stacked_axis.remove('pix_y')
        if len(stacked_axis)>0:
            assert len(stacked_axis)==1, f"More than one dimension left after removing pix_x and pix_y, I see {stacked_axis}, not sure how to handle"
            stacked_axis = stacked_axis[0]
            #print(f'looking for {stacked_axis} in {img[0].indexes} (indexes), it has dims {img[0].dims} and looks like {img[0]}')
            if(da.__getattr__(stacked_axis).shape[0]>1):
                system_to_integ = da[0].indexes[stacked_axis]
                warnings.warn(f'There are two images for {da.__getattr__(stacked_axis)}, I am ONLY INTEGRATING THE FIRST.  This may cause the labels to be dropped and the result to need manual re-tagging in the index.',stacklevel=2)
            else:
                system_to_integ = da.indexes[stacked_axis]
                
        else:
            stacked_axis = 'image_num'
            system_to_integ = [0]

        # Cartesian 2D plot transformation
        if self.output_space == 'recip':
            recip_data, qxy, qz = self.integrator.transform_reciprocal(img_to_integ,
                                                                       method='bbox',
                                                                       unit='A',
                                                                       mask=self.mask,
                                                                       correctSolidAngle=self.correctSolidAngle)
        
            out_da = xr.DataArray(data=recip_data,
                                    dims=['q_z', self.inplane_config],
                                    coords={
                                        'q_z': ('q_z', qz, {'units': '1/Å'}),
                                        self.inplane_config: (self.inplane_config, qxy, {'units': '1/Å'})
                                    },
                                    attrs=da.attrs)
        elif self.output_space == 'caked':
            caked_data, qr, chi = self.integrator.transform_image(img_to_integ, 
                                                                  process='polar',
                                                                  method = 'bbox',
                                                                  unit='q_A^-1',
                                                                  mask=self.mask,
                                                                  correctSolidAngle=True)

            out_da = xr.DataArray(data=caked_data,
                                dims=['chi', 'qr'],
                                coords={
                                    'chi': ('chi', chi, {'units': '°'}),
                                    'qr': ('qr', qr, {'units': '1/Å'})
                                },
                                attrs=da.attrs)
            out_da.attrs['inplane_config'] = self.inplane_config

        # Preseve any existing dimension if it is in the dataarray, for stacking purposes
        if stacked_axis in da.coords:
            out_da = out_da.expand_dims(dim={stacked_axis: 1}).assign_coords({stacked_axis: np.array(system_to_integ)})
            # out_da = out_da.expand_dims(dim={stacked_axis: 1})


        return out_da


    def __str__(self):
        return f"PyGIX general integrator wrapper SDD = {self.dist} m, poni1 = {self.poni1} m, poni2 = {self.poni2} m, rot1 = {self.rot1} rad, rot2 = {self.rot2} rad"


# The below function is located as a method in the CMSGIWAXS class in IntegrationUtils.py
def single_images_to_dataset(files, loader, integrator):
    """
    Function that takes a subscriptable object of filepaths corresponding to raw GIWAXS
    beamline data, loads the raw data into an xarray DataArray, generates pygix-transformed 
    cartesian and polar DataArrays, and creates 3 corresponding xarray Datasets 
    containing a DataArray per sample. 
    The raw dataarrays must contain the attributes 'scan_id' and 'incident_angle'

    Inputs: files: indexable object containing pathlib.Path filepaths to raw GIWAXS data
            loader: custom PyHyperScattering CMSGIWAXSLoader object, must return DataArray
            integrator: instance of PGGeneralIntegrator object defined above, takes raw 
                         dataarray and returns processed data in reciprocal space (recip or caked)

    Outputs: 2 Datasets: raw & reciprocal space (cartesian or polar based on integrator object)
    """
    # Select the first element of the sorted set outside of the for loop to initialize the xr.DataSet
    DA = loader.loadSingleImage(files[0])
    assert 'scan_id' in DA.attrs.keys(), "'scan_id' is a required attribute to use this function"

    # Update incident angle per sample:
    assert 'incident_angle' in DA.attrs.keys(), "'incident_angle' is a required attribute to use this function"
    integrator.incident_angle = float(DA.incident_angle[2:])

    # Integrate single image
    integ_DA = integrator.integrateSingleImage(DA)

    # Save coordinates for interpolating other dataarrays 
    integ_coords = integ_DA.coords

    # Create a DataSet, each DataArray will be named according to it's scan id
    raw_DS = DA.to_dataset(name=DA.scan_id)
    integ_DS = integ_DA.to_dataset(name=DA.scan_id)

    # Populate the DataSet with 
    for filepath in tqdm(files[1:], desc=f'Transforming Raw Data'):
        DA = loader.loadSingleImage(filepath)
        integ_DA = integrator.integrateSingleImage(DA)
        
        integ_DA = integ_DA.interp(integ_coords)

        raw_DS[f'{DA.scan_id}'] = DA
        integ_DS[f'{DA.scan_id}'] = integ_DA

    return raw_DS, integ_DS


