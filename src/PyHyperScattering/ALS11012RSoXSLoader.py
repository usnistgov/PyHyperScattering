from PyHyperScattering.FileLoader import FileLoader
import os
import xarray as xr
import pandas as pd
import numpy as np
import warnings
import re
try:
    from astropy.io import fits
except ImportError:
    warnings.warn('Could not import astropy.io.fits, needed for ALS 11.0.1.2 RSoXS loading.  Is this dependency installed?')


class ALS11012RSoXSLoader(FileLoader):
    '''
    Loader for FITS files from the ALS 11.0.1.2 RSoXS instrument


    Additional requirement: astropy, for FITS file loader


    Usage is mainly via the inherited function integrateImageStack from FileLoader

    '''
    file_ext = '(.*?).fits'
    md_loading_is_quick = True
    
    
    def __init__(self,corr_mode=None,user_corr_func=None,dark_pedestal=0,exposure_offset=0.002):
        '''
        Args:
            corr_mode (str): origin to use for the intensity correction.  Can be 'expt','i0','expt+i0','user_func','old',or 'none'
            user_corr_func (callable): that takes the header dictionary and returns the value of the correction.
            dark_pedestal (numeric): number to add to the whole image before doing dark subtraction, to avoid non-negative values.
            exposure_offset (numeric): value to add to the exposure time.  Measured at 2ms with the piezo shutter in Dec 2019 by Jacob Thelen, NIST
        
        '''
        if corr_mode == None:
            warnings.warn("Correction mode was not set, not performing *any* intensity corrections.  Are you sure this is "+ 
                          "right? Set corr_mode to 'none' to suppress this warning.")
            self.corr_mode = 'none'
        else:
            self.corr_mode = corr_mode
        self.dark_pedestal = dark_pedestal
        self.user_corr_func = user_corr_func
        self.exposure_offset = exposure_offset
        self.darks = {}
    
    def loadDarks(self,basepath,dark_base_name):
        '''
        Load a series of dark images as a function of exposure time, to be subtracted from subsequently-loaded data.

        Args:
            basepath (str or Path): path to load images from
            dark_base_name (str): str that must be in file for file to be a dark
        '''
        for file in os.listdir(basepath):
            if dark_base_name in file:
                darkimage = fits.open(basepath+file)
                assert darkimage[0].header['CCD Shutter Inhibit']==1,"CCD Shutter was not inhibited for image "+file+"... probably not a dark."

                exptime = round(darkimage[0].header['EXPOSURE'],2)

                self.darks[exptime] = darkimage[2].data

                
    def loadSampleSpecificDarks(self,basepath,file_filter='',file_skip='donotskip',md_filter={}):
        '''
        load darks matching a specific sample metadata

        Used, e.g., to load darks taken at a time of measurement in order to improve the connection between the dark and sample data.

        Args:
            basepath (str): path to load darks from
            file_filter (str): string that must be in each file name
            file_skip (str): string that, if in file name, means file should be skipped.
            md_filter (dict): dict of required metadata values.  this will be appended with dark images only, no need to put that here.
        '''
        
        md_filter.update({'CCD Shutter Inhibit':1})
        
        for file in os.listdir(basepath):
            if (re.match(self.file_ext,file) is not None) and file_filter in file and file_skip not in file:
                if self.md_loading_is_quick:
                    #if metadata loading is quick, we can just peek at the metadata and decide what to do
                    md = self.peekAtMd(basepath+file)
                    img = None
                else:
                        input_image = fits.open(basepath+file)
                        md = self.normalizeMetadata(dict(zip(input_image[0].header.keys(),input_image[0].header.values())))
                        img = input_image[2].data
                load_this_image = True
                for key,val in md_filter.items():
                    if md[key] != md_filter[key]:
                        load_this_image = False
                        #print(f'Not loading {file}, expected {key} to be {val} but it was {md[key]}')
                if load_this_image:
                    if img == None:
                        input_image = fits.open(basepath+file)
                        img = input_image[2].data
                    print(f'Loading dark for {md["EXPOSURE"]} from {file}')
                    exptime = md['EXPOSURE']
                    self.darks[exptime] = img
    def loadSingleImage(self,filepath,coords=None):
        '''
        THIS IS A HELPER FUNCTION, mostly - should not be called directly unless you know what you are doing


        Load a single image from filepath and return a single-image, raw xarray, performing dark subtraction if so configured.

        '''
        input_image = fits.open(filepath)
        headerdict =  self.normalizeMetadata(dict(zip(input_image[0].header.keys(),input_image[0].header.values())))
        img = input_image[2].data
        # two steps in this pre-processing stage: 
        #     (1) get and apply the right scalar correction term to the image
        #     (2) find and subtract the right dark
        if coords != None:
            headerdict.update(coords)
        
        #step 1: correction term
        
        if self.corr_mode == 'expt':
            corr = headerdict['exposure'] #(headerdict['AI 3 Izero']*expt)
        elif self.corr_mode == 'i0':
            corr = headerdict['AI 3 Izero']
        elif self.corr_mode == 'expt+i0':
            corr = headerdict['exposure'] * headerdict['AI 3 Izero']
        elif self.corr_mode == 'user_func':
            corr = self.user_corr_func(headerdict)
        elif self.corr_mode == 'old':
            corr = headerdict['AI 6 BeamStop'] * 2.4e10/ headerdict['Beamline Energy'] / headerdict['AI 3 Izero']  
            #this term is a mess...  @TODO check where it comes from	
        else:
            corr = 1
            
        if(corr<0):
            warnings.warn(f'Correction value is negative: {corr} with headers {headerdict}.')
            corr = abs(corr)

        
        # step 2: dark subtraction
        try:
            darkimg = self.darks[headerdict['EXPOSURE']]
        except KeyError:
            warnings.warn(f"Could not find a dark image with exposure time {headerdict['EXPOSURE']}.  Using zeros.")
            darkimg = np.zeros_like(img)

        img = (img-darkimg+self.dark_pedestal)/corr
        
        # now, match up the dims and coords
        return xr.DataArray(img,dims=['pix_x','pix_y'],attrs=headerdict)
        
    def peekAtMd(self,file):
        '''
        load the header/metadata without opening the corresponding image

        Args:
            file (str): fits file from which to load metadata
        '''
        input_image = fits.open(file)
        headerdict =  self.normalizeMetadata(dict(zip(input_image[0].header.keys(),input_image[0].header.values())))
        return headerdict
    
    
    def normalizeMetadata(self,headerdict):
        '''
        convert the local metadata terms in headerdict to standard nomenclature

        Args:
            headerdict (dict): the header returned by the file loader
        '''
        headerdict['EXPOSURE'] = round(headerdict['EXPOSURE'],4)
        headerdict['exposure'] = headerdict['EXPOSURE']+self.exposure_offset
        headerdict['energy'] = round(headerdict['Beamline Energy'],2)
        headerdict['polarization'] = round(headerdict['EPU Polarization'],0)
        headerdict['sam_x'] = headerdict['Sample X']
        headerdict['sam_y'] = headerdict['Sample Y']
        headerdict['sam_z'] = headerdict['Sample Z']
        headerdict['sam_th'] = headerdict['Sample Theta']
        headerdict['sampleid'] = headerdict['Sample Number']
        headerdict['wavelength'] = 1.239842e-6 / headerdict['energy']
        headerdict['det_x'] = round(headerdict['CCD X'],2)
        headerdict['det_y'] = round(headerdict['CCD Y'],2)
        headerdict['det_th'] = round(headerdict['CCD Theta'],2)
        return headerdict
        