from PIL import Image
import os
import pathlib
import xarray as xr
import pandas as pd
import datetime
import warnings
import json
#from pyFAI import azimuthalIntegrator
import numpy as np

try:
    os.environ["TILED_SITE_PROFILES"] = '/nsls2/software/etc/tiled/profiles'
    from tiled.client import from_profile
    from databroker.queries import RawMongo
except:
    print('Imports failed.  Are you running on a machine with proper libraries for databroker, tiled, etc.?')
    
import copy


class SST1RSoXSDB:
    '''
    Loader for bluesky run xarrays form NSLS-II SST1 RSoXS instrument
    

    '''
    file_ext = ''
    md_loading_is_quick = True
    pix_size_1 = 0.06
    pix_size_2 = 0.06
    
    

    def __init__(self,corr_mode=None,user_corr_fun=None,dark_subtract=True,dark_pedestal=0,exposure_offset=0,catalog=None,catalog_kwargs={}):
        '''
            Args:
                corr_mode (str): origin to use for the intensity correction.  Can be 'expt','i0','expt+i0','user_func','old',or 'none'
                user_corr_func (callable): takes the header dictionary and returns the value of the correction.
                dark_pedestal (numeric): value to add to the whole image before doing dark subtraction, to avoid non-negative values.
                exposure_offset (numeric): value to add to the exposure time.
                catalog (DataBroker Catalog): overrides the internally-set-up catalog with a version you provide
                catalog_kwargs (dict): kwargs to be passed to a from_profile catalog generation script.  For example, you can ask for Dask arrays here.
        '''
        if corr_mode == None:
            warnings.warn("Correction mode was not set, not performing *any* intensity corrections.  Are you sure this is "+
                          "right? Set corr_mode to 'none' to suppress this warning.")
            self.corr_mode = 'none'
        else:
            self.corr_mode = corr_mode
        
        if catalog is None:
            self.c = from_profile('rsoxs',**catalog_kwargs)
        else:
            self.c = catalog
        self.dark_subtract=dark_subtract
        self.dark_pedestal=dark_pedestal
        self.exposure_offset=exposure_offset
        
    # def loadFileSeries(self,basepath):
    #     try:
    #         flist = list(basepath.glob('*primary*.tiff'))
    #     except AttributeError:
    #         basepath = pathlib.Path(basepath)
    #         flist = list(basepath.glob('*primary*.tiff'))
    #     print(f'Found {str(len(flist))} files.')
    #
    #     out = xr.DataArray()
    #     for file in flist:
    #         single_img = self.loadSingleImage(file)
    #         out = xr.concat(out,single_img)
    #
    #     return out
    
    def runSearch(self,**kwargs):
        '''
        Search the catalog using given commands.

        Args:
            **kwargs: passed through to the RawMongo search method of the catalog.

        Returns:
            result (obj): a catalog result object

        '''
        q = RawMongo(**kwargs)
        return self.c.search(q)
    
    def loadRun(self,run,dims,coords={}):
        '''
        Loads a run entry from a catalog result into a raw xarray.

        Args:
            run (DataBroker result): a single run from BlueSky
            dims (list): list of dimensions you'd like in the resulting xarray.  See list of allowed dimensions in documentation.
            coords (dict): user-supplied dimensions, see syntax examples in documentation.

        Returns:
            raw (xarray): raw xarray containing your scan in PyHyper-compliant format

        '''
        md = self.loadMd(run)  

        

        data = run['primary']['data'][md['detector']+'_image'] 
        if self.dark_subtract:
            dark = run['dark']['data'][md['detector']+'_image'].mean('time') #@TODO: change to correct dark indexing
            image = data - dark - self.dark_pedestal
        else:
            image = data - self.dark_pedestal

        if self.corr_mode != 'none':
            warnings.warn('corrections other than none are not supported at the moment')


        dims_to_join = []
        dim_names_to_join = []

        for dim in dims:
            try:
                test = len(md[dim])
                dims_to_join.append(md[dim].data)
                dim_names_to_join.append(dim)
            except TypeError:
                dims_to_join.append(np.ones(run.start['num_points'])*md[dim])
                dim_names_to_join.append(dim)

        for key,val in coords.items():
            dims_to_join.append(val)
            dim_names_to_join.append(key)

        index = pd.MultiIndex.from_arrays(
                dims_to_join,
                names=dim_names_to_join)

        retxr = image.squeeze('dim_0').rename({'dim_1':'pix_y','dim_2':'pix_x'}).assign_coords(time=index).rename({'time':'system'})#,md['detector']+'_image':'intensity'})
        
        #this is needed for holoviews compatibility, hopefully does not break other features.
        retxr = retxr.assign_coords({'pix_x':np.arange(0,len(retxr.pix_x)),'pix_y':np.arange(0,len(retxr.pix_y))})
        
        retxr.attrs.update(md)
        return retxr


    def peekAtMd(self,run):
        return self.loadMd(run)


    def loadMd(self,run):
        '''
        return a dict of metadata entries from the databroker run xarray


        '''
        md = {}

        # items coming from the start document
        start = run.start
        
        meas_time =datetime.datetime.fromtimestamp(run.start['time'])
        md['meas_time']=meas_time
        md['sample_name'] = start['sample_name']
        if start['RSoXS_Config'] == 'SAXS':
            md['rsoxs_config'] = 'saxs'
                # discrepency between what is in .json and actual
            if (meas_time > datetime.datetime(2020,12,1)) and (meas_time < datetime.datetime(2021,1,15)):
                md['beamcenter_x'] = 489.86
                md['beamcenter_y'] = 490.75
                md['sdd'] = 521.8
            elif (meas_time > datetime.datetime(2020,11,16)) and (meas_time < datetime.datetime(2020,12,1)):
                md['beamcenter_x'] = 371.52
                md['beamcenter_y'] = 491.17
                md['sdd'] = 512.12
            else:
                md['beamcenter_x'] = run.start['RSoXS_SAXS_BCX']
                md['beamcenter_y'] = run.start['RSoXS_SAXS_BCY']
                md['sdd'] = run.start['RSoXS_SAXS_SDD']

        elif start['RSoXS_Config'] == 'WAXS':
            md['rsoxs_config'] = 'waxs'
            if (meas_time > datetime.datetime(2020,11,16)) and (meas_time < datetime.datetime(2021,1,15)):
                md['beamcenter_x'] = 400.46
                md['beamcenter_y'] = 530.99
                md['sdd'] = 38.745
            else:
                md['beamcenter_x'] = run.start['RSoXS_WAXS_BCX'] # 399 #
                md['beamcenter_y'] = run.start['RSoXS_WAXS_BCY'] # 526
                md['sdd'] = run.start['RSoXS_WAXS_SDD']

        else:
            md['rsoxs_config'] == 'unknown'
            warnings.warn(f'RSoXS_Config is neither SAXS or WAXS. Looks to be {start["RSoXS_Config"]}.  Might want to check that out.')

        if md['rsoxs_config']=='saxs':
            md['detector'] = 'Small Angle CCD Detector'
        elif md['rsoxs_config']=='waxs':
            md['detector'] = 'Wide Angle CCD Detector'
        else:
            warnings.warn(f'Cannot auto-hint detector type without RSoXS config.')


        # items coming from baseline  
        baseline = run['baseline']['data']

        # items coming from primary
        primary = run['primary']['data']

        md_lookup = {
            'sam_x':'RSoXS Sample Outboard-Inboard',
            'sam_y':'RSoXS Sample Up-Down',
            'sam_z':'RSoXS Sample Downstream-Upstream',
            'sam_th':'RSoXS Sample Rotation',
            'energy':'en_energy_setpoint',
            'polarization':'en_polarization_setpoint',
            'exposure':md['detector']+'_cam_acquire_time'
        }

        for phs,rsoxs in md_lookup.items():
            try:
                md[phs] = primary[rsoxs]
            except KeyError:
                try:
                    blval = baseline[rsoxs]
                    md[phs] = blval.mean().data
                    if blval.var() > 0:
                        warnings.warn(f'While loading {rsoxs} to infill metadata entry for {phs}, found beginning and end values unequal: {baseline[rsoxs]}.  It is possible something is messed up.')
                except KeyError:
                    warnings.warn(f'Could not find {rsoxs} in either baseline or primary.  Needed to infill value {phs}.  Setting to None.')
                    md[phs] = None
                    
        md['wavelength'] = 1.239842e-6 / md['energy']
        #md['sampleid'] = scan_id@todo this should be easy

        md['dist'] = md['sdd'] / 1000

        md['pixel1'] = self.pix_size_1 / 1000
        md['pixel2'] = self.pix_size_2 / 1000

        md['poni1'] = md['beamcenter_y'] * md['pixel1']
        md['poni2'] = md['beamcenter_x'] * md['pixel2']

        md['rot1'] = 0
        md['rot2'] = 0
        md['rot3'] = 0

        md.update(run.metadata)
        return md

    def loadSingleImage(self,filepath,coords=None, return_q=False):
        '''
            DO NOT USE

            This function is preserved as reference for the qx/qy loading conventions.

            NOT FOR ACTIVE USE.  DOES NOT WORK.
        '''
        img = Image.open(filepath)

        headerdict = self.loadMd(filepath)
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


        # # step 2: dark subtraction
        # try:
        #     darkimg = self.darks[headerdict['EXPOSURE']]
        # except KeyError:
        #     warnings.warn(f"Could not find a dark image with exposure time {headerdict['EXPOSURE']}.  Using zeros.")
        #     darkimg = np.zeros_like(img)

        # img = (img-darkimg+self.dark_pedestal)/corr
        if return_q:
            qpx = 2*np.pi*60e-6/(headerdict['sdd']/1000)/(headerdict['wavelength']*1e10)
            qx = (np.arange(1,img.size[0]+1)-headerdict['beamcenter_x'])*qpx
            qy = (np.arange(1,img.size[1]+1)-headerdict['beamcenter_y'])*qpx
            # now, match up the dims and coords
            return xr.DataArray(img,dims=['qy','qx'],coords={'qy':qy,'qx':qx},attrs=headerdict)
        else:
            return xr.DataArray(img,dims=['pix_x','pix_y'],attrs=headerdict)


