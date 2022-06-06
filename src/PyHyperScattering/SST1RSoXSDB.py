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
import pandas as pd
from tqdm.auto import tqdm
import scipy.ndimage
import asyncio
import time



try:
    os.environ["TILED_SITE_PROFILES"] = '/nsls2/software/etc/tiled/profiles'
    from tiled.client import from_profile
    from databroker.queries import RawMongo, Key, FullText, Contains,Regex
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
                          "right? Set corr_mode to 'none' to suppress this warning.",stacklevel=2)
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
 
    def summarize_run(self,proposal=None,saf=None,user=None,institution=None,project=None,sample=None,plan=None):
            '''
            Returns a Pandas dataframe with a summary of runs matching a set of search criteria.

            Args:
                proposal, saf, user, institution (str or None): if str, adds an exact match search on the appropriate parameter to the set
                project,sample,plan (str or None): if str, adds a regex match search on the appropriate parameter to the set.
                    example: project='*Liquid*' matches 'Liquid','Liquids','Liquid-RSoXS')

            Returns:
                pd.Dataframe containing the results of the search.
            '''
            catalog = self.c
            if proposal is not None:
                catalog = catalog.search(Key('proposal_id')==proposal)
            if saf is not None:
                catalog = catalog.search(Key('saf_id')==saf)
            if user is not None:
                catalog = catalog.search(Key('user_name')==user)
            if institution is not None:
                catalog = catalog.search(Key('institution')==institution)
            if project is not None:
                catalog = catalog.search(Regex("project_name",project))
            if sample is not None:
                catalog = catalog.search(Regex('sample_name',sample))
            if plan is not None:
                catalog = catalog.search(Regex('plan_name',plan))
            cat = catalog
            #print(cat)
            #print('#    scan_id        sample_id           plan_name')
            scan_ids = []
            sample_ids = []
            plan_names = []
            start_times = []
            npts = []
            uids = []
            for num,entry in tqdm((enumerate(cat)),total=len(cat)):
                doc = catalog[entry].start
                scan_ids.append(doc["scan_id"])
                sample_ids.append(doc["sample_id"])
                plan_names.append(doc["plan_name"])
                uids.append(doc["uid"])
                try:
                    npts.append(catalog[entry].stop['num_events']['primary'])
                except KeyError:
                    npts.append(0)
                start_times.append(doc["time"])
                #do_list_append(catalog[entry],scan_ids,sample_ids,plan_names,uids,npts,start_times)
                #print(f'{num}  {cat[entry].start["scan_id"]}  {cat[entry].start["sample_id"]} {cat[entry].start["plan_name"]}')
            return pd.DataFrame(list(zip(scan_ids,sample_ids,plan_names,npts,uids,start_times)),
                       columns =['scan_id', 'sample_id','plan_name','npts','uid','time'])

    def background(f):
        def wrapped(*args, **kwargs):
            return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

        return wrapped
    @background
    def do_list_append(run,scan_ids,sample_ids,plan_names,uids,npts,start_times):
        doc = run.start
        scan_ids.append(doc["scan_id"])
        sample_ids.append(doc["sample_id"])
        plan_names.append(doc["plan_name"])
        uids.append(doc["uid"])
        try:
            npts.append(run.stop['num_events']['primary'])
        except KeyError:
            npts.append(0)
        start_times.append(doc["time"])
    def loadRun(self,run,dims=None,coords={},return_dataset=False):
        '''
        Loads a run entry from a catalog result into a raw xarray.

        Args:
            run (DataBroker result): a single run from BlueSky
            dims (list): list of dimensions you'd like in the resulting xarray.  See list of allowed dimensions in documentation.  If not set or None, tries to auto-hint the dims from the RSoXS plan_name.
            coords (dict): user-supplied dimensions, see syntax examples in documentation.
            return_dataset (bool,default False): return both the data and the monitors as a xr.dataset.  If false (default), just returns the data.
        Returns:
            raw (xarray): raw xarray containing your scan in PyHyper-compliant format

        '''
        md = self.loadMd(run)
        monitors = self.loadMonitors(run)
        if 'NEXAFS' in md['start']['plan_name']:
            raise NotImplementedError(f"Scan {md['start']['scan_id']} is a {md['start']['plan_name']} NEXAFS scan.  NEXAFS loading is not yet supported.")
        elif ('full' in md['start']['plan_name'] or 'short' in md['start']['plan_name'] or 'custom_rsoxs_scan' in md['start']['plan_name']) and dims is None:
            dims = ['energy']
        elif 'spiralsearch' in md['start']['plan_name'] and dims is None:
            dims = ['sam_x','sam_y']
        elif 'count' in md['start']['plan_name'] and dims is None:
            dims = ['epoch']
        elif dims is None:
            raise NotImplementedError(f"Cannot infer dimensions for a {md['start']['plan_name']} plan.  If this should be broadly supported, please raise an issue with the expected dimensions on the project GitHub.")
        #data = run['primary']['data'][md['detector']+'_image'] 
        #if self.dark_subtract:
        #    dark = run['dark']['data'][md['detector']+'_image'].mean('time') #@TODO: change to correct dark indexing
        #    image = data - dark - self.dark_pedestal
        #else:
        #    image = data - self.dark_pedestal
            
        data = run['primary']['data'][md['detector']+'_image'].astype(int) # convert from uint to handle dark subtraction
        
        if self.dark_subtract:
            dark = run['dark']['data'][md['detector']+'_image']
            darkframe = np.copy(data.time)
            for n,time in enumerate(dark.time):
                darkframe[(data.time - time)>0]=int(n)
            data = data.assign_coords(dark_id=("time", darkframe))
            def subtract_dark(img,pedestal=100,darks=None):
                return img + pedestal - darks[int(img.dark_id.values)]
            data = data.groupby('time').map(subtract_dark,darks=dark,pedestal=self.dark_pedestal)

      

        dims_to_join = []
        dim_names_to_join = []

        for dim in dims:
            try:
                test = len(md[dim])
                dims_to_join.append(md[dim])
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

        #handle the edge case of a partly-finished scan
        if len(index) != len(data['time']):
            index = index[:len(data['time'])]
        retxr = data.squeeze('dim_0').rename({'dim_1':'pix_y','dim_2':'pix_x'}).rename({'time':'system'}).assign_coords(system=index)#,md['detector']+'_image':'intensity'})
        
        #this is needed for holoviews compatibility, hopefully does not break other features.
        retxr = retxr.assign_coords({'pix_x':np.arange(0,len(retxr.pix_x)),'pix_y':np.arange(0,len(retxr.pix_y))})
        monitors = monitors.rename({'time':'system'}).reset_index('system').assign_coords(system=index).drop('system_')
        retxr.attrs.update(md)
          
        #now do corrections:
        frozen_attrs = retxr.attrs
        if self.corr_mode == 'i0':
            retxr = retxr / monitors['RSoXS Au Mesh Current']
        elif self.corr_mode != 'none':
            warnings.warn('corrections other than none are not supported at the moment',stacklevel=2)
        
        retxr.attrs.update(frozen_attrs)

        # deal with the edge case where the LAST energy of a run is repeated... this may need modification to make it correct (did the energies shift when this happened??)
        try:
            if retxr.system[-1] == retxr.system[-2]:
                retxr = retxr[:-1]
        except IndexError:
            pass
        
        if return_dataset:
            #retxr = (index,monitors,retxr)
            monitors.attrs.update(retxr.attrs)
            retxr = monitors.merge(retxr)
            
            
        return retxr


    def peekAtMd(self,run):
        return self.loadMd(run)

    def loadMonitors(self,entry,integrate_onto_images=True,n_thinning_iters=20):
        '''
        Load the monitor streams for entry.
        Args:
           entry (Bluesky document): run to extract monitors from
           integrate_onto_images (bool, default True): return integral of monitors while shutter was open for images.  if false, returns raw data.
           n_thinning_iters (int, default 20): how many iterations of binary thinning to use to exclude shutter edges.
        
        '''
        monitors = None
        for stream_name in list(entry.keys()):
            if 'monitor' in stream_name:
                if monitors is None:
                    monitors = entry[stream_name].data.read()
                else:
                    monitors = xr.merge((monitors,entry[stream_name].data.read()))
        monitors = monitors.ffill('time').bfill('time')
        if integrate_onto_images:
            monitors['RSoXS Shutter Toggle_thinned'] = monitors['RSoXS Shutter Toggle']
            monitors['RSoXS Shutter Toggle_thinned'].values = scipy.ndimage.binary_erosion(monitors['RSoXS Shutter Toggle'].values,iterations=n_thinning_iters,border_value=0)
            monitors = monitors.where(monitors['RSoXS Shutter Toggle_thinned']>0).dropna('time')
            monitors = monitors.groupby_bins('time',
np.insert(entry.primary.data['time'].values,0,0)).mean().rename_dims({'time_bins':'time'})
            monitors = monitors.assign_coords({'time':entry.primary.data['time']}).reset_coords('time_bins',drop=True)

        return monitors
    
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
            md['rsoxs_config'] = 'unknown'
            warnings.warn(f'RSoXS_Config is neither SAXS or WAXS. Looks to be {start["RSoXS_Config"]}.  Might want to check that out.',stacklevel=2)

        if md['rsoxs_config']=='saxs':
            md['detector'] = 'Small Angle CCD Detector'
        elif md['rsoxs_config']=='waxs':
            md['detector'] = 'Wide Angle CCD Detector'
        else:
            warnings.warn(f'Cannot auto-hint detector type without RSoXS config.',stacklevel=2)


        # items coming from baseline  
        baseline = run['baseline']['data']

        # items coming from primary
        try:
            primary = run['primary']['data']
        except KeyError:
            raise Exception('No primary stream --> probably you caught run before image was written.  Try again.')
        md_lookup = {
            'sam_x':'RSoXS Sample Outboard-Inboard',
            'sam_y':'RSoXS Sample Up-Down',
            'sam_z':'RSoXS Sample Downstream-Upstream',
            'sam_th':'RSoXS Sample Rotation',
            'polarization':'en_polarization_setpoint',
            'energy':'en_energy_setpoint',
            'exposure':'RSoXS Shutter Opening Time (ms)' #md['detector']+'_cam_acquire_time'
        }

        for phs,rsoxs in md_lookup.items():
            try:
                md[phs] = primary[rsoxs].values
                #print(f'Loading from primary: {phs}, value {primary[rsoxs].values}')
            except KeyError:
                try:
                    blval = baseline[rsoxs]
                    md[phs] = blval.mean().data.round(4)
                    if blval.var() > 0:
                        warnings.warn(f'While loading {rsoxs} to infill metadata entry for {phs}, found beginning and end values unequal: {baseline[rsoxs]}.  It is possible something is messed up.',stacklevel=2)
                except KeyError:
                    warnings.warn(f'Could not find {rsoxs} in either baseline or primary.  Needed to infill value {phs}.  Setting to None.',stacklevel=2)
                    md[phs] = None
        md['epoch'] = md['meas_time'].timestamp()
                    
        md['wavelength'] = 1.239842e-6 / md['energy']
        md['sampleid'] = start['scan_id']

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

    def loadSingleImage(self,filepath,coords=None, return_q=False,**kwargs):
        '''
            DO NOT USE

            This function is preserved as reference for the qx/qy loading conventions.

            NOT FOR ACTIVE USE.  DOES NOT WORK.
        '''
        if len(kwargs.keys())>0:
            warnings.warn(f'Loader does not support features for args: {kwargs.keys()}',stacklevel=2)
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
            warnings.warn(f'Correction value is negative: {corr} with headers {headerdict}.',stacklevel=2)
            corr = abs(corr)


        # # step 2: dark subtraction
        # try:
        #     darkimg = self.darks[headerdict['EXPOSURE']]
        # except KeyError:
        #     warnings.warn(f"Could not find a dark image with exposure time {headerdict['EXPOSURE']}.  Using zeros.",stacklevel=2)
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


