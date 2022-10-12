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
    from httpx import HTTPStatusError
    import tiled
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
    
    
    def summarize_run(self, outputType:str = 'default', cycle:str = None, proposal:str =None, saf:str = None, user:str = None, institution:str = None, project:str = None, sample:str = None, sampleID:str = None,  plan:str = None, userOutputs: list = [], **kwargs) -> pd.DataFrame:
        ''' Search the databroker.client.CatalogOfBlueskyRuns for scans matching all provided keywords and return metadata as a dataframe. 
        
        Matches are made based on the values in the top level of the 'start' dict within the metadata of each 
        entry in the Bluesky Catalog (databroker.client.CatalogOfBlueskyRuns). Based on the search arguments provided, 
        a pandas dataframe is returned where rows correspond to catalog entries (scans) and columns contain  metadata.
        Several presets are provided for choosing which columns are generated, along with an interface for 
        user-provided search arguments and additional metadata. Fails gracefully on bad user input/ changes to 
        underlying metadata scheme. 
        
        Ex1: All of the carbon,fluorine,or oxygen scans for a single sample series in the most recent cycle:
            bsCatalogReduced4 = db_loader.summarize_run(sample="bBP_", institution="NIST", cycle = "2022-2", plan="carbon|fluorine|oxygen")
        
        Ex2: Just all of the scan Ids for a particular sample:
            bsCatalogReduced4 = db_loader.summarize_run(sample="BBP_PFP09A", outputType='scans')
        
        Ex3: Complex Search with custom parameters
            bsCatalogReduced3 = db_loader.summarize_run(['angle', '-1.6', 'numeric'], outputType='all',sample="BBP_", cycle = "2022-2", 
            institution="NIST",plan="carbon", userOutputs = [["Exposure Multiplier", "exptime", r'catalog.start'], ["Stop 
            Time","time",r'catalog.stop']])
        
        Args:
            outputType (str, optional): modulates the content of output columns in the returned dataframe
                'default' returns scan_id, start time, cycle, institution, project, sample_name, sample_id, plan name, detector, 
                polarization, exit_status, and num_images
                'scans' returns only the scan_ids (1-column dataframe)
                'ext_msmt' returns default columns AND bar_spot, sample_rotation
                'ext_bio' returns default columns AND uid, saf, user_name
                'all' is equivalent to 'default' and all other additive choices
            cycle (str, optional): NSLS2 beamtime cycle, regex search e.g., "2022" matches "2022-2", "2022-1"
            proposal (str, optional): NSLS2 PASS proposal ID, case-insensitive, exact match, e.g., "GU-310176"
            saf (str, optional): Safety Approval Form (SAF) number, exact match, e.g., "309441" 
            user (str, optional): User name, case-insensitive, regex search e.g., "eliot" matches "Eliot", "Eliot Gann"
            institution (str, optional): Research Institution, case-insensitive, exact match, e.g., "NIST"
            project (str, optional): Project code, case-insensitive, regex search, 
                e.g., "liquid" matches "Liquids", "Liquid-RSoXS"
            sample (str, optional): Sample name, case-insensitive, regex search, e.g., "BBP_" matches "BBP_PF902A"
            sampleID (str, optional): Sample ID, case-insensitive, regex search, e.g., "BBP_" matches "BBP_PF902A"
            plan (str, optional): Measurement Plan, case-insensitive, regex search,  
                e.g., "Full" matches "full_carbon_scan_nd", "full_fluorine_scan_nd"
                e.g., "carbon|oxygen|fluorine" matches carbon OR oxygen OR fluorine scans
            **kwargs: Additional search terms can be provided as keyword args and will further filter 
                the catalog Valid input follows metadataLabel='searchTerm' or metadataLavel = ['searchTerm','matchType'].
                Metadata labels must match an entry in the 'start' dictionary of the catalog. Supported match types are
                combinations of 'case-insensitive', 'case-sensitive', and 'exact' OR 'numeric'. Default behavior is to 
                do a case-sensitive regex match. For metadata labels that are not valid python names, create the kwarg 
                dict before passing into the function (see example 3). Additional search terms will appear in the 
                output data columns.
                Ex1: passing in cycle='2022' would match 'cycle'='2022-2' AND 'cycle='2022-1'
                Ex2: passing in grazing=[0,'numeric'] would match grazing==0
                Ex3: create kwargs first, then pass it into the function. 
                    kwargs = {'2weird metadata label': "Bob", 'grazing': 0, 'angle':-1.6}
                    db_loader.summarize_run(sample="BBP_PFP09A", outputType='scans', **kwargs)
            userOutputs (list of lists, optional): Additional metadata to be added to output can be specified as a list of lists. Each 
                sub-list specifies a metadata field as a 3 element list of format:
                [Output column title (str), Metadata label (str), Metadata Source (raw str)],
                Valid options for the Metadata Source are any of [r'catalog.start', r'catalog.start["plan_args"], r'catalog.stop', 
                r'catalog.stop["num_events"]']
                e.g., userOutputs = [["Exposure Multiplier","exptime", r'catalog.start'], ["Stop Time","time",r'catalog.stop']]

        Returns:
            pd.Dataframe containing the results of the search, or an empty dataframe if the search fails
        '''
        
        # Pull in the reference to the databroker.client.CatalogOfBlueskyRuns attribute
        bsCatalog = self.c

        ### Part 1: Search the database sequentially, reducing based on matches to search terms
        # Plan the 'default' search through the keyword parameters, build list of [metadata ID, user input value, match type]
        defaultSearchDetails = [['cycle', cycle, 'case-insensitive'],
                                ['proposal_id',proposal,'case-insensitive exact'],
                                ['saf_id',saf,'case-insensitive exact'], 
                               ['user_name',user,'case-insensitive'],
                               ['institution',institution,'case-insensitive exact'],
                               ['project_name',project,'case-insensitive'],
                               ['sample_name',sample,'case-insensitive'],
                               ['sample_id',sampleID,'case-insensitive'],
                               ['plan_name',plan,'case-insensitive']]
        
        # Pull any user-provided search terms
        userSearchList = []
        for userLabel, value in kwargs.items():
            #Minimial check for bad user input
            if isinstance(value, str):
                userSearchList.append([userLabel,value,''])
            elif isinstance(value, int) or isinstance(value, float):
                userSearchList.append([userLabel,value,'numeric'])
            elif isinstance(value, list) and len(value)==2:
                userSearchList.append([userLabel,value[0],value[1]])
            else: #bad user input
                warnString = ("Error parsing a keyword search term, check the format.\nSkipped argument: " 
                              + str(value))
                warnings.warn(warnString,stacklevel=2)

        
        #combine the lists of lists
        fullSearchList = defaultSearchDetails + userSearchList
        
        df_SearchDet = pd.DataFrame(fullSearchList, columns=['Metadata field:', 'User input:', 'Search scheme:'])
    
        # Iterate through search terms sequentially, reducing the size of the catalog based on successful matches
        
        reducedCatalog = bsCatalog
        loopDesc = "Searching by keyword arguments"
        for index, searchSeries in tqdm(df_SearchDet.iterrows(), total=df_SearchDet.shape[0], desc=loopDesc):
            
            # Skip arguments with value None, and quits if the catalog was reduced to 0 elements
            if (searchSeries[1] is not None) and (len(reducedCatalog)> 0):
                
                # For numeric entries, do Key equality
                if 'numeric' in str(searchSeries[2]):
                    reducedCatalog = reducedCatalog.search(Key(searchSeries[0])==float(searchSeries[1]))
                
                else: #Build regex search string
                    reg_prefix = ''
                    reg_postfix = ''

                    # Regex cheatsheet: 
                        #(?i) is case insensitive
                        #^_$ forces exact match to _, ^ anchors the start, $ anchors the end  
                    if 'case-insensitive' in str(searchSeries[2]):
                        reg_prefix += "(?i)"
                    if 'exact' in searchSeries[2]:
                        reg_prefix += "^"
                        reg_postfix += "$"


                    regexString = reg_prefix + str(searchSeries[1]) + reg_postfix

                    # Search/reduce the catalog
                    reducedCatalog = reducedCatalog.search(Regex(searchSeries[0], regexString))
                
                # If a match fails, notify the user which search parameter yielded 0 results
                if len(reducedCatalog) == 0:
                    warnString = ("Catalog reduced to zero when attempting to match the following condition:\n" 
                                  + searchSeries.to_string() 
                                  + "\n If this is a user-provided search parameter, check spelling/syntax.\n")
                    warnings.warn(warnString,stacklevel=2)
                    return pd.DataFrame()
        
        ### Part 2: Build and return output dataframe
            
        if outputType=='scans': # Branch 2.1, if only scan IDs needed, build and return a 1-column dataframe
            scan_ids = []
            loopDesc = "Building scan list"
            for index,scanEntry in tqdm((enumerate(reducedCatalog)),total=len(reducedCatalog), desc = loopDesc):
                scan_ids.append(reducedCatalog[scanEntry].start["scan_id"])
            return pd.DataFrame(scan_ids, columns=["Scan ID"])
        
        else: # Branch 2.2, Output metadata from a variety of sources within each the catalog entry 
            
            # Store details of output values as a list of lists
            # List elements are [Output Column Title, Bluesky Metadata Code, Metadata Source location, Applicable Output flag]
            outputValueLibrary = [["scan_id","scan_id",r'catalog.start','default'],
                                   ["uid","uid",r'catalog.start','ext_bio'],
                                   ["start time","time",r'catalog.start','default'],
                                   ["cycle","cycle",r'catalog.start','default'],
                                   ["saf","SAF",r'catalog.start','ext_bio'],
                                   ["user_name","user_name",r'catalog.start','ext_bio'],
                                   ["institution","institution",r'catalog.start','default'],
                                   ["project","project_name",r'catalog.start','default'],
                                   ["sample_name","sample_name",r'catalog.start','default'],
                                   ["sample_id","sample_id",r'catalog.start','default'],
                                   ["bar_spot","bar_spot",r'catalog.start','ext_msmt'],
                                   ["plan","plan_name",r'catalog.start','default'],
                                   ["detector","RSoXS_Main_DET",r'catalog.start','default'],
                                   ["polarization","pol",r'catalog.start["plan_args"]','default'],
                                   ["sample_rotation","angle",r'catalog.start','ext_msmt'],
                                   ["exit_status","exit_status",r'catalog.stop','default'],
                                   ["num_Images","primary",r'catalog.stop["num_events"]','default'],
                                  ]
            
            # Subset the library based on the output flag selected
            activeOutputValues = []
            activeOutputLabels = []
            for outputEntry in outputValueLibrary:
                if (outputType == 'all') or (outputEntry[3] == outputType) or (outputEntry[3] == 'default'):
                    activeOutputValues.append(outputEntry)
                    activeOutputLabels.append(outputEntry[0])
            
            # Add any user-provided Output labels
            userOutputList = []
            for userOutEntry in userOutputs:
                #Minimial check for bad user input
                if isinstance(userOutEntry, list) and len(userOutEntry)==3:
                    activeOutputValues.append(userOutEntry)
                    activeOutputLabels.append(userOutEntry[0])
                else: #bad user input
                    warnString = ("Error parsing user-provided output request, check the format.\nSkipped: " 
                                  + str(userOutEntry))
                    warnings.warn(warnString,stacklevel=2)
            
            # Add any user-provided search terms
            for userSearchEntry in userSearchList:
                activeOutputValues.append([userSearchEntry[0],userSearchEntry[0],r'catalog.start','default'])
                activeOutputLabels.append(userSearchEntry[0])

            
            # Build output dataframe as a list of lists
            outputList = []
           
            # Outer loop: Catalog entries
            loopDesc =  "Building output dataframe"
            for index,scanEntry in tqdm((enumerate(reducedCatalog)),total=len(reducedCatalog), desc = loopDesc):
                
                singleScanOutput = []
                
                # Pull the start and stop docs once
                currentCatalogStart =  reducedCatalog[scanEntry].start
                currentCatalogStop =  reducedCatalog[scanEntry].stop
                
                currentScanID = currentCatalogStart["scan_id"]
                
                # Inner loop: append output values
                for outputEntry in activeOutputValues:
                    outputVariableName = outputEntry[0]
                    metaDataLabel = outputEntry[1]
                    metaDataSource = outputEntry[2]
                    
                    try: # Add the metadata value depending on where it is located                    
                        if metaDataSource == r'catalog.start':
                            singleScanOutput.append(currentCatalogStart[metaDataLabel])
                        elif metaDataSource == r'catalog.start["plan_args"]':
                            singleScanOutput.append(currentCatalogStart["plan_args"][metaDataLabel])
                        elif metaDataSource == r'catalog.stop':
                            singleScanOutput.append(currentCatalogStop[metaDataLabel])
                        elif metaDataSource == r'catalog.stop["num_events"]':
                            singleScanOutput.append(currentCatalogStop["num_events"][metaDataLabel])
                        else:
                            warnString =("Scan: > " + str(currentScanID) + " < Failed to locate metaData entry for > " 
                                         + str(outputVariableName) + " <\n Tried looking for label: > " 
                                         + str(metaDataLabel) + " < in: " + str(metaDataSource))
                            warnings.warn(warnString,stacklevel=2)
                            
                    except (KeyError,TypeError):
                        warnString =("Scan: > " + str(currentScanID) + " < Failed to locate metaData entry for > " 
                                     + str(outputVariableName) + " <\n Tried looking for label: > " 
                                     + str(metaDataLabel) + " < in: " + str(metaDataSource))
                        warnings.warn(warnString,stacklevel=2)
                        singleScanOutput.append("N/A")
                    
                #Append to the filled output list for this entry to the list of lists
                outputList.append(singleScanOutput)
            
            # Convert to dataframe for export
            return pd.DataFrame(outputList, columns = activeOutputLabels)          
            
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
        except (KeyError,TypeError):
            npts.append(0)
        start_times.append(doc["time"])
        
    def loadSeries(self,run_list,meta_dim,loadrun_kwargs={},):
        '''
        Loads a series of runs into a single xarray object, stacking along meta_dim.
        
        Useful for a set of samples, or a set of polarizations, etc., taken in different scans.
        
        Args:
        
            run_list (list): list of scan ids to load
            
            meta_dim (str): dimension to stack along.  must be a valid attribute/metadata value, such as polarization or sample_name
            
        Returns:
            raw: xarray.Dataset with all scans stacked
        
        '''
        
        scans = []
        axes = []
        label_vals = []
        for run in run_list:
            loaded = self.loadRun(self.c[run],**loadrun_kwargs).unstack('system')
            axis = list(loaded.indexes.keys())
            try:
                axis.remove('pix_x')
                axis.remove('pix_y')
            except ValueError:
                pass
            try:
                axis.remove('qx')
                axis.remove('qy')
            except ValueError:
                pass
            axes.append(axis)
            scans.append(loaded)
            label_vals.append(loaded.__getattr__(meta_dim))
        assert len(axes) == axes.count(axes[0]), f'Error: not all loaded data have the same axes.  This is not supported yet.\n {axes}'
        axes[0].insert(0,meta_dim)
        new_system = axes[0]
        #print(f'New system to be stacked as: {new_system}')
        #print(f'meta_dimension = {meta_dim}')
        #print(f'labels in this dim are {label_vals}')
        return xr.concat(scans,dim=meta_dim).assign_coords({meta_dim:label_vals}).stack(system=new_system)
        
        
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
        

        data = run['primary']['data'][md['detector']+'_image']
        if type(data) == tiled.client.array.ArrayClient:
            data = xr.DataArray(data)
        data = data.astype(int)   # convert from uint to handle dark subtraction

        if self.dark_subtract:
            dark = run['dark']['data'][md['detector']+'_image']
            if type(dark) == tiled.client.array.ArrayClient:
                dark = xr.DataArray(dark)
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
        try:
            monitors = monitors.rename({'time':'system'}).reset_index('system').assign_coords(system=index).drop('system_')
        except:
            warnings.warn('Error assigning monitor readings to system.  Problem with monitors.  Please check.',stacklevel=2)
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
            try:
                monitors['RSoXS Shutter Toggle_thinned'] = monitors['RSoXS Shutter Toggle']
                monitors['RSoXS Shutter Toggle_thinned'].values = scipy.ndimage.binary_erosion(monitors['RSoXS Shutter Toggle'].values,iterations=n_thinning_iters,border_value=0)
                monitors = monitors.where(monitors['RSoXS Shutter Toggle_thinned']>0).dropna('time')
                monitors = monitors.groupby_bins('time',
    np.insert(entry.primary.data['time'].values,0,0)).mean().rename_dims({'time_bins':'time'})
                monitors = monitors.assign_coords({'time':entry.primary.data['time']}).reset_coords('time_bins',drop=True)
            except:
                warnings.warn('Error while time-integrating onto images.  Check data.',stacklevel=2)
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
            elif (meas_time > datetime.datetime(2022,5,1)) and (meas_time < datetime.datetime(2022,7,7)):
                # these params determined by Camille from Igor
                md['beamcenter_x'] = 498 # not the best estimate; I didn't have great data
                md['beamcenter_y'] =  498
                md['sdd'] = 512.12 # GUESS; SOMEONE SHOULD CONFIRM WITH A BCP MAYBE??
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
            elif (meas_time > datetime.datetime(2022,5,1)) and (meas_time < datetime.datetime(2022,7,7)):
                # these params determined by Camille from Igor
                md['beamcenter_x'] = 397.91
                md['beamcenter_y'] = 549.76
                md['sdd'] = 34.5 # GUESS; SOMEONE SHOULD CONFIRM WITH A BCP MAYBE??
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
        except (KeyError,HTTPStatusError):
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
        md_secondary_lookup = {
            'energy':'en_monoen_setpoint',
            }
        for phs,rsoxs in md_lookup.items():
            try:
                md[phs] = primary[rsoxs].read()
                #print(f'Loading from primary: {phs}, value {primary[rsoxs].values}')
            except (KeyError,HTTPStatusError):
                try:
                    blval = baseline[rsoxs]
                    if type(blval) == tiled.client.array.ArrayClient:
                        blval = blval.read()
                    md[phs] = blval.mean().round(4)
                    if blval.var() > 0:
                        warnings.warn(f'While loading {rsoxs} to infill metadata entry for {phs}, found beginning and end values unequal: {baseline[rsoxs]}.  It is possible something is messed up.',stacklevel=2)
                except (KeyError,HTTPStatusError):
                    try:
                        md[phs] = primary[md_secondary_lookup[phs]].read()
                    except (KeyError,HTTPStatusError):
                        try:
                            blval = baseline[md_secondary_lookup[phs]]
                            if type(blval) == tiled.client.array.ArrayClient:
                                blval = blval.read()
                            md[phs] = blval.mean().round(4)
                            if blval.var() > 0:
                                warnings.warn(f'While loading {md_secondary_lookup[phs]} to infill metadata entry for {phs}, found beginning and end values unequal: {baseline[rsoxs]}.  It is possible something is messed up.',stacklevel=2)  
                        except (KeyError,HTTPStatusError):
                            warnings.warn(f'Could not find {rsoxs} in either baseline or primary.  Needed to infill value {phs}.  Setting to None.',stacklevel=2)
                            md[phs] = None
        md['epoch'] = md['meas_time'].timestamp()
        
        try:
            md['wavelength'] = 1.239842e-6 / md['energy']
        except TypeError:
            md['wavelength'] = None
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


