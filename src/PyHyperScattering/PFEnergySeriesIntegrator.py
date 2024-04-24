from pyFAI import azimuthalIntegrator
from PyHyperScattering.PFGeneralIntegrator import PFGeneralIntegrator
import h5py
import warnings
import xarray as xr
import numpy as np
import math
import pandas as pd
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

DataArrayGroupBy.progress_apply_ufunc = inner_generator(df_function='apply_ufunc')
DatasetGroupBy.progress_apply_ufunc = inner_generator(df_function='apply_ufunc')

#end monkey patch

class PFEnergySeriesIntegrator(PFGeneralIntegrator):

    def integrateSingleImage(self,img):
        # for each image: 
        #    get the energy and locate the matching integrator
        #    use that integrator to reduce
        #    return single reduced frame

        #print(f'    img.energy is a {type(img.energy)}, len = {len(img.energy)} and is {img.energy}') 
        #print(f'    img.system is a {type(img.system)}, len = {len(img.energy)} and is {img.system}')
        #print(f'    img.system.levels: {img.indexes["system"].names}')
        #print(f'    img.indexes: {img.indexes}')
        #print(f'    type of system value: {type(getattr(img,"system").values)}')
        #print(f'    shape of system value: {getattr(img,"system").values.shape}')
        en = None
        if 'energy' not in img.indexes and type(img.energy) != float:
            try: 
                multiindex_name = None
                for idx in img.indexes:
                    if type(img.indexes[idx]) == pd.core.indexes.multi.MultiIndex:
                        multiindex_name = idx
                        break
                if 'energy' in img.indexes[multiindex_name].names:
                    for i,n in enumerate(img.indexes[multiindex_name].names):
                        if n == 'energy':
                            idx_of_energy = i
                    try:
                        en = float(getattr(img,multiindex_name).values[idx_of_energy][0]) # this does not work for 2022-2 data; does it work for other cycles?
                    except IndexError:
                        en = float(getattr(img,multiindex_name).values[0][idx_of_energy])
            except KeyError:
                pass
        if en is not None:
            pass
        elif type(img.energy) == float:
            en = img.energy
        else:
            try:
                en = img.energy.values[0]
                if len(img.energy.values)>1:
                    warnings.warn(f'Using the first energy value of {img.energy.values}, check that this is correct.',stacklevel=2)
            except IndexError:
                en = float(img.energy)
            except AttributeError:
                try:
                    en = img.energy[0]
                    warnings.warn(f'Using the first energy value of {img.energy}, if this contains more than one energy, your data are likely wrong.',stacklevel=2)
                except IndexError:
                    en = float(img.energy)
            
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
        except (TypeError,AttributeError):
            return res
    def setupIntegrators(self,energies):
        '''
        Sets up the integrator stack as a function of energy.

        The final statement ensures that the integrator for the median of the set is created.  This integrator is used to set
        the output q-binning.

        Details: (copied from a message)

        The fact that energy is changing during reduction means that if not forced to something, the output q bins of the integrator will move as well (since the pixel to q mappings are moving with energy). Because sparse data in q is a nightmare, we pick a given set of q bins corresponding to the median of the energies in the scan. That is a compromise between a few approaches. This line manually creates that integrator with default q binning settings so we can take those bins and tell all the other integrators to use that output q grid.

It would cosmically be better (for things like resolution calculation) to have the q bins actually move, but sparse arrays are computationally hard. Eventually (2-3 years of high performance Python evolution) I think that will be the right way to do it, this is an intermediate.
        '''
        for en in energies:
            self.createIntegrator(en)
        self.createIntegrator(np.median(energies))
    def setupDestQ(self,energies):
        self.dest_q = self.integrator_stack[np.median(energies)].integrate2d(np.zeros_like(self.mask).astype(int), self.npts, 
                                                   unit='arcsinh(q.µm)' if self.use_log_ish_binning else 'q_A^-1',
                                                   method=self.integration_method).radial

    def integrateImageStack_dask(self,img_stack,chunksize=5):
        self.setupIntegrators(img_stack.energy.data)
        self.setupDestQ(img_stack.energy.data)
        indexes = list(img_stack.dims)
        indexes.remove('pix_x')
        indexes.remove('pix_y')

        # idx_name_to_use = 'energy'#indexes[0]
        # idx_val_to_use = img_stack.indexes[idx_name_to_use]
        

        if 'energy' in indexes:
            dim_to_chunk = 'energy'
        else:
            dim_to_chunk = indexes[0]
            #this probably should check which is the longest?  Shortest?  and chunk that.
        print(f'chunking on {dim_to_chunk}')
        
        coord_dict = {}
        shape = tuple([])
        order_list = []
        for idx in indexes:
            order_list.append(idx)
            coord_dict[idx] = img_stack.indexes[idx]
            shape = shape + tuple([len(img_stack.indexes[idx])])
        shape = (360,self.npts) + shape 
        
        
        fake_image_to_process = img_stack.isel(**{dim_to_chunk:0},drop=False)
        #fake_image_to_process.attrs['energy'] = img_stack.energy.isel(**{idx_name_to_use:0})
        demo_integration = self.integrateSingleImage(fake_image_to_process)
        coord_dict.update({'chi':demo_integration.chi,'q':self.dest_q})
        
        desired_order_list = ['chi','q']+order_list
        coord_dict_sorted = {k: coord_dict[k] for k in desired_order_list}
        
        template = xr.DataArray(np.empty(shape),coords=coord_dict_sorted)
   
        
        template = template.chunk({dim_to_chunk:chunksize})
        if 'image_num' in demo_integration.dims:
            template = template.transpose(*[item if item != 'image_num' else dim_to_chunk for item in demo_integration.dims])
         
        integ_fly = img_stack.chunk({dim_to_chunk:chunksize}).map_blocks(self.integrateImageStack_legacy,template=template)#integ_traditional.chunk({'energy':5}))
        return integ_fly 

    def integrateImageStack_legacy(self,img_stack):
        # get just the energies of the image stack
       # if type(img_stack.energy)== np.ndarray:
       
        # get just the energies of the image stack
        #energies = img_stack.energy.to_dataframe()
        
        #energies = energies['energy'].drop_duplicates()
        energies = np.unique(img_stack.energy.data)
        #create an integrator for each energy
        self.setupIntegrators(energies)
        # find the output q for the midpoint and set the final q binning
        if not hasattr(self,'dest_q'):
            try:
                self.setupDestQ(energies)
            except TypeError as e:
                if 'diffSolidAngle() missing 2 required positional arguments: ' in str(e):
                    raise TypeError('Geometry is incorrect, cannot integrate.\n \n - Do your mask dimensions match your image dimensions? \n - Do you have pixel sizes set that are not zero?\n - Is SDD, beamcenter/poni, and tilt set correctly?') from e
                else:
                    raise e
        if self.use_log_ish_binning:
            self.dest_q = np.sinh(self.dest_q)/10000
        # single image reduce each entry in the stack
        # + 
        # restack the reduced data
        data = img_stack
        indexes = list(data.dims)
        indexes.remove('pix_x')
        indexes.remove('pix_y')

        # the following section attempts to shape the array that we have so that it meets the requirements for the xarray GroupBy/map
        # paradigm; specifically, it needs one index that's not pixel x/y.  We either identify if that is the case, or we make it the
        # case.  however, there are probably edge cases not handled here
        
        if len(indexes) == 1:
            if img_stack.__getattr__(indexes[0]).to_pandas().drop_duplicates().shape[0] != img_stack.__getattr__(indexes[0]).shape[0]:
                warnings.warn(f'Axis {indexes[0]} contains duplicate conditions.  This is not supported and may not work.  Try adding additional coords to separate image conditions',stacklevel=2)
            data_int = data.groupby(indexes[0],squeeze=False).progress_apply(self.integrateSingleImage)
        elif len(indexes) == 0:
            data_int = self.integrateSingleImage(data).isel(image_num=0)
        else:
            #some kinda logic to check for existing multiindexes and stack into them appropriately maybe
            data = data.stack({'pyhyper_internal_multiindex':indexes})
            if data.pyhyper_internal_multiindex.to_pandas().drop_duplicates().shape[0] != data.pyhyper_internal_multiindex.shape[0]:
                warnings.warn('Your index set contains duplicate conditions.  This is not supported and may not work.  Try adding additional coords to separate image conditions',stacklevel=2)
        
            data_int = data.groupby('pyhyper_internal_multiindex',squeeze=False).progress_apply(self.integrateSingleImage).unstack('pyhyper_internal_multiindex')
        return data_int
        #return img_stack.groupby('system',squeeze=False).progress_apply(self.integrateSingleImage)
    
    def integrateImageStack(self,img_stack,method=None,chunksize=None):
        '''
        
        '''

        if (self.use_chunked_processing and method is None) or method=='dask':
            func_args = {}
            if chunksize is not None:
                func_args['chunksize'] = chunksize
            return self.integrateImageStack_dask(img_stack,**func_args)
        elif (method is None) or method == 'legacy':
            return self.integrateImageStack_legacy(img_stack)
        else:
            raise NotImplementedError(f'unsupported integration method {method}')



    def createIntegrator(self,en,recreate=False):
        if en not in self.integrator_stack.keys() or recreate:
            self.integrator_stack[en] = azimuthalIntegrator.AzimuthalIntegrator(
            self.dist, self.poni1, self.poni2, self.rot1, self.rot2, self.rot3 ,pixel1=self.pixel1,pixel2=self.pixel2, wavelength = 1.239842e-6/en)
        return self.integrator_stack[en]
    def __init__(self,**kwargs):
        self.integrator_stack = {}
        
        super().__init__(**kwargs)
    def recreateIntegrator(self):
        pass
    
    def __str__(self):
        return f"PyFAI energy-series integrator  SDD = {self.dist} m, poni1 = {self.poni1} m, poni2 = {self.poni2} m, rot1 = {self.rot1} rad, rot2 = {self.rot2} rad"
