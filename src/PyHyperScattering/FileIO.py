import warnings
import xarray as xr
import numpy as np
import pickle
import math
import h5py
import pathlib
import datetime
import six
import PyHyperScattering
import pandas
import json

from . import _version
phs_version = _version.get_versions()['version']



@xr.register_dataset_accessor('fileio')
@xr.register_dataarray_accessor('fileio')
class FileIO:
    def __init__(self,xr_obj):
        self._obj=xr_obj
        
        self._pyhyper_type = 'reduced'
        try:
            self._chi_min = np.min(xr_obj.chi)
            self._chi_max = np.max(xr_obj.chi)
            self._chi_range = [self._chi_min,self._chi_max]
        except AttributeError:
            self._pyhyper_type = 'raw'
        
    def savePickle(self,filename):
        with open(filename, 'wb') as file:
            pickle.dump(self._obj, file)
    
    def loadPickle(filename):
        return pickle.load( open( filename, "rb" ) )
    
    def saveNexus(self,fileName):
        data = self._obj
        timestamp = datetime.datetime.now()
        # figure out if xr is a raw or integrated array
        
        axes = list(data.indexes.keys())
        array_to_save = data.variable.to_numpy()
        dims_of_array_to_save = data.variable.dims
    
        dim_to_index = {}
        index_to_dim = {}
        
        for n,dim in enumerate(dims_of_array_to_save):
            dim_to_index[dim] = n
            index_to_dim[n] = dim
        
        if 'pix_x' in axes:
            self.pyhyper_type='raw'
            nonspatial_coords = axes
            nonspatial_coords.remove('pix_x')
            nonspatial_coords.remove('pix_y')
        elif 'qx' in axes:
            self.pyhyper_type='qxy'
            nonspatial_coords = axes
            nonspatial_coords.remove('qx')
            nonspatial_coords.remove('qy')
        elif 'chi' in axes:
            self.pyhyper_type='red2d'
            nonspatial_coords = axes
            nonspatial_coords.remove('chi')
            nonspatial_coords.remove('q')
        elif 'q' in axes:
            self.pyhyper_type='red1d'
            nonspatial_coords = axes
            nonspatial_coords.remove('q')
        else:
            raise Exception(f'Invalid PyHyper_type {self.pyhyper_type}.  Cannot write Nexus.')
        raw_axes = list(data.indexes.keys())
            
            # create the HDF5 NeXus file
        with h5py.File(fileName, "w") as f:
            # point to the default data to be plotted
            f.attrs[u'default']          = u'entry'
            # give the HDF5 root some more attributes
            f.attrs[u'file_name']        = fileName
            f.attrs[u'file_time']        = str(timestamp)
            #f.attrs[u'instrument']       = u'CyRSoXS v'
            f.attrs[u'creator']          = u'PyHyperScattering NeXus writer'
            f.attrs[u'creator_version']  = phs_version
            f.attrs[u'NeXus_version']    = u'4.3.0'
            f.attrs[u'HDF5_version']     = six.u(h5py.version.hdf5_version)
            f.attrs[u'h5py_version']     = six.u(h5py.version.version)

            # create the NXentry group
            nxentry = f.create_group(u'entry')
            nxentry.attrs[u'NX_class'] = u'NXentry'
            nxentry.attrs[u'canSAS_class'] = u'SASentry'
            nxentry.attrs[u'default'] = u'data'
            #nxentry.create_dataset(u'title', data=u'SIMULATION NAME GOES HERE') # do we have a sample name field?

            #setup general file stuff
            nxdata = nxentry.create_group(u'sasdata')
            nxdata.attrs[u'NX_class'] = u'NXdata'
            nxdata.attrs[u'canSAS_class'] = u'SASdata'
            nxdata.attrs[u'canSAS_version'] = u'0.1' #required for Nika to read the file.
            nxdata.attrs[u'signal'] = u'I'      # Y axis of default plot
                

            
            '''if self.pyhyper_type == 'raw':
                nxdata.attrs[u'I_axes'] = u'pix_x,pix_y'         # X axis of default plot
                nxdata.attrs[u'Q_indices'] = f'[{dim_to_index["pix_x"]},{dim_to_index["pix_y"]}]'               
            else:
                if self.pyhyper_type == 'qxy':
                    nxdata.attrs[u'I_axes'] = u'Qx,Qy'         # X axis of default plot
                    nxdata.attrs[u'Q_indices'] = f'[{dim_to_index["Qx"]},{dim_to_index["Qy"]}]'  
                elif self.pyhyper_type == 'red2d':
                    nxdata.attrs[u'I_axes'] = u'q,chi'         # X axis of default plot
                    nxdata.attrs[u'Q_indices'] = f'[{dim_to_index["q"]},{dim_to_index["chi"]}]' 
                elif self.pyhyper_type == 'red1d':
                    nxdata.attrs[u'I_axes'] = u'q'         # X axis of default plot
                    nxdata.attrs[u'Q_indices'] = f'[{dim_to_index["q"]}]' 
                else:
                    raise Exception(f'Invalid PyHyper_type {self.pyhyper_type}.  Cannot write Nexus.')
            '''
            
            ds = nxdata.create_dataset(u'I', data=array_to_save)
            ds.attrs[u'units'] = u'arbitrary'
            ds.attrs[u'long_name'] = u'Intensity (arbitrary units)'    # suggested X axis plot label
            # the following are to enable compatibility with Nika canSAS loading
           # ds.attrs[u'signal'] = 1
            #ds.attrs[u'axes'] = u'Qx,Qy'
            I_axes = '['
            Q_indices = '['
            for axis in raw_axes:
                I_axes += f'{axis},'
                Q_indices += f'{dim_to_index[axis]},'
                if type(data.indexes[axis]) == pandas.core.indexes.multi.MultiIndex:
                    idx = data.indexes[axis]
                    I_axes = I_axes[:-1]+'('
                    for level in idx.levels:
                        ds = nxdata.create_dataset(level.name, data=level.values)
                        I_axes += f'{level.name},'
                        ds.attrs[u'PyHyper_origin'] = axis
                    I_axes = I_axes[:-1]+'),'
                else:
                    ds = nxdata.create_dataset(data.indexes[axis].name, data=data.indexes[axis].values)
                    if 'q' in axis:
                        ds.attrs[u'units'] = u'1/angstrom'
                    elif 'chi' in axis:
                        ds.attrs[u'units'] = u'degree'
                    #ds.attrs[u'long_name'] = u'Qx (A^-1)'    # suggested Y axis plot label
            I_axes = I_axes[:-1]+']'
            Q_indices = Q_indices[:-1]+']'
            nxdata.attrs[u'I_axes'] = I_axes
            nxdata.attrs[u'Q_indices'] = Q_indices
                        
            residual_attrs = nxentry.create_group(u'attrs')
            for k,v in data.attrs.items():
                #print(f'Serializing {k}...')
                if type(v)==datetime.datetime:
                    ds = residual_attrs.create_dataset(k,data=v.strftime('%Y-%m-%dT%H:%M:%SZ'))
                    ds.attrs['phs_encoding'] = 'strftime-%Y-%m-%dT%H:%M:%SZ'
                else:
                    try:
                        residual_attrs.create_dataset(k, data=v)
                    except TypeError:
                        ds = residual_attrs.create_dataset(k, data=json.dumps(v))
                        ds.attrs['phs_encoding'] = 'json'
        print("wrote file:", fileName)
 
    def loadNexus(filename):
        raise NotImplementedError


            