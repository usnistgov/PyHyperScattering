import os
import xarray as xr
import pandas as pd
import numpy as np
import warnings
import re
import os
import datetime
import time
import h5py
import pathlib
try:
    import dask.array as da
    import dask
except ImportError:
    warnings.warn('Failed to import Dask, if Dask reduction desired install pyhyperscattering[performance]',stacklevel=2)

class cyrsoxsLoader():
    '''
    Loader for cyrsoxs simulation files

    This code is modified from Dean Delongchamp.

    '''
    file_ext = 'config.txt'
    md_loading_is_quick = False
    
    
    def __init__(self,eager_load=False,profile_time=True,use_chunked_loading=False):
        '''
        Args:
            eager_load (bool, default False): block and wait for files to be created rather than erroring.  useful for live intake as simulations are being run to save time.
            profile_time (bool, default True): print time/profiling data to console
            use_chunked_loading (bool, default False): generate Dask-backed arrays
        '''
        self.eager_load = eager_load
        self.profile_time = profile_time
        self.use_chunked_loading = use_chunked_loading
    
    def read_config(self,fname):
        '''
        Reads config.txt from a CyRSoXS simulation and produces a dictionary of values.
        
        Args:
            fname (string or Path): path to file
            
        Returns: 
            config: dict representation of config file
        '''
        config = {}
        with open(fname) as f:
            for line in f:
                key,value = line.split('=')
                key = key.strip()
                value = value.split(';')[0].strip()
                if key in ['NumX','NumY','NumZ','NumThreads','EwaldsInterpolation','WindowingType']:
                    value = int(value)
                elif key in ['RotMask','WriteVTI']:
                    value = bool(value)
                elif key in ['Energies']:
                    value = value.replace("[", "")
                    value = value.replace("]", "")
                    value = np.array(value.split(","), dtype = 'float')
                else:
                    value = str(value)
                config[key] = value
        return config
    def loadDirectory(self,directory,method=None,**kwargs):
        if method == 'dask' or (method is None and self.use_chunked_loading):
            return self.loadDirectoryDask(directory,**kwargs)
        elif method == 'legacy' or (method is None and not self.use_chunked_loading):
            return self.loadDirectoryLegacy(directory,**kwargs)
        else:
            raise NotImplementedError('unsupported method {method}, expected "dask" or "legacy"')
            
    def loadDirectoryDask(self,directory,output_dir='HDF5',morphology_file=None, PhysSize=None):
        '''
        Loads a CyRSoXS simulation output directory into a Dask-backed qx/qy xarray.
        
        Args:
            directory  (string or Path): root simulation directory
            output_dir (string or Path, default /HDF5): directory relative to the base to look for hdf5 files in.
        '''
        if self.profile_time:
            start = datetime.datetime.now()
        directory = pathlib.Path(directory)
        #these waits are here intentionally so that data read-in can be started simultaneous with launching sim jobs and data will be read-in as soon as it is available.
        if self.eager_load:
            while not (directory/'config.txt').is_file():
                time.sleep(0.5)
        config = self.read_config(directory/'config.txt')
        
        if self.eager_load:
            while not (directory / output_dir).is_dir():
                time.sleep(0.5)
        #os.chdir(directory / output_dir)

        elist = config['Energies']
        num_energies = len(elist)
######### No longer contained in config.txt ###########        
#         PhysSize = float(config['PhysSize']) 
#         NumX = int(config['NumX'])
#         NumY = int(config['NumY'])

        # if we have a PhysSize, we don't need to read it in from the morphology file
        if (PhysSize is not None):
            read_morphology = False
        # if we don't have a PhysSize and no morphology file is specified, find the morphology file in the directory
        elif (morphology_file is None):
            read_morphology = True
            morphology_list = list(directory.glob('*.hdf5'))
            
            if len(morphology_list) == 1:
                morphology_file = morphology_list[0]

            # if we don't find a morphology file, warn and use default value for PhysSize
            elif len(morphology_list) == 0:
                warnings.warn('No morphology file found. Using default PhysSize of 5 nm.')
                PhysSize = 5
                read_morphology = False

            # if we find more than one morphology, warn and use first in list
            elif len(morphology_list) > 1:
                warnings.warn(f'More than one morphology.hdf5 file in directory. Choosing {morphology_list[0]}. Specify morphology_file if this is not the correct one',stacklevel=2)
                morphology_file = morphology_list[0]


        # read in PhysSize from morphology file if we need to
        if read_morphology:
            with h5py.File(morphology_file,'r') as f:
                    PhysSize = f['Morphology_Parameters/PhysSize'][()]

        #Synthesize list of filenames; note this is not using glob to see what files are there so you are at the mercy of config.txt
        hd5files = [f'Energy_{e:0.2f}.h5' for e in elist]

        outlist = []
        filehandles = []
        for i, e in enumerate(elist):
            if self.eager_load:
                while not (directory/'HDF5'/hd5files[i]).is_file():
                    time.sleep(0.5)
            
            h5 = h5py.File(directory/'HDF5'/hd5files[i],'r')
            filehandles.append(h5)

            try:
                img = da.from_array(h5['K0']['projection'])
            except KeyError:
                img = da.from_array(h5['projection'])
            if i==0:
                NumY, NumX = img.shape
                img = da.rechunk(img,chunks=(None,None))
                Qx = 2.0*np.pi*np.fft.fftshift(np.fft.fftfreq(NumX,d=PhysSize))
                Qy = 2.0*np.pi*np.fft.fftshift(np.fft.fftfreq(NumY,d=PhysSize))
                
            outlist.append(img)
        data = da.stack(outlist,axis=2)

        config['filehandles'] = filehandles
        if self.profile_time: 
             print(f'Finished reading ' + str(num_energies) + ' energies. Time required: ' + str(datetime.datetime.now()-start))
        # index = pd.MultiIndex.from_arrays([elist],names=['energy'])
        # index.name = 'system'
        return xr.DataArray(data, dims=("qx", "qy","energy"), coords={ "qx":Qx, "qy":Qy, "energy":elist},attrs=config)
        

    def loadDirectoryLegacy(self,directory,output_dir='HDF5',morphology_file=None, PhysSize=None):
        '''
        Loads a CyRSoXS simulation output directory into a qx/qy xarray.
        
        Args:
            directory  (string or Path): root simulation directory
            output_dir (string or Path, default /HDF5): directory relative to the base to look for hdf5 files in.
        '''
        if self.profile_time:
            start = datetime.datetime.now()
        directory = pathlib.Path(directory)
        #these waits are here intentionally so that data read-in can be started simultaneous with launching sim jobs and data will be read-in as soon as it is available.
        if self.eager_load:
            while not (directory/'config.txt').is_file():
                time.sleep(0.5)
        config = self.read_config(directory/'config.txt')
        
        if self.eager_load:
            while not (directory / output_dir).is_dir():
                time.sleep(0.5)
        #os.chdir(directory / output_dir)

        elist = config['Energies']
        num_energies = len(elist)
######### No longer contained in config.txt ###########        
#         PhysSize = float(config['PhysSize']) 
#         NumX = int(config['NumX'])
#         NumY = int(config['NumY'])

        # if we have a PhysSize, we don't need to read it in from the morphology file
        if (PhysSize is not None):
            read_morphology = False
        # if we don't have a PhysSize and no morphology file is specified, find the morphology file in the directory
        elif (morphology_file is None):
            read_morphology = True
            morphology_list = list(directory.glob('*.hdf5'))
            
            if len(morphology_list) == 1:
                morphology_file = morphology_list[0]

            # if we don't find a morphology file, warn and use default value for PhysSize
            elif len(morphology_list) == 0:
                warnings.warn('No morphology file found. Using default PhysSize of 5 nm.')
                PhysSize = 5
                read_morphology = False

            # if we find more than one morphology, warn and use first in list
            elif len(morphology_list) > 1:
                warnings.warn(f'More than one morphology.hdf5 file in directory. Choosing {morphology_list[0]}. Specify morphology_file if this is not the correct one',stacklevel=2)
                morphology_file = morphology_list[0]


        # read in PhysSize from morphology file if we need to
        if read_morphology:
            with h5py.File(morphology_file,'r') as f:
                    PhysSize = f['Morphology_Parameters/PhysSize'][()]

        #Synthesize list of filenames; note this is not using glob to see what files are there so you are at the mercy of config.txt
        hd5files = [f'Energy_{e:0.2f}.h5' for e in elist]

        for i, e in enumerate(elist):
            if self.eager_load:
                while not (directory/'HDF5'/hd5files[i]).is_file():
                    time.sleep(0.5)
                    
            if i==0:
                with h5py.File(directory/'HDF5'/hd5files[i],'r') as h5:
                    try:
                        img = h5['K0']['projection'][()]
                    except KeyError:
                        img = h5['projection'][()]
                    NumY, NumX = img.shape
                Qx = 2.0*np.pi*np.fft.fftshift(np.fft.fftfreq(NumX,d=PhysSize))
                Qy = 2.0*np.pi*np.fft.fftshift(np.fft.fftfreq(NumY,d=PhysSize))
                data = np.zeros([NumX*NumY*num_energies])
                
            else:
                with h5py.File(directory/'HDF5'/hd5files[i],'r') as h5:
                    try:
                        img = h5['K0']['projection'][()]
                    except KeyError:
                        img = h5['projection'][()]
                #remeshed = warp_polar_gpu(img)



            data[i*NumX*NumY:(i+1)*NumX*NumY] = img[:,:].reshape(-1, order='C')

        data = np.moveaxis(data.reshape(-1, NumY, NumX, order ='C'),0,-1)

        if self.profile_time: 
             print(f'Finished reading ' + str(num_energies) + ' energies. Time required: ' + str(datetime.datetime.now()-start))
        # index = pd.MultiIndex.from_arrays([elist],names=['energy'])
        # index.name = 'system'
        return xr.DataArray(data, dims=("qx", "qy","energy"), coords={ "qx":Qx, "qy":Qy, "energy":elist},attrs=config)
        
        #bar = xr.DataArray(data_remeshed, dims=("chi", "q", "energy"), coords={"chi":output_chi, "q":output_q, "energy":elist})        
'''
    @TODO: support larger axes based on regex of dir names - this should be implemented at the fileloader level, I think, personally.
    def datacubes_params(maindir, prefix, params):
        start = datetime.datetime.now()

        numparams = len(params)

        for j, p in enumerate(params):
        #    foo, bar = cyrsoxs_datacubes(maindir+prefix+str(p))
            mydir = maindir+prefix+str(p).zfill(4)
            if j ==0:
                #need to get all that info from config.txt including elist
                while not os.path.isfile(mydir + '/config.txt'):
                    time.sleep(0.5)
                config = read_config(mydir + '/config.txt')

                while not os.path.isdir(mydir + '/HDF5'):
                    time.sleep(0.5)
                #os.chdir(mydir +'/HDF5')

                elist = config['Energies']
                num_energies = len(elist)
                PhysSize = float(config['PhysSize'])
                NumX = int(config['NumX'])
                NumY = int(config['NumY'])

                #Synthesize list of filenames; note this is not using glob to see what files are there so you are at the mercy of config.txt
                hd5files = ["{:0.2f}".format(e) for e in elist]
                hd5files = np.core.defchararray.add("Energy_", hd5files)
                hd5files = np.core.defchararray.add(hd5files, ".h5")
            else:
                while not os.path.isdir(mydir + '/HDF5'):
                    time.sleep(0.5)
                #os.chdir(mydir +'/HDF5')

            estart = datetime.datetime.now()
            for i, e in enumerate(elist):
                while not os.path.isfile(mydir + '/HDF5/' + hd5files[i]):
                    time.sleep(0.5)
                with h5py.File(hd5files[i],'r') as h5:
                    img = h5['projection'][()]
                    remeshed = warp_polar_gpu(img)
                if (j==0) and (i==0):
                    #create all this only once
                    Qx = 2.0*np.pi*np.fft.fftshift(np.fft.fftfreq(img.shape[1],d=PhysSize))
                    Qy = 2.0*np.pi*np.fft.fftshift(np.fft.fftfreq(img.shape[0],d=PhysSize))
                    q = np.sqrt(Qy**2+Qx**2)
                    output_chi = np.linspace(0,360,360)
                    lenchi = len(output_chi)
                    output_q = np.linspace(0,np.amax(q), remeshed.shape[1])
                    lenq = len(output_q)
                    data = np.zeros([NumX*NumY*num_energies*numparams])
                    data_remeshed = np.zeros([len(output_chi)*len(output_q)*num_energies*numparams])

                data[j*num_energies*NumX*NumY + i*NumX*NumY:j*num_energies*NumX*NumY +(i+1)*NumX*NumY] = img[:,:].reshape(-1, order='C')
                data_remeshed[j*num_energies*lenchi*lenq + i*lenchi*lenq:j*num_energies*lenchi*lenq +(i+1)*lenchi*lenq] = remeshed[:,:].reshape(-1, order='C')
            print(f'Finished reading ' + str(num_energies) + ' energies. Time required: ' + str(datetime.datetime.now()-estart))
        data = data.reshape(numparams*num_energies, NumY, NumX, order ='C')
        data = data.reshape(numparams, num_energies, NumY, NumX, order ='C')
        data_remeshed = data_remeshed.reshape(numparams*num_energies,lenchi, lenq, order ='C')
        data_remeshed = data_remeshed.reshape(numparams, num_energies,lenchi, lenq, order ='C')

        lfoo = xr.DataArray(data, dims=("param","energy", "Qy", "Qx"), coords={"energy":elist, "param":params, "Qy":Qy, "Qx":Qx})
        lbar = xr.DataArray(data_remeshed, dims=("param", "energy", "chi", "q"), coords={"chi":output_chi, "q":output_q, "energy":elist, "param":params})

        print(f'Finished reading ' + str(numparams) + ' parameters. Time required: ' + str(datetime.datetime.now()-start))

        return lfoo, lbar
'''
