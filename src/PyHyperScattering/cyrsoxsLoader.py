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

class cyrsoxsLoader():
    '''
    Loader for cyrsoxs simulation files

    This code is modified from Dean Delongchamp.

    '''
    file_ext = 'config.txt'
    md_loading_is_quick = False
    
    
    def __init__(self,eager_load=False,profile_time=True):
        '''
        Args:
            eager_load (bool, default False): block and wait for files to be created rather than erroring.  useful for live intake as simulations are being run to save time.
            profile_time (bool, default True): print time/profiling data to console
        '''
        self.eager_load = eager_load
        self.profile_time = profile_time
    
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

   

    def loadDirectory(self,directory,output_dir='HDF5'):
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
            while not (directory/'config.txt').is_dir():
                time.sleep(0.5)
        config = self.read_config(directory/'config.txt')
        
        if self.eager_load:
            while not (directory / output_dir).is_dir():
                time.sleep(0.5)
        #os.chdir(directory / output_dir)

        elist = config['Energies']
        num_energies = len(elist)
        PhysSize = float(config['PhysSize'])
        NumX = int(config['NumX'])
        NumY = int(config['NumY'])

        #Synthesize list of filenames; note this is not using glob to see what files are there so you are at the mercy of config.txt
        hd5files = [f'Energy_{e:0.2f}.h5' for e in elist]

        for i, e in enumerate(elist):
            if self.eager_load:
                while not (directory/'HDF5'/hd5files[i]).is_dir():
                    time.sleep(0.5)
            with h5py.File(directory/'HDF5'/hd5files[i],'r') as h5:
                img = h5['projection'][()]
                #remeshed = warp_polar_gpu(img)
            if i==0:
                Qx = 2.0*np.pi*np.fft.fftshift(np.fft.fftfreq(img.shape[1],d=PhysSize))
                Qy = 2.0*np.pi*np.fft.fftshift(np.fft.fftfreq(img.shape[0],d=PhysSize))
                #q = np.sqrt(Qy**2+Qx**2)
                #output_chi = np.linspace(0,360,360)
                #output_q = np.linspace(0,np.amax(q), remeshed.shape[1])
                data = np.zeros([NumX*NumY*num_energies])
                #data_remeshed = np.zeros([len(output_chi)*len(output_q)*num_energies])

            data[i*NumX*NumY:(i+1)*NumX*NumY] = img[:,:].reshape(-1, order='C')
            #data_remeshed[i*len(output_chi)*len(output_q):(i+1)*len(output_chi)*len(output_q)] = remeshed[:,:].reshape(-1, order='C')

        data = np.moveaxis(data.reshape(-1, NumY, NumX, order ='C'),0,-1)
        #data_remeshed = np.moveaxis(data_remeshed.reshape(-1, len(output_chi), len(output_q), order ='C'),0,-1)
        if self.profile_time: 
             print(f'Finished reading ' + str(num_energies) + ' energies. Time required: ' + str(datetime.datetime.now()-start))
        index = pd.MultiIndex.from_arrays([elist],names=['energy'])
        index.name = 'system'
        return xr.DataArray(data, dims=("qx", "qy", "system"), coords={"qx":Qx, "qy":Qy, "system":index},attrs=config)
        
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