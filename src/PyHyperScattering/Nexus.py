import warnings
import xarray as xr
import numpy as np
import math
import pathlib
import h5py
import pathlib
'''
def save(xr,fileName):
    
    # figure out if xr is a raw or integrated array
        # create the HDF5 NeXus file
    with h5py.File(fileName, "w") as f:
        # point to the default data to be plotted
        f.attrs[u'default']          = u'entry'
        # give the HDF5 root some more attributes
        f.attrs[u'file_name']        = fileName
        f.attrs[u'file_time']        = timestamp
        f.attrs[u'instrument']       = u'CyRSoXS v'
        f.attrs[u'creator']          = u'PyHyperScattering NeXus writer'
        f.attrs[u'NeXus_version']    = u'4.3.0'
        f.attrs[u'HDF5_version']     = six.u(h5py.version.hdf5_version)
        f.attrs[u'h5py_version']     = six.u(h5py.version.version)

        # create the NXentry group
        nxentry = f.create_group(u'entry')
        nxentry.attrs[u'NX_class'] = u'NXentry'
        nxentry.attrs[u'canSAS_class'] = u'SASentry'
        nxentry.attrs[u'default'] = u'data'
        nxentry.create_dataset(u'title', data=u'SIMULATION NAME GOES HERE') #@TODO

        # figure out if one image or more
        if 'system' in xr.dimensions:
            # writing a stack of images
            pass
        else: 
            # writing a single image
            
            if single_image_energy is not None:
                imgpos = coords['energy'][single_image_energy].index()
            else:
                imgpos = 0
            nxdata = nxentry.create_group(u'sasdata_singleimg')
            nxdata.attrs[u'NX_class'] = u'NXdata'
            nxdata.attrs[u'canSAS_class'] = u'SASdata'
            #nxdata.attrs[u'canSAS_version'] = u'0.1' #required for Nika to read the file.
            nxdata.attrs[u'signal'] = u'I'      # Y axis of default plot
            nxdata.attrs[u'I_axes'] = u'Qx,Qy'         # X axis of default plot
            nxdata.attrs[u'Q_indices'] = '[0,1]'   # use "mr" as the first dimension of I00

            # X axis data
            ds = nxdata.create_dataset(u'I', data=data['img'][imgpos])
            ds.attrs[u'units'] = u'arbitrary'
            ds.attrs[u'long_name'] = u'Intensity (arbitrary units)'    # suggested X axis plot label
            # the following are to enable compatibility with Nika canSAS loading
           # ds.attrs[u'signal'] = 1
            #ds.attrs[u'axes'] = u'Qx,Qy'

            # Y axis data
            ds = nxdata.create_dataset(u'Qx', data=data['qx'][0])
            ds.attrs[u'units'] = u'1/angstrom'
            ds.attrs[u'long_name'] = u'Qx (A^-1)'    # suggested Y axis plot label

            ds = nxdata.create_dataset(u'Qy', data=data['qy'][0])
            ds.attrs[u'units'] = u'1/angstrom'
            ds.attrs[u'long_name'] = u'Qy (A^-1)'    # suggested Y axis plot label


        if write_stack_qxy:
            # create the NXdata group for I(Qx,Qy)
            nxdata = nxentry.create_group(u'sasdata_energyseries')
            nxdata.attrs[u'NX_class'] = u'NXdata'
            nxdata.attrs[u'canSAS_class'] = u'SASdata'
            nxdata.attrs[u'signal'] = u'I'      # Y axis of default plot
            nxdata.attrs[u'I_axes'] = u'Qx,Qy,E'         # X axis of default plot
            nxdata.attrs[u'Q_indices'] = '[0,1]'   # use "mr" as the first dimension of I00

            # X axis data
            ds = nxdata.create_dataset(u'I', data=np.swapaxes(np.swapaxes(data['img'],0,1),1,2))
            ds.attrs[u'units'] = u'arbitrary'
            ds.attrs[u'long_name'] = u'Simulated Intensity (arbitrary units)'    # suggested X axis plot label

            # Y axis data
            ds = nxdata.create_dataset(u'Qx', data=data['qx'][0])
            ds.attrs[u'units'] = u'1/angstrom'
            ds.attrs[u'long_name'] = u'Qx (A^-1)'    # suggested Y axis plot label

            ds = nxdata.create_dataset(u'Qy', data=data['qy'][0])
            ds.attrs[u'units'] = u'1/angstrom'
            ds.attrs[u'long_name'] = u'Qy (A^-1)'    # suggested Y axis plot label


            ds = nxdata.create_dataset(u'E', data=coords['energy'])
            ds.attrs[u'units'] = u'eV'
            ds.attrs[u'long_name'] = u'Simulation Energy (eV)'    # suggested Y axis plot label
        if write_stack_qphi:
            # create the NXdata group for I(Q,phi)
            nxdata = nxentry.create_group(u'sasdata_unwrap')
            nxdata.attrs[u'NX_class'] = u'NXdata'
            nxdata.attrs[u'canSAS_class'] = u'SASdata'
            nxdata.attrs[u'signal'] = u'I'      # Y axis of default plot
            nxdata.attrs[u'I_axes'] = u'E,chi,Q'         # X axis of default plot
            nxdata.attrs[u'Q_indices'] = [2]   # use "mr" as the first dimension of I00

            # X axis data
            ds = nxdata.create_dataset(u'I', data=data['imgu'])
            ds.attrs[u'units'] = u'arbitrary'
            ds.attrs[u'long_name'] = u'Simulated Intensity (arbitrary units)'    # suggested X axis plot label

            # Y axis data
            ds = nxdata.create_dataset(u'Q', data=coords['q'])
            ds.attrs[u'units'] = u'1/angstrom'
            ds.attrs[u'long_name'] = u'Q (A^-1)'    # suggested Y axis plot label

            ds = nxdata.create_dataset(u'chi', data=coords['chi'])
            ds.attrs[u'units'] = u'degree'
            ds.attrs[u'long_name'] = u'azimuthal angle chi (deg)'    # suggested Y axis plot label


            ds = nxdata.create_dataset(u'E', data=coords['energy'])
            ds.attrs[u'units'] = u'eV'
            ds.attrs[u'long_name'] = u'Simulation Energy (eV)'    # suggested Y axis plot label
        if write_oned_traces:
            # create the NXdata group for I(Q,E) at two fixed orientations horizontal and vertical
            nxdata = nxentry.create_group(u'sasdata_horizontal')
            nxdata.attrs[u'NX_class'] = u'NXdata'
            nxdata.attrs[u'canSAS_class'] = u'SASdata'
            nxdata.attrs[u'signal'] = u'I'      # Y axis of default plot
            nxdata.attrs[u'I_axes'] = u'E,Q'         # X axis of default plot
            nxdata.attrs[u'Q_indices'] = [1]   # use "mr" as the first dimension of I00

            # X axis data
            ds = nxdata.create_dataset(u'I', data=data['Ihoriz'])
            ds.attrs[u'units'] = u'arbitrary'
            ds.attrs[u'long_name'] = u'Simulated Intensity (arbitrary units)'    # suggested X axis plot label

            # Y axis data
            ds = nxdata.create_dataset(u'Q', data=coords['q'])
            ds.attrs[u'units'] = u'1/angstrom'
            ds.attrs[u'long_name'] = u'Q (A^-1)'    # suggested Y axis plot label

            ds = nxdata.create_dataset(u'E', data=coords['energy'])
            ds.attrs[u'units'] = u'eV'
            ds.attrs[u'long_name'] = u'Simulated Photon Energy (eV)'    # suggested Y axis plot label

             # create the NXdata group for I(Q,E) at two fixed orientations horizontal and vertical
            nxdata = nxentry.create_group(u'sasdata_vertical')
            nxdata.attrs[u'NX_class'] = u'NXdata'
            nxdata.attrs[u'canSAS_class'] = u'SASdata'
            nxdata.attrs[u'signal'] = u'I'      # Y axis of default plot
            nxdata.attrs[u'I_axes'] = u'E,Q'         # X axis of default plot
            nxdata.attrs[u'Q_indices'] = [1]   # use "mr" as the first dimension of I00

            # X axis data
            ds = nxdata.create_dataset(u'I', data=data['Ivert'])
            ds.attrs[u'units'] = u'arbitrary'
            ds.attrs[u'long_name'] = u'Simulated Intensity (arbitrary units)'    # suggested X axis plot label

            # Y axis data
            ds = nxdata.create_dataset(u'Q', data=coords['q'])
            ds.attrs[u'units'] = u'1/angstrom'
            ds.attrs[u'long_name'] = u'Q (A^-1)'    # suggested Y axis plot label

            ds = nxdata.create_dataset(u'E', data=coords['energy'])
            ds.attrs[u'units'] = u'eV'
            ds.attrs[u'long_name'] = u'Simulated Photon Energy (eV)'    # suggested Y axis plot label

       
        # create the NXinstrument metadata group
        nxinstr = nxentry.create_group(u'instrument')
        nxinstr.attrs[u'NX_class'] = u'NXinstrument'
        nxinstr.attrs[u'canSAS_class'] = u'SASinstrument'

        nxprocess = nxinstr.create_group(u'simulation_engine')
        nxprocess.attrs[u'NX_class'] = u'NXprocess'
        nxprocess.attrs[u'canSAS_class'] = u'SASprocess'
        nxprocess.attrs[u'name'] = u'CyRSoXS Simulation Engine'
        nxprocess.attrs[u'date'] = timestamp # @TODO: get timestamp from simulation run and embed here.
        nxprocess.attrs[u'description'] = u'Simulation of RSoXS pattern from optical constants del/beta and morphology'

        sim_notes = nxprocess.create_group(u'NOTE')
        sim_notes.attrs[u'NX_class'] = u'NXnote'

        sim_notes.attrs[u'description'] = u'Simulation Engine Input Parameters/Run Data'
        sim_notes.attrs[u'author'] = u'CyRSoXS PostProcessor'
        sim_notes.attrs[u'data'] = u'Run metadata goes here' #@TODO

        for key in config:
            if 'Energy' in key:
                units = u'eV'
            elif 'Angle' in key:
                units = u'degree'
            elif 'PhysSize' in key:
                units = u'nm' #@TODO: is this correct?
            else:
                units = u''

            metads = sim_notes.create_dataset(key,(config[key],),dtype='f')
            metads.attrs[u'units'] = units
        nxsample = nxentry.create_group(u'sample')
        nxsample.attrs[u'NX_class'] = u'NXsample'
        nxsample.attrs[u'canSAS_class'] = u'SASsample'
        
        nxsample.attrs[u'name'] = 'SAMPLE NAME GOES HERE'
        nxsample.attrs[u'description'] = 'SAMPLE DESCRIPTION GOES HERE'
        nxsample.attrs[u'type'] = 'simulated data'

        #comp.create_dataset
        
    print("wrote file:", fileName)
    if 'pix_x' in xr.dimensions:
        pass
    elif 'q' in xr.dimensions:
        pass
    else:
        raise NotImplementedError(f'I do not support xarrays with dimensions of {xr.dimensions}')
    
def load(path):
    if type(path) is str:
        raise NotImplementedError

'''
            