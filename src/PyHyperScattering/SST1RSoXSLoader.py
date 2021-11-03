from PIL import Image
from PyHyperScattering.FileLoader import FileLoader
import os
import pathlib
import xarray as xr
import pandas as pd
import datetime
import warnings
import json
#from pyFAI import azimuthalIntegrator
import numpy as np


class SST1RSoXSLoader(FileLoader):
    '''
    Loader for TIFF files from NSLS-II SST1 RSoXS instrument

    '''
    file_ext = '(.*?)primary(.*?).tiff'
    md_loading_is_quick = True
    pix_size_1 = 0.06
    pix_size_2 = 0.06

    def __init__(self,corr_mode=None,user_corr_func=None,dark_pedestal=100,exposure_offset=0,constant_md={}):
        '''
        Args:
            corr_mode (str): origin to use for the intensity correction.  Can be 'expt','i0','expt+i0','user_func','old',or 'none'
            user_corr_func (callable): takes the header dictionary and returns the value of the correction.
            dark_pedestal (numeric): value to subtract(/add, if negative) to the whole image.  this should match the instrument setting for suitcased tiffs, typically 100.
            exposure_offset (numeric): value to add to the exposure time.  Measured at 2ms with the piezo shutter in Dec 2019 by Jacob Thelen, NIST
            constant_md (dict): values to insert into every metadata load. 
        '''

        if corr_mode == None:
            warnings.warn("Correction mode was not set, not performing *any* intensity corrections.  Are you sure this is "+
                          "right? Set corr_mode to 'none' to suppress this warning.",stacklevel=2)
            self.corr_mode = 'none'
        else:
            self.corr_mode = corr_mode


        self.constant_md = constant_md

        self.dark_pedestal = dark_pedestal
        self.user_corr_func = user_corr_func
        self.exposure_offset = exposure_offset
        # self.darks = {}
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



    def loadSingleImage(self,filepath,coords=None, return_q=False):
        '''
        HELPER FUNCTION that loads a single image and returns an xarray with either pix_x / pix_y dimensions (if return_q == False) or qx / qy (if return_q == True)


        Args:
            filepath (Pathlib.path): path of the file to load
            coords (dict-like): coordinate values to inject into the metadata
            return_q (bool): return qx / qy coords.  If false, returns pixel coords.

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
            warnings.warn(f'Correction value is negative: {corr} with headers {headerdict}.',stacklevel=2)
            corr = abs(corr)


        # # step 2: dark subtraction
        # this is already done in the suitcase, but we offer the option to add/subtract a pedestal.
        image_data = (np.array(img)-self.dark_pedestal)/corr
        if return_q:
            qpx = 2*np.pi*60e-6/(headerdict['sdd']/1000)/(headerdict['wavelength']*1e10)
            qx = (np.arange(1,img.size[0]+1)-headerdict['beamcenter_y'])*qpx
            qy = (np.arange(1,img.size[1]+1)-headerdict['beamcenter_x'])*qpx
            # now, match up the dims and coords
            return xr.DataArray(image_data,dims=['qy','qx'],coords={'qy':qy,'qx':qx},attrs=headerdict)
        else:
            return xr.DataArray(image_data,dims=['pix_x','pix_y'],attrs=headerdict)

    def read_json(self,jsonfile):
        json_dict = {}
        with open(jsonfile) as f:
            data = json.load(f)
            meas_time =datetime.datetime.fromtimestamp(data[1]['time'])
            json_dict['sample_name'] = data[1]['sample_name']
        if data[1]['RSoXS_Main_DET'] == 'SAXS':
            json_dict['rsoxs_config'] = 'saxs'
            # discrepency between what is in .json and actual
            if (meas_time > datetime.datetime(2020,12,1)) and (meas_time < datetime.datetime(2021,1,15)):
                json_dict['beamcenter_x'] = 489.86
                json_dict['beamcenter_y'] = 490.75
                json_dict['sdd'] = 521.8
            elif (meas_time > datetime.datetime(2020,11,16)) and (meas_time < datetime.datetime(2020,12,1)):
                json_dict['beamcenter_x'] = 371.52
                json_dict['beamcenter_y'] = 491.17
                json_dict['sdd'] = 512.12
            else:
                json_dict['beamcenter_x'] = data[1]['RSoXS_SAXS_BCX']
                json_dict['beamcenter_y'] = data[1]['RSoXS_SAXS_BCY']
                json_dict['sdd'] = data[1]['RSoXS_SAXS_SDD']

        elif data[1]['RSoXS_Main_DET'] == 'WAXS':
            json_dict['rsoxs_config'] = 'waxs'
            if (meas_time > datetime.datetime(2020,11,16)) and (meas_time < datetime.datetime(2021,1,15)):
                json_dict['beamcenter_x'] = 400.46
                json_dict['beamcenter_y'] = 530.99
                json_dict['sdd'] = 38.745
            else:
                json_dict['beamcenter_x'] = data[1]['RSoXS_WAXS_BCX'] # 399 #
                json_dict['beamcenter_y'] = data[1]['RSoXS_WAXS_BCY'] # 526
                json_dict['sdd'] = data[1]['RSoXS_WAXS_SDD']

        else:
            json_dict['rsoxs_config'] == 'unknown'
            warnings.warn('RSoXS_Config is neither SAXS or WAXS. Check json file',stacklevel=2)

        if json_dict['sdd'] == None:
            warnings.warn('sdd is None, reverting to default values. Check json file',stacklevel=2)
            if json_dict['rsoxs_config'] == 'waxs':
                json_dict['sdd'] = 38.745
            elif json_dict['rsoxs_config'] == 'saxs':
                json_dict['sdd'] = 512.12
        if json_dict['beamcenter_x'] == None:
            warnings.warn('beamcenter_x/y is None, reverting to default values. Check json file',stacklevel=2)
            if json_dict['rsoxs_config'] == 'waxs':
                json_dict['beamcenter_x'] = 400.46
                json_dict['beamcenter_y'] = 530.99
            elif json_dict['rsoxs_config'] == 'saxs':
                json_dict['beamcenter_x'] = 371.52
                json_dict['beamcenter_y'] = 491.17
        return json_dict

    def read_baseline(self,baseline_csv):
        baseline_dict = {}
        df_baseline = pd.read_csv(baseline_csv)
        baseline_dict['sam_x'] = round(df_baseline['RSoXS Sample Outboard-Inboard'][0],4)
        baseline_dict['sam_y'] = round(df_baseline['RSoXS Sample Up-Down'][0],4)
        baseline_dict['sam_z'] = round(df_baseline['RSoXS Sample Downstream-Upstream'][0],4)
        baseline_dict['sam_th'] = round(df_baseline['RSoXS Sample Rotation'][0],4)

        return baseline_dict

    def read_primary(self,primary_csv,json_dict,seq_num):
        primary_dict = {}
        df_primary = pd.read_csv(primary_csv)
        if json_dict['rsoxs_config'] == 'waxs':
            primary_dict['exposure'] = df_primary['RSoXS Shutter Opening Time (ms)'][seq_num]
        elif json_dict['rsoxs_config'] == 'saxs':
            primary_dict['exposure'] = df_primary['RSoXS Shutter Opening Time (ms)'][seq_num]
        else:
            warnings.warn('Check rsoxs_config in json file',stacklevel=2)

        primary_dict['energy'] = round(df_primary['en_energy_setpoint'][seq_num],4)
        primary_dict['polarization'] = df_primary['en_polarization_setpoint'][seq_num]

        return primary_dict


    def loadMd(self,filepath):
        # get sequence number of image for primary csv
        fname = os.path.basename(filepath)
        split_fname = fname.split('-')
        seq_num = int(split_fname[-1][:-5])
        scan_id = split_fname[0]

        # This allows for passing just the filename without the full path
        dirPath = os.path.dirname(filepath)

        if dirPath == '':
            cwd = pathlib.Path('.').absolute()

            json_fname = list(cwd.glob('*.jsonl'))
            json_dict = self.read_json(json_fname[0])

            baseline_fname = list(cwd.glob('*baseline.csv'))
            baseline_dict = self.read_baseline(baseline_fname[0])


            primary_path = os.path.dirname(cwd)
            primary_fname = list(primary_path.glob(f'{scan_id}*primary.csv'))
            primary_dict = self.read_primary(primary_fname[0],json_dict,seq_num)
        else:
            json_fname = list(pathlib.Path(dirPath).glob('*jsonl'))
            json_dict = self.read_json(json_fname[0])

            baseline_fname = list(pathlib.Path(dirPath).glob('*baseline.csv'))
            baseline_dict = self.read_baseline(baseline_fname[0])

            primary_path = os.path.dirname(dirPath)
            primary_fname = list(pathlib.Path(primary_path).glob(f'{scan_id}*primary.csv'))
            primary_dict = self.read_primary(primary_fname[0],json_dict,seq_num)

        headerdict = {**primary_dict,**baseline_dict,**json_dict}

        headerdict['wavelength'] = 1.239842e-6 / headerdict['energy']
        headerdict['seq_num'] = seq_num
        headerdict['sampleid'] = scan_id
        try:
            headerdict['dist'] = headerdict['sdd'] / 1000
        except TypeError:
            print(headerdict['sdd'])
        
        headerdict['pixel1'] = self.pix_size_1 / 1000
        headerdict['pixel2'] = self.pix_size_2 / 1000

        headerdict['poni1'] = headerdict['beamcenter_y'] * headerdict['pixel1']
        headerdict['poni2'] = headerdict['beamcenter_x'] * headerdict['pixel2']

        headerdict['rot1'] = 0
        headerdict['rot2'] = 0
        headerdict['rot3'] = 0

        headerdict.update(self.constant_md)
        return headerdict

    def peekAtMd(self,filepath):
        return self.loadMd(filepath)
