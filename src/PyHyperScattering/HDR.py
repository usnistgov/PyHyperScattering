import numpy as np
import skimage.morphology
from collections import defaultdict
from copy import deepcopy
import pandas as pd
import xarray as xr


def scaleAndMask(raw_xr,mask_hi=True,mask_lo=True,exposure_cutoff_hi=45000,exposure_cutoff_lo=20,close_mask=True):
    groupby_dims = []
    for dim in raw_xr.coords['system'].unstack('system').coords.keys():
        if dim not in ['filenumber','exposure']:
            groupby_dims.append(dim)
    print(f'Grouping by: {groupby_dims}')
    
    data_rows,dest_coords= hdr_recurse(raw_xr,groupby_dims,{},
                             mask_hi=mask_hi,mask_lo=mask_lo,
                             exposure_cutoff_hi=exposure_cutoff_hi,exposure_cutoff_lo=exposure_cutoff_lo,
                             close_mask=close_mask)
    #return data_rows,dest_coords
    index = pd.MultiIndex.from_arrays(list(dest_coords.values()),names=list(dest_coords.keys()))
    index.name = 'system'
    out = xr.concat(data_rows,dim=index)
    return out

def hdr_recurse(input_xr,groupby_dims,dest_coords,**kw):
        data_rows_accumulator = []
        dest_coords_accumulator = defaultdict(list)
        if len(groupby_dims) > 0:
            target_dim = groupby_dims.pop()
            print(f'Grouping on {target_dim}')
            print(f'  number of groups {len(input_xr.groupby(target_dim))}')
            for xre in input_xr.groupby(target_dim,squeeze=False):
                #print(f'    Element {xre[target_dim]}')
                dest_coords[target_dim] = (xre[0])
                print(f'    Launching workOrRecurse with xr, groupby {groupby_dims}, coords {dest_coords}')
                data_rows_new,dest_coords_new = hdr_recurse(xre[1],deepcopy(groupby_dims),deepcopy(dest_coords),**kw)
                for dim in dest_coords_new.keys():
                    if type(dest_coords_new[dim]) is list:
                        for item in dest_coords_new[dim]:
                            dest_coords_accumulator[dim].append(item)
                    else:
                        dest_coords_accumulator[dim].append(dest_coords_new[dim])
                if type(data_rows_new) is list:
                    for item in data_rows_new:
                        data_rows_accumulator.append(item)
                else:
                    data_rows_accumulator.append(data_rows_new)
            return data_rows_accumulator, dest_coords_accumulator
        else: # if there are no more dimensions to unstack
            return hdr_work(input_xr,groupby_dims,dest_coords,**kw)
            
def hdr_work(input_xr,groupby_dims,dest_coords,**kw):
    masked_accumulator = []
    exposure_accumulator = []
    for da in input_xr.groupby('exposure',squeeze=False):
        print(f'        Processing exposure {da[0]}')
        exposure = da[0]
        da = da[1]
        new_data = da.mean('system').values
        if kw['mask_hi']: 
            new_data = np.ma.masked_greater_equal(new_data,kw['exposure_cutoff_hi']/exposure)
            mask_hi_stat = np.sum(new_data.mask.astype(bool))
            print(f"                Masking hi: pixels >= {kw['exposure_cutoff_hi']} cts or {kw['exposure_cutoff_hi']/exposure} cps resulted in {mask_hi_stat} pixels masked")
        if kw['mask_lo']:
            new_data = np.ma.masked_less_equal(new_data,kw['exposure_cutoff_lo']/exposure)
            mask_lo_stat = np.sum(new_data.mask.astype(bool))-mask_hi_stat
            print(f"                Masking lo: pixels <= {kw['exposure_cutoff_lo']} cts or {kw['exposure_cutoff_lo']/exposure} cps resulted in {mask_lo_stat} pixels masked")
        print(f'            masking resulted in {np.sum(new_data.mask.astype(bool))} masked pixels')

        if kw['close_mask']:
            before = new_data.mask.sum()
            new_data.mask = skimage.morphology.binary_closing(new_data.mask)
            print(f'            binary closing completed, masked pixels {before} --> {new_data.mask.sum()}')

        masked_accumulator.append(new_data)
        exposure_accumulator.append(exposure)
    avg = np.ma.average(masked_accumulator,axis=0,weights=exposure_accumulator)
    print(f'            after averaging, masked pixels = {avg.mask.sum()}')

    return xr.DataArray(avg,dims=['pix_x','pix_y'],attrs={}),dest_coords