import numpy as np
from collections import defaultdict
from copy import deepcopy
import pandas as pd
import xarray as xr
from .optional_dependencies import requires_optional, check_optional_dependency, warn_if_missing

# Check for optional dependencies
HAS_SKIMAGE = check_optional_dependency('scikit-image')

if HAS_SKIMAGE:
    import skimage.morphology
else:
    warn_if_missing('scikit-image')


def scaleAndMask(raw_xr, mask_hi=True, mask_lo=True, exposure_cutoff_hi=45000, exposure_cutoff_lo=20, close_mask=True):
    """
    Scale and mask the input data array.

    Note: The close_mask parameter requires scikit-image to be installed.
    If not installed, close_mask will be ignored.
    """
    if close_mask and not HAS_SKIMAGE:
        warnings.warn("scikit-image is not installed. close_mask will be ignored.", UserWarning)
        close_mask = False

    groupby_dims = []
    for dim in raw_xr.coords['system'].unstack('system').coords.keys():
        if dim not in ['filenumber','exposure']:
            groupby_dims.append(dim)
    print(f'Grouping by: {groupby_dims}')
    
    data_rows, dest_coords = hdr_recurse(raw_xr, groupby_dims, {},
                             mask_hi=mask_hi, mask_lo=mask_lo,
                             exposure_cutoff_hi=exposure_cutoff_hi, exposure_cutoff_lo=exposure_cutoff_lo,
                             close_mask=close_mask)
    
    index = pd.MultiIndex.from_arrays(list(dest_coords.values()), names=list(dest_coords.keys()))
    index.name = 'system'
    out = xr.concat(data_rows, dim=index)
    return out


def hdr_recurse(input_xr, groupby_dims, dest_coords, **kw):
    data_rows_accumulator = []
    dest_coords_accumulator = defaultdict(list)
    if len(groupby_dims) > 0:
        target_dim = groupby_dims.pop()
        print(f'Grouping on {target_dim}')
        print(f'  number of groups {len(input_xr.groupby(target_dim))}')
        for xre in input_xr.groupby(target_dim, squeeze=False):
            dest_coords[target_dim] = (xre[0])
            print(f'    Launching workOrRecurse with xr, groupby {groupby_dims}, coords {dest_coords}')
            data_rows_new, dest_coords_new = hdr_recurse(xre[1], deepcopy(groupby_dims), deepcopy(dest_coords), **kw)
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
    else:
        return hdr_work(input_xr, groupby_dims, dest_coords, **kw)


@requires_optional('scikit-image')
def hdr_work(input_xr, groupby_dims, dest_coords, mask_hi=True, mask_lo=True, 
            exposure_cutoff_hi=45000, exposure_cutoff_lo=20, close_mask=True):
    """
    Process high dynamic range data.

    Note: This function requires scikit-image to be installed for mask closing operations.
    """
    data = input_xr.data
    exposures = input_xr.coords['system'].unstack('system').coords['exposure']
    
    # Create masks for high and low cutoffs
    if mask_hi:
        mask_hi = data > exposure_cutoff_hi
        if close_mask:
            mask_hi = skimage.morphology.binary_closing(mask_hi)
    else:
        mask_hi = np.zeros_like(data, dtype=bool)
        
    if mask_lo:
        mask_lo = data < exposure_cutoff_lo
        if close_mask:
            mask_lo = skimage.morphology.binary_closing(mask_lo)
    else:
        mask_lo = np.zeros_like(data, dtype=bool)
        
    # Scale data by exposure time
    data_scaled = data / exposures[:, np.newaxis, np.newaxis]
    
    # Apply masks
    data_scaled[mask_hi] = np.nan
    data_scaled[mask_lo] = np.nan
    
    # Take median of non-masked values
    output = np.nanmedian(data_scaled, axis=0)
    
    # Create output DataArray
    for dim in input_xr.coords['system'].unstack('system').coords.keys():
        if dim not in ['filenumber', 'exposure']:
            dest_coords[dim] = input_xr.coords['system'].unstack('system').coords[dim][0]
    
    return xr.DataArray(output, dims=input_xr.dims[1:], coords={k: input_xr.coords[k] for k in input_xr.dims[1:]})