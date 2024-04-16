import itertools
import warnings
import xarray as xr
import numpy as np
import math
from typing import Union


@xr.register_dataarray_accessor('rsoxs')
class RSoXS:
    '''Contains methods for common RSoXS/P-RSoXS data analysis operations'''

    def __init__(self, xr_obj):
        self._obj = xr_obj

        self._pyhyper_type = 'reduced'
        try:
            self._chi_min = np.min(xr_obj.chi)
            self._chi_max = np.max(xr_obj.chi)
            self._chi_range = [self._chi_min, self._chi_max]
        except AttributeError:
            self._pyhyper_type = 'raw'

    def slice_chi(self, chi, chi_width=5, do_avg: bool = True):
        '''Slice and average an xarray along the chi coordinate

        Accounts for wrapping of chi values beyond ends of the range.

        Parameters
        ----------
        chi : numeric
            chi about which slice should be centered, in deg
        chi_width : numeric, optional
            width of slice in each direction in deg, by default 5
        do_avg : bool, optional
            if true, averages along the chi dimension, by default True

        Returns
        -------
        xr.DataArray
            datarray averaged along the specified chi slice
        '''
        slice_begin = chi - chi_width
        slice_end = chi + chi_width

        '''
            cases to handle:
            1) wrap-around slice.  begins before we start and ends after we end.  return whole array and warn.
            2) wraps under, ends inside
            3) wraps under, ends under --> translate both coords
            4) wraps over, ends inside
            5) wraps over, ends over --> translate both coords.
            6) begins inside, ends inside
        '''

        if slice_begin < self._chi_min and slice_end < self._chi_min:
            # case 3
            nshift = math.floor((self._chi_min - slice_end) / 360) + 1
            slice_begin += 360 * nshift
            slice_end += 360 * nshift
        elif slice_begin > self._chi_max and slice_end > self._chi_max:
            # case 5
            nshift = math.floor((slice_begin - self._chi_max) / 360) + 1
            slice_begin -= 360 * nshift
            slice_end -= 360 * nshift

        if slice_begin < self._chi_min and slice_end > self._chi_max:
            # case 1
            warnings.warn(
                f'Chi slice specified from {slice_begin} to {slice_end}, which exceeds range of {self._chi_min} to {self._chi_max}.  Returning mean across all values of chi.',
                stacklevel=2,
            )
            selector = np.ones_like(self._obj.chi, dtype=bool)
        elif slice_begin < self._chi_min and slice_end < self._chi_max:
            # wrap-around _chi_min: case 2
            selector = np.logical_and(self._obj.chi >= self._chi_min, self._obj.chi <= slice_end)
            selector = np.logical_or(
                selector,
                np.logical_and(
                    self._obj.chi <= self._chi_max,
                    self._obj.chi >= (self._chi_max - (self._chi_min - slice_begin) + 1),
                ),
            )
        elif slice_end > self._chi_max and slice_begin > self._chi_min:
            # wrap-around _chi_max: case 4
            selector = np.logical_and(self._obj.chi <= self._chi_max, self._obj.chi >= slice_begin)
            selector = np.logical_or(
                selector,
                np.logical_and(
                    self._obj.chi >= self._chi_min,
                    self._obj.chi <= (self._chi_min + (slice_end - self._chi_max) - 1),
                ),
            )
        else:
            # simple slice, case 6, hooray
            selector = np.logical_and(self._obj.chi >= slice_begin, self._obj.chi <= slice_end)

        if do_avg == True:
            return self._obj.isel({'chi': selector}).mean('chi')
        else:
            return self._obj.isel({'chi': selector})

    def slice_q(self, q, q_width=None):
        '''Slice and average an xarray along the q coordinate

        Parameters
        ----------
        q : numeric
            q value about which slice should be centered
        q_width :numeric, optional
            width of slice in each direction, in q units, defaults to 0.1 * q

        Returns
        -------
        xr.DataArray
            datarray averaged along the specified q slice
        '''
        img = self._obj
        if q_width == None:
            q_width = 0.1 * q
        return img.sel(q=slice(q - q_width, q + q_width)).mean('q')

    def select_chi(self, chi, method='nearest'):
        '''Enables xarray subsetting by chi values that are out of range

        If chi is outside the dataset, this will adjust it by adding or subtracting multiples of 360 until it falls in the valid range.

        Parameters
        ----------
        chi : numeric
            target chi value to apply xr.DataArray.sel() with
        method : str, optional
            search method to pass to xr.DataArray.sel(), by default 'nearest'

        Returns
        -------
        xr.DataArray
            DataArray whose data match the provided chi value (adjusted into range)
        '''
        # Select data along the chi dimension using the specified method
        return self._obj.sel(chi=self._reRange_chi(chi), method=method)

    def select_q(self, q, method='nearest'):
        '''Alias of the xr.DataArray .sel method for selection in q

        Parameters
        ----------
        q : numeric
            Desired q value
        method : str, optional
            method for inexact matches, by default 'nearest'

        Returns
        -------
        xr.DataArray
            DataArray whose data match the provided q value
        '''
        return self._obj.sel(q=q, method=method)

    def select_pol(self, pol, method='nearest'):
        '''Alias of the xr.DataArray .sel method for selection in polarization

        Parameters
        ----------
        pol : numeric
            Desired polarization value
        method : str, optional
            method for inexact matches, by default 'nearest'

        Returns
        -------
        xr.DataArray
            DataArray whose data match the provided polarization value
        '''
        return self._obj.sel(polarization=pol, method=method)

    def AR(
        self,
        chi_center1_deg: Union[int, float] = None,
        chi_center2_deg: Union[int, float] = None,
        chi_width_deg: Union[int, float] = 45,
        use_reflection: bool = True,
        infer_chi_from_pol: bool = True,
        use_paired_scans: bool = True,
        paired_normalization_energy: Union[int, float] = None,
        AR_return_method: str = 'average',
        verbose: bool = False,
    ):
        '''Returns a DataArray containing the RSoXS Anisotropic Ratio of a single scan or polarized
        pair of scans

        Single Image AR defined as AR1 = (I_chi1 - Ichi2)/ (I_chi1 + I_chi2)
        I_x represents the integration of a section in q, chi space centered on chi = x
        Conventionally, chi1 and chi2 are chosen parallel and perpendicular to the beam
        polarization, respectively

        Parameters
        ----------
        chi_center1_deg : Union[int, float], optional
            Center (in degrees chi) of the first chi wedge to integrate, defaults to 0 unless
            infer_ChiFromPol is True
        chi_center2_deg : Union[int, float], optional
            Center (in degrees chi) of the second chi wedge to integrate, defaults to
            (chi_center1 + 90) unless two scans are provided and infer_ChiFromPol is True
        chi_width_deg : Union[int, float], optional
            width of slice in each direction in deg, by default 45
        use_reflection : bool, optional
            if true, also integrate chi wedges centered 180 degrees from the chi_centers,
            by default False
        infer_chi_from_pol : bool, optional
            if true, and if data contains a polarization dim, sets chi1 = pol1, chi2 = pol2,
            by default False
        use_paired_scans : bool, optional
            if true, calculate the AR using scans at two polarizations, by default False
        paired_normalization_energy : Union[int, float], optional
            if set, normalizes each polarization's AR at a given energy.
            THIS EFFECTIVELY FORCES THE AR TO 0 AT THIS ENERGY., by default None
        AR_return_method : str, optional
            if 'average', return a single value AR = (AR1 + AR2)/2, if 'separate',
            return a tuple (AR1, AR2), if 'components', return a dict containing the integrated
            wedge data, by default 'average'
        verbose : bool, optional
            if true, runs the checkAR command, plotting wedges, and prints intermediate values,
            by default False

        Returns
        -------
        xr.DataArray
            DataArray, tuple of DataArrays, opr dict of DataArrays containing AR values or components
        '''
        # Build Report for diagnosing AR quality
        reportString = "Anisotropic Ratio Calculation:\n"

        # Determine wedge centers criteria

        # User wants to infer chi centers from beam polarization metadata
        if infer_chi_from_pol == True:

            # Make sure the user didnt provide chi1 and chi2 already
            if chi_center1_deg is not None:
                raise ValueError(
                    "infer_ChiFromPol must be False if you provide chi values manually"
                )

            # If we have polarization as a data dimensions
            if 'polarization' in self._obj.dims:
                # get polarization values
                num_pol_vals = self._obj.coords['polarization'].values.tolist()
                # if we have 1, save as as chi1,
                if len(num_pol_vals) == 1:
                    chi_center1_deg = num_pol_vals[0]
                # if two save as chi1 and chi2
                elif len(num_pol_vals) == 2:
                    chi_center1_deg = num_pol_vals[0]
                    chi_center2_deg = num_pol_vals[1]
                # Else raise error
                else:
                    raise NotImplementedError(
                        f'Can only infer if 1 or 2 polarization values are provided, found {len(num_pol_vals)} values'
                    )
            # Error if we don't have polarization as a data dimensions
            else:
                raise NotImplementedError(
                    'Cannot infer chi unless polarization values are provided as data dimensions'
                )

        # Hardcode default values/relationships between chi1 and chi2
        # If no info provided, default chi_1 to 0
        if chi_center1_deg is None:
            chi_center1_deg = 0
        # Make sure it fits in the valid range
        else:
            chi_center1_deg = self._reRange_chi(chi_center1_deg)

        # If no info provided, chi_2 is chi_1 - 90
        if chi_center2_deg is None:
            chi_center2_deg = self._reRange_chi(chi_center1_deg + 90)

        # Add to report
        reportString += f"\tchi_width = {chi_width_deg} deg\n"
        reportString += (
            f"\tchi_center1 = {chi_center1_deg:.3f} deg, chi_center2 = {chi_center2_deg:.3f} deg\n"
        )

        # If the user wants to use reflected wedges
        if use_reflection == True:
            chi_center1r_deg = self._reRange_chi(chi_center1_deg + 180)
            chi_center2r_deg = self._reRange_chi(chi_center2_deg + 180)
            reportString += f"\tchi_center1r = {chi_center1r_deg:.3f} deg, chi_center2r = {chi_center2r_deg:.3f} deg\n"

        # Warn the user if the wedges are not perpendicular +- 2 deg
        # Calculate the absolute circular difference between chi_center1 and chi_center2
        chi_center_difference = min(
            (chi_center1_deg - chi_center2_deg) % 360, (chi_center2_deg - chi_center1_deg) % 360
        )
        max_allowed_difference = 2  # deg
        # Check if the absolute circular difference is within the specified range
        if not (
            (90 - max_allowed_difference) <= chi_center_difference <= (90 + max_allowed_difference)
        ):
            warnings.warn(
                "The difference between chi_center1 and chi_center2 is not within 90 ± 2 degrees.",
                stacklevel=2,
            )

        # Compute anisotropy component integrals
        # Note that I1 and I2 corrspond to integrals centered along the chi1 and chi2 directions
        # (decoupled from para, perp formalism)

        # extract 2 wedges to determine I_1, I_2
        if use_reflection == False:
            I_1 = self.slice_chi(chi=chi_center1_deg, chi_width=chi_width_deg, do_avg=False)
            I_1.name = 'I_chi1'
            I_1.attrs['chi_center'] = chi_center1_deg
            I_1.attrs['chi_width'] = chi_width_deg
            I_2 = self.slice_chi(chi=chi_center2_deg, chi_width=chi_width_deg, do_avg=False)
            I_2.name = 'I_chi2'
            I_2.attrs['chi_center'] = chi_center2_deg
            I_2.attrs['chi_width'] = chi_width_deg
            # check for overlap
            self._checkChiOverlap([I_1, I_2])

            # Compute Integrals and AR
            I_1 = I_1.mean('chi', keep_attrs=True)
            I_2 = I_2.mean('chi', keep_attrs=True)

        # else extract 4 wedges
        else:
            I_1 = self.slice_chi(chi=chi_center1_deg, chi_width=chi_width_deg, do_avg=False)
            I_1.name = 'I_chi1'
            I_1.attrs['chi_center'] = chi_center1_deg
            I_1.attrs['chi_width'] = chi_width_deg
            I_2 = self.slice_chi(chi=chi_center2_deg, chi_width=chi_width_deg, do_avg=False)
            I_2.name = 'I_chi2'
            I_2.attrs['chi_center'] = chi_center2_deg
            I_2.attrs['chi_width'] = chi_width_deg
            I_1r = self.slice_chi(chi=chi_center1r_deg, chi_width=chi_width_deg, do_avg=False)
            I_1r.name = 'I_chi1reflected'
            I_1r.attrs['chi_center'] = chi_center1r_deg
            I_1r.attrs['chi_width'] = chi_width_deg
            I_2r = self.slice_chi(chi=chi_center2r_deg, chi_width=chi_width_deg, do_avg=False)
            I_2r.name = 'I_chi2reflected'
            I_2r.attrs['chi_center'] = chi_center2r_deg
            I_2r.attrs['chi_width'] = chi_width_deg

            # check for overlap
            self._checkChiOverlap([I_1, I_2, I_1r, I_2r])

            # Compute Integrals and AR
            I_1 = I_1.mean('chi', keep_attrs=True)
            I_2 = I_2.mean('chi', keep_attrs=True)
            I_1r = I_1r.mean('chi', keep_attrs=True)
            I_2r = I_2r.mean('chi', keep_attrs=True)

        # Calculate AR for the simple case (single polarization)
        if not use_paired_scans:

            # add components to report
            reportString += (
                f"\tI_1 Total Mean: {I_1.mean():.3f}\n" f"\tI_2 Total Mean: {I_2.mean():.3f}\n"
            )

            # Simple case, all frames are individually constrainted to chi1, chi2
            if use_reflection == True:
                # add reflected components to report
                reportString += (
                    f"\tI_1r Total Mean: {I_1r.mean():.3f}\n"
                    f"\tI_2r Total Mean: {I_2r.mean():.3f}\n"
                )

                # Merge reflected components
                I_1 = I_1 + I_1r
                I_2 = I_2 + I_2r

            # Compute AR
            AR = (I_1 - I_2) / (I_1 + I_2)

            # add to report
            reportString += f"\tAR Total Mean: {AR.mean():.3f}\n"

            # if report requested
            if verbose == True:
                self._checkAR()
                print(reportString)

            # return AR in appropriate format
            if AR_return_method.lower() == 'average':
                return AR
            elif AR_return_method.lower() == 'separate':
                return (AR,)
            elif AR_return_method.lower() == 'components':
                if use_reflection == False:
                    return dict(I_1=I_1, I_2=I_2)
                else:
                    return dict(I_1=I_1, I_2=I_2, I_1r=I_1r, I_2r=I_2r)

        # Calculate AR for the case of two polarizations
        elif use_paired_scans:

            # add components to report
            f"\tI_1 Total Mean (pol 1 | pol 2): {I_1.isel(polarization=0).mean():.3f}"
            f" | {I_1.isel(polarization=1).mean():.3f}\n"
            f"\tI_2 Total Mean (pol 1 | pol 2): {I_2.isel(polarization=0).mean():.3f}"
            f" | {I_2.isel(polarization=1).mean():.3f}\n"

            if use_reflection == True:
                # add reflected components to report
                reportString += (
                    f"\tI_1r Total Mean (pol 1 | pol 2): {I_1r.isel(polarization=0).mean():.3f}"
                    f" | {I_1r.isel(polarization=1).mean():.3f}\n"
                    f"\tI_2r Total Mean (pol 1 | pol 2): {I_2r.isel(polarization=0).mean():.3f}"
                    f" | {I_2r.isel(polarization=1).mean():.3f}\n"
                )

                # Merge reflected components
                I_1 = I_1 + I_1r
                I_2 = I_2 + I_2r

            # Compute Anisotropic Ratio (AR)
            # As before
            AR1 = ((I_1 - I_2) / (I_1 + I_2)).isel(polarization=0)
            # More complicated, need to remember that AR is defined with respect to polarization direction
            # Here I_1 is actually perpendicular to polarization direction
            AR2 = ((I_2 - I_1) / (I_2 + I_1)).isel(polarization=1)

            # Normalize if requested
            if paired_normalization_energy is not None:
                AR1 = AR1 / AR1.sel(energy=paired_normalization_energy, method='nearest')
                AR2 = AR2 / AR2.sel(energy=paired_normalization_energy, method='nearest')

            if (AR1 < AR2).all() or (AR2 < AR1).all():
                warnings.warn(
                    '''One polarization has a systematically higher/lower AR than the other.  
                    Typically this indicates bad intensity values, or lack of calibration.''',
                    stacklevel=2,
                )

            # add to report
            reportString += (
                f"\tAR Total Mean (pol 1 | pol 2): {AR1.mean():.3f}" f" | {AR2.mean():.3f}\n"
            )

            # if report requested
            if verbose == True:
                self._checkAR()
                print(reportString)

            # return AR in appropriate format
            if AR_return_method.lower() == 'average':
                return (AR1 + AR2) / 2
            elif AR_return_method.lower() == 'separate':
                return (AR1, AR2)
            elif AR_return_method.lower() == 'components':
                if use_reflection == False:
                    return dict(I_1=I_1, I_2=I_2)
                else:
                    return dict(I_1=I_1, I_2=I_2, I_1r=I_1r, I_2r=I_2r)

    def collate_AR_stack(sample, energy):
        raise NotImplementedError(
            'This is a stub function. Should return tuple of the two polarizations, but it does not yet.'
        )
        '''for sam in data_idx.groupby('sample'):
            print(f'Processing for {sam[0]}')
            for enset in sam[1].groupby('energy'):
                print(f'    Processing energy group {enset[0]}')
                pol90 = enset[1][enset[1]['pol']==90.0].num
                pol0 = enset[1][enset[1]['pol']==0.0].num
                print(f'        Pol 0: {pol0}')
                print(f'        Pol 90: {pol90}')'''

    def _reRange_chi(self, chi):
        '''Utility Function to shift chi values to be within valid range

        If chi is outside the dataset, this will adjust it by adding or subtracting multiples of 360 until it falls in the valid range.

        Parameters
        ----------
        chi : numeric
            target chi value to apply xr.DataArray.sel() with

        Returns
        -------
        float
            chi value shifted into valid range
        '''
        # If chi is less than the minimum chi value in the dataset
        if chi < self._chi_min:
            # Calculate the number of shifts needed to bring chi within the valid range and adjust
            nshift = math.floor((self._chi_min - chi) / 360) + 1
            chi += 360 * nshift
        # If chi is greater than the maximum chi value in the dataset
        elif chi > self._chi_max:
            # Calculate the number of shifts needed to bring chi within the valid range and adjust
            nshift = math.floor((chi - self._chi_max) / 360) + 1
            chi -= 360 * nshift
        # Select data along the chi dimension using the specified method
        return chi

    def _checkChiOverlap(
        self,
        l_dataArrays: list,
        tolerance: int = 1,
    ):
        '''Checks for overlapping chi coordinates in a list of DataArrays

        Parameters
        ----------
        l_dataArrays : list of xr.DataArrays
            a list of DataArrays with dimensions chi
        tolerance : int, optional
            The acceptable number of overlapping chi values between any pair of DataArrays, by default 1

        Returns
        -------
        float
            number of overlapping chi indices
        '''
        for array1, array2 in itertools.combinations(l_dataArrays, 2):
            overlapping_values = array1['chi'].where(array1['chi'] == array2['chi'], drop=True)
            if len(overlapping_values) > 1:
                warnings.warn(
                    (
                        f"Caution: {len(overlapping_values)} "
                        f"overlapping chi values found between {array1.name} and {array2.name}, "
                        f"you may wish to run checkAr() for more details"
                    ),
                    stacklevel=2,
                )

    def _checkAR(self):
        pass


'''
    
for img in int_stack:
    f = plt.figure()

    img.sel(chi=slice(-5,5)).unstack('system').mean('chi').plot(label='0 deg ± 5 deg',norm=LogNorm(1e1,1e5))
    plt.title(f'{img.sample_name} @ pol = {float(img.polarization[0])}, chi = 0 deg ± 5 deg')
    plt.legend()
    plt.show()
    plt.savefig(f'2D_chi0_{img.sample_name}_pol{float(img.polarization[0])}.png')
    plt.close()
    img.sel(chi=slice(-95,-85)).unstack('system').mean('chi').plot(label='90 deg ± 5 deg',norm=LogNorm(1e1,1e5))
    plt.title(f'{img.sample_name} @ pol = {float(img.polarization[0])}, chi = 90 deg ± 5 deg')
    plt.legend()
    plt.show()
    plt.savefig(f'2D_chi90_{img.sample_name}_pol{float(img.polarization[0])}.png')
    plt.close()
    '''
