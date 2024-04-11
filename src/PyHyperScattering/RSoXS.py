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
        chi_center1: Union[int, float] = None,
        chi_center2: Union[int, float] = None,
        chi_width: Union[int, float] = 5,
        reflectWedges: bool = False,
        calc2d: bool = False,
        two_AR: bool = False,
        calc2d_norm_energy: Union[int, float] = None,
        infer_ChiFromPol: bool = False,
        printReport: bool = False,
    ):
        '''Returns a DataArray containing the RSoXS Anisotropic Ratio of a single scan or polarized pair of scans

        Single Image AR defined as AR1 = (I_chi1 - Ichi2)/ (I_chi1 + I_chi2)
        Conventionally, chi1 and chi2 are chosen parallel and perpendicular to the polarization

        Parameters
        ----------
        chi_center1 : Union[int, float], optional
            Center (in degrees chi) of the first chi wedge to integrate, defaults to 0 unless infer_ChiFromPol is True
        chi_center2 : Union[int, float], optional
            Center (in degrees chi) of the second chi wedge to integrate, defaults to (chi_center1 - 90) unless two scans are provided and infer_ChiFromPol is True
        chi_width : Union[int, float], optional
            width of slice in each direction in deg, by default 5
        reflectWedges : bool, optional
            if true, also integrate chi wedges centered 180 degrees from the chi_centers, by default False
        calc2d : bool, optional
            if true, calculate the AR using both polarization scans, by default False
        two_AR : bool, optional
            if true, return (AR1, AR2) separately, if false return a single value AR = (AR1 + AR2)/2, by default False
        calc2d_norm_energy : Union[int, float], optional
            if set, normalizes each polarization's AR at a given energy.  THIS EFFECTIVELY FORCES THE AR TO 0 AT THIS ENERGY., by default None
        infer_ChiFromPol : bool, optional
            if true, and if data contains a polarization dim, sets chi1 = pol1, chi2 = pol2, by default False
        printReport : bool, optional
            if true, runs the checkAR command, plotting wedges, and prints intermediate values, by default False

        Returns
        -------
        xr.DataArray
            DataArray or tuple of DataArrays containing AR values

        '''
        # Build Report for diagnosing AR quality
        reportAR = "Anisotropic Ratio Calculation:\n"

        # Determine wedge centers criteria

        # User wants to infer chi centers from beam polarization metadata
        if infer_ChiFromPol == True:

            # Make sure the user didnt provide chi1 and chi2 already
            if chi_center1 is not None:
                raise ValueError(
                    "infer_ChiFromPol must be False if you provide chi values manually"
                )

            # If we have polarization as a data dimensions
            if 'polarization' in self._obj.dims:
                # get polarization values
                pol_vals = self._obj.coords['polarization'].values.tolist()
                # if we have 1, save as as chi1,
                if len(pol_vals) == 1:
                    chi_center1 = pol_vals[0]
                # if two save as chi1 and chi2
                elif len(pol_vals) == 2:
                    chi_center1 = pol_vals[0]
                    chi_center2 = pol_vals[1]
                # Else raise error
                else:
                    raise NotImplementedError(
                        f'Can only infer if 1 or 2 polarization values are provided, found {len(pol_vals)} values'
                    )
            # Error if we don't have polarization as a data dimensions
            else:
                raise NotImplementedError(
                    'Cannot infer chi unless polarization values are provided as data dimensions'
                )

        # Hardcode default values/relationships between chi1 and chi2
        # If no info provided, default chi_1 to 0
        if chi_center1 is None:
            chi_center1 = 0
        # Make sure it fits in the valid range
        else:
            chi_center1 = self._reRange_chi(chi_center1)

        # If no info provided, chi_2 is chi_1 - 90
        if chi_center2 is None:
            chi_center2 = self._reRange_chi(chi_center1 - 90)

        # Add to report
        reportAR += f"\tchi_width = {chi_width}\n"
        reportAR += f"\tchi_center1 = {chi_center1:.3f}, chi_center2 = {chi_center2:.3f}\n"

        # If the user wants to use reflected wedges
        if reflectWedges == True:
            chi_center1r = self._reRange_chi(chi_center1 + 180)
            chi_center2r = self._reRange_chi(chi_center2 + 180)
            reportAR += f"\tchi_center1r = {chi_center1r:.3f}, chi_center2r = {chi_center2r:.3f}\n"

        # Calculate the absolute circular difference between chi_center1 and chi_center2
        abs_diff = min((chi_center1 - chi_center2) % 360, (chi_center2 - chi_center1) % 360)
        max_delta = 2  # deg
        # Check if the absolute circular difference is within the specified range
        if not ((90 - max_delta) <= abs_diff <= (90 + max_delta)):
            warnings.warn(
                "The difference between chi_center1 and chi_center2 is not within 90 ± 2 degrees.",
                stacklevel=2,
            )

        # Calculate for the simple case (single polarization)
        if not calc2d:

            # extract 2 wedges
            if reflectWedges == False:
                I_1 = self.slice_chi(chi=chi_center1, chi_width=chi_width, do_avg=False)
                I_1.name = 'I_chi1'
                I_2 = self.slice_chi(chi=chi_center2, chi_width=chi_width, do_avg=False)
                I_2.name = 'I_chi2'

                # check for overlap
                self._checkChiOverlap([I_1, I_2])

                # Compute Integrals and AR
                I_1 = I_1.mean('chi')
                I_2 = I_2.mean('chi')
                AR = (I_1 - I_2) / (I_1 + I_2)

                # add to report
                reportAR += (
                    f"\tI_1 Total Mean: {I_1.mean():.3f}\n"
                    f"\tI_2 Total Mean: {I_2.mean():.3f}\n"
                    f"\tAR Total Mean: {AR.mean():.3f}\n"
                )

            # else extract 4 wedges
            else:
                I_1 = self.slice_chi(chi=chi_center1, chi_width=chi_width, do_avg=False)
                I_1.name = 'I_chir1'
                I_2 = self.slice_chi(chi=chi_center2, chi_width=chi_width, do_avg=False)
                I_2.name = 'I_chir2'
                I_1r = self.slice_chi(chi=chi_center1r, chi_width=chi_width, do_avg=False)
                I_1r.name = 'I_chi1reflected'
                I_2r = self.slice_chi(chi=chi_center2r, chi_width=chi_width, do_avg=False)
                I_2r.name = 'I_chi2reflected'

                # check for overlap
                self._checkChiOverlap([I_1, I_2, I_1r, I_2r])

                # Compute Integrals and AR
                I_1 = I_1.mean('chi')
                I_2 = I_2.mean('chi')
                I_1r = I_1r.mean('chi')
                I_2r = I_2r.mean('chi')

                AR = (I_1 + I_1r) - (I_2 + I_2r) / ((I_1 + I_1r) + (I_2 + I_2r))

                # add to report
                reportAR += (
                    f"\tI_1 Total Mean: {I_1.mean():.3f}\n"
                    f"\tI_2 Total Mean: {I_2.mean():.3f}\n"
                    f"\tI_1r Total Mean: {I_1r.mean():.3f}\n"
                    f"\tI_2r Total Mean: {I_2r.mean():.3f}\n"
                    f"\tAR Total Mean: {AR.mean():.3f}\n"
                )

            # if report requested
            if printReport == True:
                self._checkAR()
                print(reportAR)

            return AR

        # elif calc2d:
        #     para_pol = self.select_pol(0)
        #     perp_pol = self.select_pol(90)

        #     para_para = para_pol.rsoxs.slice_chi(0, chi_width=chi_width)
        #     para_perp = para_pol.rsoxs.slice_chi(-90, chi_width=chi_width)

        #     perp_perp = perp_pol.rsoxs.slice_chi(-90, chi_width=chi_width)
        #     perp_para = perp_pol.rsoxs.slice_chi(0, chi_width=chi_width)

        #     AR_para = (para_para - para_perp) / (para_para + para_perp)
        #     AR_perp = (perp_perp - perp_para) / (perp_perp + perp_para)

        #     if calc2d_norm_energy is not None:
        #         AR_para = AR_para / AR_para.sel(energy=calc2d_norm_energy)
        #         AR_perp = AR_perp / AR_perp.sel(energy=calc2d_norm_energy)

        #     if (AR_para < AR_perp).all() or (AR_perp < AR_para).all():
        #         warnings.warn(
        #             'One polarization has a systematically higher/lower AR than the other.  Typically this indicates bad intensity values.',
        #             stacklevel=2,
        #         )

        #     if two_AR:
        #         return (AR_para, AR_perp)
        #     else:
        #         return (AR_para + AR_perp) / 2
        # else:
        #     raise NotImplementedError('Need either a single DataArray or a list of 2 dataarrays')

        print(reportAR)

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
