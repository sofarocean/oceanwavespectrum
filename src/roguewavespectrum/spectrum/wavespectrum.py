import typing

import numpy as np
from linearwavetheory import (
    inverse_intrinsic_dispersion_relation,
    intrinsic_group_speed,
    intrinsic_phase_speed,
)
from roguewavespectrum.time import to_datetime64
from roguewavespectrum.physical_constants import (
    PHYSICSOPTIONS,
    PhysicsOptions,
    _as_physicsoptions_lwt,
)
from abc import ABC, abstractmethod

from typing import TypeVar, List, Mapping
from xarray import Dataset, DataArray, concat, where, open_dataset
from xarray.core.coordinates import DatasetCoordinates
from warnings import warn
from .spline_interpolation import spline_peak_frequency
from numbers import Number
from .variable_names import (
    NAME_F,
    NAME_T,
    NAME_LAT,
    NAME_LON,
    NAME_DEPTH,
    SPECTRAL_VARS,
    SPECTRAL_DIMS,
    NAME_K,
    NAME_W,
    set_conventions,
)

from ._directions import (
    wave_mean_direction,
    wave_directional_spread,
    DirectionalUnit,
    DirectionalConvention,
)

_T = TypeVar("_T")
_number_or_dataarray = TypeVar("_number_or_dataarray", DataArray, Number)


class WaveSpectrum(ABC):
    """
    Base class for wave spectra. This class provides the basic functionality for wave spectra, which is extended by
    the FrequencySpectrum and FrequencyDirectionSpectrum classes for either 1D or 2D spectra. This class should never
    be instantiated directly, but should be used as a base class for the other classes.

    ** Implementation Considerations **

    - All state is stored in a dataset and is directly accessible through the dataset property. All other functions
      merely provide a convenient interface to-, and functions to operate on-, the dataset.

    - The only exception to this is the `physics_options` property, which is stored in a separate attribute. This
      stores physical options that are used to calculate derived variables from linear wave theory.

    - No caching of results. Certain operations (e.g. estimating the wavenumber) can be relatively costly. However,
      caching would introduce state and would make the class less predictable in case e.g. frequencies were changed.
      While workarounds exist - or we could state that inputs are assumed immutable - it is easier to simply not cache
      for now and put the onus on the user to properly reuse results if performance is of concern.
    """

    spectral_axis = NAME_F

    def __init__(
        self, dataset: Dataset, physics_options: PhysicsOptions = None, **kwargs
    ):
        # Check if at least frequency and variance density are present.

        self.physics_options = (
            physics_options if physics_options is not None else PHYSICSOPTIONS
        )

        # Make a shallow copy of the dataset to store as we will modify attributes / contents.
        self.dataset = dataset.copy(deep=False)

        # Ensure that the variables conform to the CF conventions. Note that we fail silently for
        # variables that we do not recognize as the user may have added custom variables.
        for var in self.dataset:
            try:
                self.dataset[var] = set_conventions(self.dataset[var], var)
            except KeyError:
                continue

        for coor in self.dataset.coords:
            try:
                self.dataset[coor] = set_conventions(self.dataset[coor], coor)
            except KeyError:
                continue

    def __copy__(self: _T) -> _T:
        return self.__class__(self.dataset.copy())

    @property
    @abstractmethod
    def _spectrum(self) -> DataArray:
        ...

    @_spectrum.setter
    @abstractmethod
    def _spectrum(self, value) -> DataArray:
        ...

    def copy(self: _T, deep=True) -> _T:
        if deep:
            return self.__deepcopy__({})
        else:
            return self.__copy__()

    def __deepcopy__(self: _T, memodict) -> _T:
        return self.__class__(self.copy(deep=True))

    @property
    def coords(self) -> DatasetCoordinates:
        return self.dataset.coords

    def sel(self: _T, *args, **kwargs) -> _T:
        dataset = Dataset()
        for var in self.dataset:
            dataset = dataset.assign({var: self.dataset[var].sel(*args, **kwargs)})
        return self.__class__(dataset=dataset)

    def isel(self: _T, *args, **kwargs) -> _T:
        dataset = Dataset()
        for var in self.dataset:
            dataset = dataset.assign({var: self.dataset[var].isel(*args, **kwargs)})
        return self.__class__(dataset=dataset)

    @property
    def ndims(self) -> int:
        """
        Calculate the number of dimensions of the spectrum.
        :return: number of dimensions
        """

        return len(self.dims)

    @property
    def frequency_binwidth(self) -> DataArray:
        """
        Calculate the frequency bin width.

        :return: Data array with the frequency stepsize
        """

        prepend = 2 * self.frequency[0] - self.frequency[1]
        append = 2 * self.frequency[-1] - self.frequency[-2]
        diff = np.diff(self.frequency, append=append, prepend=prepend)
        return set_conventions(
            DataArray(
                data=(diff[0:-1] * 0.5 + diff[1:] * 0.5),
                dims=NAME_F,
                coords={NAME_F: self.frequency},
            ),
            "frequency_bins",
            overwrite=True,
        )

    def fillna(self, value=0.0):
        """
        Fill NaN values in the dataset with a given value. Uses xarray's fillna method on a dataset object
        (see xarray documentation for more information).

        :param value: fill value
        :return: None
        """
        self.dataset = self.dataset.fillna(value)

    def is_invalid(self) -> DataArray:
        """
        Find invalid spectra. Invalid spectra are defined as spectra with NaN values in all spectral variables. Returns
        a data array with boolean values.
        :return: Data array with boolean values.
        """
        return self._spectrum.isnull().all(dim=self.dims_spectral)

    def is_valid(self) -> DataArray:
        """
        Find valid spectra. Valid spectra are defined as spectra that have Non-nan values (but may contain some NaN's).
        Inverse of is_invalid.

        :return: Data array with boolean values indicating valid spectra
        """
        return ~self.is_invalid()

    def drop_invalid(self: _T) -> _T:
        """
        Drop invalid spectra from the dataset. Invalid spectra are defined as spectra with NaN values in all spectral
        :return: Returns a new WaveSpectrum object with the invalid spectra removed.
        """
        return self.where(self.is_valid())

    def where(self: _T, condition: DataArray) -> _T:
        """
        Apply a boolean mask to the dataset.
        :param condition: Boolean mask indicating which spectra to keep
        :return:
        """
        dataset = Dataset()
        for var in self.dataset:
            data = self.dataset[var].where(
                condition.reindex_like(self.dataset[var]), drop=True
            )
            dataset = dataset.assign({var: data})

        return self.__class__(dataset)

    def flatten(self: "WaveSpectrum", flattened_coordinate="index") -> _T:
        """
        Serialize the non-spectral dimensions creating a single leading dimension without a coordinate.
        """

        # Get the current dimensions and shape
        dims = self.dims_space_time
        coords = self.coords_space_time
        shape = self.space_time_shape()
        if len(shape) == 0:
            length = 1
            shape = (1,)
        else:
            length = np.product(shape)

        # Calculate the flattened shape
        new_shape = (length,)
        new_spectral_shape = (length, *self.spectral_shape)
        new_dims = [flattened_coordinate] + self.dims_spectral

        linear_index = DataArray(data=np.arange(0, length), dims=flattened_coordinate)
        indices = np.unravel_index(linear_index.values, shape)

        dataset = {}
        for index, dim in zip(indices, dims):
            dataset[dim] = DataArray(
                data=coords[dim].values[index], dims=flattened_coordinate
            )

        for name in self.dataset:
            if name in SPECTRAL_VARS:
                x = DataArray(
                    data=self.dataset[name].values.reshape(new_spectral_shape),
                    dims=new_dims,
                    coords=self.coords_spectral,
                )
            else:
                x = DataArray(
                    data=self.dataset[name].values.reshape(new_shape),
                    dims=flattened_coordinate,
                )
            dataset[name] = x

        cls = type(self)
        return cls(Dataset(dataset))

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Tuple of array dimensions of the variance density spectrum

        See Also: np.ndarray.shape

        :return: Tuple of array dimensions.
        """
        return self._spectrum.shape

    @property
    def spectral_shape(self) -> tuple[int, ...]:
        """
        Tuple of array dimensions of the spectral dimensions (frequency and possibly direction) of the
        variance density spectrum

        See Also: np.ndarray.shape

        :return: Tuple of array dimensions.
        """
        number_of_spectral_dims = len(self.dims_spectral)
        return self.shape[-number_of_spectral_dims:]

    def space_time_shape(self) -> tuple[int, ...]:
        """
        Tuple of array dimensions of the spatial/temporal dimensions of the variance density spectrum.

        See Also: np.ndarray.shape

        :return: Tuple of array dimensions.
        """
        number_of_spectral_dims = len(self.dims_spectral)
        return self.shape[:-number_of_spectral_dims]

    def frequency_moment(
        self, power: int, fmin: float = 0.0, fmax: float = np.inf
    ) -> DataArray:
        """
        Calculate a "spectral moment" over the given range. A spectral moment here refers to the integral:

                    Integral-over-spectral-domain[ e * f**power ]

        :param power: power of the frequency
        :param fmin: minimum frequency, inclusive
        :param fmax: maximum frequency, inclusive
        :return: frequency moment
        """
        return set_conventions(
            _integrate(self.e * self.frequency**power, NAME_F, fmin, fmax),
            "MN",
            overwrite=True,
        )

    @property
    def number_of_frequencies(self) -> int:
        """
        :return: number of frequencies
        """
        return len(self.frequency)

    @property
    def dims_space_time(self) -> List[str]:
        """
        Return a list of the spatial and temporal dimensions of the variance density spectrum.

        :return: list[str] with names of spatial and temporal dimensions.
        """
        return [str(x) for x in self._spectrum.dims if x not in SPECTRAL_DIMS]

    @property
    def coords_space_time(self) -> Mapping[str, DataArray]:
        """
        Return a dictionary of the spatial and temporal coordinates of the variance density spectrum. Output has the
        form:

            { "DIMENSION_NAME1": DataArray, "DIMENSION_NAME2": DataArray, ... }

        :return: dictionary of spatial and temporal coordinates.
        """
        return {dim: self.dataset[dim] for dim in self.dims_space_time}

    @property
    def coords_spectral(self) -> Mapping[str, DataArray]:
        """
        Return a dictionary of the spectral coordinates of the variance density spectrum. Output has the form:

            { "DIMENSION_NAME1": DataArray, "DIMENSION_NAME2": DataArray }

        :return: dictionary of spectral coordinates.
        """
        return {dim: self.dataset[dim] for dim in self.dims_spectral}

    @property
    def dims_spectral(self) -> List[str]:
        """
        Return a list of the spectral dimensions of the variance density spectrum.

        :return: list[str] with names of spectral dimensions.
        """
        return [str(x) for x in self._spectrum.dims if x in SPECTRAL_DIMS]

    @property
    def dims(self) -> List[str]:
        """
        Return a list of the dimensions of the variance density spectrum.

        :return: list[str] with names of dimensions.
        """
        return [str(x) for x in self._spectrum.dims]

    @property
    def number_of_spectra(self) -> int:
        """
        Total number of spectra. This is the product of the size of the spatial and temporal leading
        dimensions.

        :return: Total number of spectra.
        """
        dims = self.dims_space_time
        if dims:
            shape = 1
            for d in dims:
                shape *= len(self.dataset[d])
            return shape
        else:
            return 1

    @property
    def angular_frequency(self) -> DataArray:
        """
        Angular frequency of the variance density spectra. The radial frequency is defined as:

        :return: Angular frequency - 2*pi*frequency
        """
        data_array = self.dataset[NAME_F] * 2 * np.pi
        return set_conventions(data_array, NAME_W, overwrite=True)

    @property
    def latitude(self) -> DataArray:
        """
        Latitudinal locations of the variance density spectra, The latitudes are given as decimal degree north.

        :return: latitudes decimal degree north.
        """
        if NAME_LAT in self.dataset:
            return self.dataset[NAME_LAT]
        else:
            raise ValueError("No latitude data associated with the spectrum")

    @property
    def longitude(self) -> DataArray:
        """
        Longitudinal locations of the variance density spectra, The longitudes are given as decimal degree east.

        :return: longitudes decimal degree east.
        """
        if NAME_LON in self.dataset:
            return self.dataset[NAME_LON]

        else:
            raise ValueError("No longitude data associated with the spectrum")

    @property
    def time(self) -> DataArray:
        """
        Return the time axis as a data array (time stored as datetime64)

        :return: Time
        """
        if NAME_T in self.dataset:
            return self.dataset[NAME_T]

        else:
            raise ValueError("No time axis found in dataset")

    @property
    @abstractmethod
    def variance_density(self) -> DataArray:
        ...

    @property
    def values(self) -> np.ndarray:
        """
        Get the numpy representation of the wave spectrum.

        :return: Numpy ndarray of the wave spectrum. [m^2/Hz or m^2/Hz/deg] depending on the type of spectrum.
        """
        return self._spectrum.values

    @property
    @abstractmethod
    def e(self) -> DataArray:
        ...

    @property
    @abstractmethod
    def a1(self) -> DataArray:
        """
        Normalized Fourier moment of the directional distribution function. See Kuik et al. (1988), eq A1.

        Kuik, A. J., Van Vledder, G. P., & Holthuijsen, L. H. (1988). A method for the routine analysis of
        pitch-and-roll buoy wave data. Journal of physical oceanography, 18(7), 1020-1034.

        :return: normalized Fourier moment cos(theta)
        """
        ...

    @property
    @abstractmethod
    def b1(self) -> DataArray:
        """
        Normalized Fourier moment of the directional distribution function. See Kuik et al. (1988), eq A1.

        Kuik, A. J., Van Vledder, G. P., & Holthuijsen, L. H. (1988). A method for the routine analysis of
        pitch-and-roll buoy wave data. Journal of physical oceanography, 18(7), 1020-1034.

        :return: normalized Fourier moment sin(theta)
        """
        ...

    @property
    @abstractmethod
    def a2(self) -> DataArray:
        """
        Normalized Fourier moment of the directional distribution function. See Kuik et al. (1988), eq A1.

        Kuik, A. J., Van Vledder, G. P., & Holthuijsen, L. H. (1988). A method for the routine analysis of
        pitch-and-roll buoy wave data. Journal of physical oceanography, 18(7), 1020-1034.

        :return: normalized Fourier moment cos(2*theta)
        """
        ...

    @property
    @abstractmethod
    def b2(self) -> DataArray:
        """
        Normalized Fourier moment of the directional distribution function. See Kuik et al. (1988), eq A1.

        Kuik, A. J., Van Vledder, G. P., & Holthuijsen, L. H. (1988). A method for the routine analysis of
        pitch-and-roll buoy wave data. Journal of physical oceanography, 18(7), 1020-1034.

        :return: normalized Fourier moment sin(2*theta)
        """
        ...

    @property
    def A1(self) -> DataArray:
        """
        Fourier moment of the directional distribution function (=a1(f)*e(f)).

        :return: Fourier moment cos(theta)
        """
        return set_conventions(self.a1 * self.e, "A1", overwrite=True)

    @property
    def B1(self) -> DataArray:
        """
        Fourier moment of the directional distribution function (=b1(f)*e(f)).

        :return: Fourier moment sin(theta)
        """
        return set_conventions(self.b1 * self.e, "B1", overwrite=True)

    @property
    def A2(self) -> DataArray:
        """
        Fourier moment of the directional distribution function (=a2(f)*e(f)).

        :return: Fourier moment cos(2*theta)
        """
        return set_conventions(self.a2 * self.e, "A2", overwrite=True)

    @property
    def B2(self) -> DataArray:
        """
        Fourier moment of the directional distribution function (=b2(f)*e(f)).

        :return: Fourier moment sin(2*theta)
        """
        return set_conventions(self.b2 * self.e, "B2", overwrite=True)

    @property
    def frequency(self) -> DataArray:
        """
        Frequency coordinates of the spectrum (Hz)

        :return: Frequencies (Hz)
        """
        return self.dataset[NAME_F]

    def m0(self, fmin: float = 0.0, fmax: float = np.inf) -> DataArray:
        """
        Zero order frequency moment of the spectrum. Also referred to as total variance. Dimension

        :param fmin: minimum frequency, inclusive
        :param fmax: maximum frequency, inclusive
        :return: variance (m^2)
        """
        return set_conventions(
            self.frequency_moment(0, fmin, fmax), "M0", overwrite=True
        )

    def m1(self, fmin: float = 0.0, fmax: float = np.inf) -> DataArray:
        """
        First order frequency moment. Primarily used in calculating a mean period measure (Tm01)

        :param fmin: minimum frequency, inclusive
        :param fmax: maximum frequency, inclusive
        :return: first order frequency moment.
        """
        return set_conventions(
            self.frequency_moment(1, fmin, fmax), "M1", overwrite=True
        )

    def wave_speed(self) -> DataArray:
        """
        Estimate the wave phase speed based on the dispersion relation. Return value is a DataArray with the same
        dimensions as the directionally integrated wave spectrum.

        :return: wave phase speed (m/s)
        """

        wavenumber = self.wavenumber
        wavespeed = intrinsic_phase_speed(
            wavenumber.values,
            self.depth.values,
            physics_options=_as_physicsoptions_lwt(self.physics_options),
        )
        return set_conventions(
            DataArray(wavespeed, dims=wavenumber.dims, coords=wavenumber.coords),
            "C",
            overwrite=True,
        )

    def wave_age(self, windspeed: typing.Union[Number, DataArray]) -> DataArray:
        """
        Simple wave age estimate based on the ratio of the peak wave speed to the given wind speed scale.

        :param windspeed: windspeed scale in m/s
        :return: wave age [-]
        """
        return set_conventions(
            self.peak_wave_speed() / windspeed, "wave_age", overwrite=True
        )

    def peak_wave_speed(self) -> DataArray:
        """
        Wave phase speed at the peak frequency. Return value is a DataArray with the same dimensions as the
        space/time dimensions of the wave spectrum.

        :return: peak wave speed (m/s)
        """
        return set_conventions(
            2 * np.pi * self.peak_frequency() / self.peak_wavenumber,
            "Cp",
            overwrite=True,
        )

    @property
    def wavenumber_spectral_density(self) -> DataArray:
        """
        Wavenumber Spectral density. Conversion through multiplication with the Jacobian of the
        transformation such that

            e(f) df = e(k) dk

        with e the density as function of frequency (f) or wavenumber (k), and df and dk the differentials of the
        respective variables. Note that with w = 2 * pi * f, the Jacobian is equal to

        df/dk =   dw/dk * df/dw = groupspeed / ( 2 * pi)

        :return: Wavenumber spectral density.
        """
        wavenumber = self.wavenumber
        data_array = set_conventions(
            self.e * self.groupspeed / (np.pi * 2),
            "wavenumber_variance_density",
            overwrite=True,
        )
        data_array = data_array.assign_coords({NAME_K: wavenumber})
        return data_array

    @property
    def saturation_spectrum(self) -> DataArray:
        """
        Saturation spectrum as introduced by Phillips.
        :return:
        """
        wavenumber = self.wavenumber
        data_array = set_conventions(
            self.wavenumber_spectral_density * wavenumber**3,
            "saturation_spectrum",
            overwrite=True,
        )
        data_array = data_array.assign_coords({NAME_K: wavenumber})
        return data_array

    @property
    def slope_spectrum(self) -> DataArray:
        wavenumber = self.wavenumber
        data_array = set_conventions(
            self.e * wavenumber**2, "slope_spectrum", overwrite=True
        )
        data_array = data_array.assign_coords({NAME_K: wavenumber})
        return data_array

    def mean_squared_slope(self, fmin: float = 0.0, fmax: float = np.inf) -> DataArray:
        """
        Mean sqyared slope of the surface. Defined as the integral of the slope spectrum over frequency, i.e.

            :math:` \\int_{f_min}^{f_max} E k^2 dk`

        :param fmin: minimum frequency, inclusive
        :param fmax: maximum frequency, inclusive
        :return:
        """
        return set_conventions(
            _integrate(self.slope_spectrum, NAME_F, fmin, fmax),
            "mean_squared_slope",
            overwrite=True,
        )

    def m2(self, fmin: float = 0.0, fmax: float = np.inf) -> DataArray:
        """
        Second order frequency moment. Primarily used in calculating the zero
        crossing period (Tm02)

        :param fmin: minimum frequency, inclusive
        :param fmax: maximum frequency, inclusive
        :return: Second order frequency moment.
        """
        return set_conventions(
            self.frequency_moment(2, fmin, fmax), "M2", overwrite=True
        )

    def hm0(self, fmin: float = 0.0, fmax: float = np.inf) -> DataArray:
        """
        Significant wave height estimated from the spectrum, i.e. waveheight
        h estimated from variance m0. Common notation in literature.

        :param fmin: minimum frequency, inclusive
        :param fmax: maximum frequency, inclusive
        :return: Significant wave height
        """
        return set_conventions(4 * np.sqrt(self.m0(fmin, fmax)), "Hm0", overwrite=True)

    def tm01(self, fmin: float = 0.0, fmax: float = np.inf) -> DataArray:
        """
        Mean period, estimated as the inverse of the center of mass of the
        spectral curve under the 1d spectrum.

        :param fmin: minimum frequency, inclusive
        :param fmax: maximum frequency, inclusive
        :return: Mean period
        """
        return set_conventions(
            self.m0(fmin, fmax) / self.m1(fmin, fmax), "Tm01", overwrite=True
        )

    def tm02(self, fmin: float = 0.0, fmax: float = np.inf) -> DataArray:
        """
        Zero crossing period based on Rice's spectral estimate.

        :param fmin: minimum frequency, inclusive
        :param fmax: maximum frequency, inclusive
        :return: Zero crossing period
        """
        return set_conventions(
            np.sqrt(self.m0(fmin, fmax) / self.m2(fmin, fmax)), "Tm02", overwrite=True
        )

    def peak_index(self, fmin: float = 0.0, fmax: float = np.inf) -> DataArray:
        """
        Index of the peak frequency of the 1d spectrum within the given range
        :param fmin: minimum frequency, inclusive
        :param fmax: maximum frequency, inclusive
        :return: peak indices
        """
        mask = (self.dataset[NAME_F].values >= fmin) & (
            self.dataset[NAME_F].values < fmax
        )
        return self.e.where(mask, 0).argmax(dim=NAME_F)

    def peak_frequency(
        self, fmin: float = 0.0, fmax: float = np.inf, use_spline=False, **kwargs
    ) -> DataArray:
        """
        Peak frequency of the spectrum, i.e. frequency at which the spectrum
        obtains its maximum.

        :param fmin: minimum frequency, inclusive
        :param fmax: maximum frequency, inclusive
        :param use_spline: Use a spline based interpolation and determine peak frequency from the spline. This
        allows for a continuous estimate of the peak frequency. WARNING: if True the fmin and fmax paramteres are
        IGNORED
        :return: peak frequency
        """
        if use_spline:
            if not fmin == 0.0 or np.isfinite(fmax):
                warn(
                    f"The fmin and fmax parameters are ignored if use_spline is set to True"
                )

            data = spline_peak_frequency(self.frequency.values, self.e.values, **kwargs)
            if len(self.dims_space_time) == 0:
                data = data[0]

            da = DataArray(
                data=data,
                coords=self.coords_space_time,
                dims=self.dims_space_time,
            )
        else:
            da = self.dataset[NAME_F][self.peak_index(fmin, fmax)]
        return set_conventions(da, "peak_frequency", overwrite=True)

    def peak_angular_frequency(
        self, fmin: float = 0.0, fmax: float = np.inf
    ) -> DataArray:
        """
        Peak angular frequency of the spectrum.

        :param fmin: minimum frequency, inclusive
        :param fmax: maximum frequency, inclusive
        :return: peak frequency
        """
        return set_conventions(
            self.peak_frequency(fmin, fmax) * np.pi * 2,
            "peak_angular_frequency",
            overwrite=True,
        )

    def peak_period(
        self, fmin: float = 0.0, fmax: float = np.inf, use_spline=False, **kwargs
    ) -> DataArray:
        """
        Peak period of the spectrum.

        :param fmin: minimum frequency, inclusive
        :param fmax: maximum frequency, inclusive
        :param use_spline: Use a spline based interpolation and determine peak period from the spline. This
        allows for a continuous estimate of the peak period. WARNING: if True the fmin and fmax paramteres are IGNORED
        :param kwargs: kwargs passed to spline_peak_frequency
        :return: peak period
        """
        peak_period = 1 / self.peak_frequency(
            fmin, fmax, use_spline=use_spline, **kwargs
        )
        peak_period = peak_period.drop_vars(names=NAME_F, errors="ignore")

        return set_conventions(peak_period, "Tp", overwrite=True)

    def peak_direction(
        self,
        fmin: float = 0.0,
        fmax: float = np.inf,
        directional_unit: DirectionalUnit = "degree",
        directional_convention: DirectionalConvention = "mathematical",
    ) -> DataArray:
        """
        Direction of the peak of the spectrum. The direction is expressed in degree anti-clockwise from east and
        represents the direction the waves travel *to*.

        Calculated in terms of the moments a1/b1 as defined by Kuik et al. (1988), their equation 33.

        Kuik, A. J., Van Vledder, G. P., & Holthuijsen, L. H. (1988). A method for the routine analysis of
        pitch-and-roll buoy wave data. Journal of physical oceanography, 18(7), 1020-1034.

        :param fmin: minimum frequency, inclusive
        :param fmax: maximum frequency, inclusive
        :param directional_unit: units of the output angle, one of 'degree' or 'radians'
        :param directional_convention: convention of the output angle, one of 'mathematical' or 'oceanographical'
            or 'meteorological'
        :return: peak direction
        """
        index = self.peak_index(fmin, fmax)
        return wave_mean_direction(
            self.a1.isel(**{NAME_F: index}),
            self.b1.isel(**{NAME_F: index}),
            directional_unit,
            directional_convention,
            "peak_direction",
        )

    def peak_directional_spread(
        self,
        fmin: float = 0.0,
        fmax: float = np.inf,
        directional_unit: DirectionalUnit = "degree",
    ) -> DataArray:
        """
        Directional spread of the peak of the spectrum. The spread is expressed in degree and calculated in terms of
        the definition by Kuik et al. (1988), their equation 34.

        Kuik, A. J., Van Vledder, G. P., & Holthuijsen, L. H. (1988). A method for the routine analysis of
        pitch-and-roll buoy wave data. Journal of physical oceanography, 18(7), 1020-1034.

        :param fmin: minimum frequency, inclusive
        :param fmax: maximum frequency, inclusive
        :param directional_unit: units of the output angle, one of 'degree' or 'radians'
        :return: directional spread
        """
        index = self.peak_index(fmin, fmax)
        a1 = self.a1.isel(**{NAME_F: index})
        b1 = self.b1.isel(**{NAME_F: index})
        return wave_directional_spread(
            a1, b1, unit=directional_unit, name="peak_directional_spread"
        )

    def mean_direction_per_frequency(
        self,
        directional_unit: DirectionalUnit = "degree",
        directional_convention: DirectionalConvention = "mathematical",
    ) -> DataArray:
        """
        Mean direction of each frequency bin. The direction is expressed in degree anti-clockwise from east and
        represents the direction the waves travel *to*.

        Calculated in terms of the moments a1/b1 as defined by Kuik et al. (1988), their equation 33.

        Kuik, A. J., Van Vledder, G. P., & Holthuijsen, L. H. (1988). A method for the routine analysis of
        pitch-and-roll buoy wave data. Journal of physical oceanography, 18(7), 1020-1034.

        :param directional_unit: units of the output angle, one of 'degree' or 'radians'
        :param directional_convention: convention of the output angle, one of 'mathematical' or 'oceanographical'
            or 'meteorological'

        :return: direction
        """
        return wave_mean_direction(
            self.a1, self.b1, directional_unit, directional_convention, "direction"
        )

    def mean_spread_per_frequency(
        self, directional_unit: DirectionalUnit = "degree"
    ) -> DataArray:
        """
        Directional spread of each frequency bin. The spread is expressed in degree and calculated in terms of
        the definition by Kuik et al. (1988), their equation 34.

        Kuik, A. J., Van Vledder, G. P., & Holthuijsen, L. H. (1988). A method for the routine analysis of
        pitch-and-roll buoy wave data. Journal of physical oceanography, 18(7), 1020-1034.

        :param directional_unit: units of the output angle, one of 'degree' or 'radians'

        :return: directional spread
        """
        return wave_directional_spread(
            self.a1, self.b1, directional_unit, "directional_spread"
        )

    def _spectral_weighted(
        self, _property: DataArray, fmin: float = 0.0, fmax: float = np.inf
    ):
        """
        Calculate the spectral weighted mean property.
        :param _property: spectral property to calculate weighted mean for
        :param fmin: minimum frequency, inclusive
        :param fmax: maximum frequency, inclusive
        :return: spectral weighted mean property
        """
        return _integrate(_property * self.e, NAME_F, fmin, fmax) / self.m0(fmin, fmax)

    def mean_direction(
        self,
        fmin: float = 0.0,
        fmax: float = np.inf,
        directional_unit: DirectionalUnit = "degree",
        directional_convention: DirectionalConvention = "mathematical",
    ) -> DataArray:
        """
        Mean direction the spectrum. Per default the direction is expressed in degree anti-clockwise from east and
        represents the direction the waves travel *to*.

        Calculated according to Kuik et al. (1988), their equation 33, using weighted directional moments (see,
        mean_a1, mean_b1).


        Kuik, A. J., Van Vledder, G. P., & Holthuijsen, L. H. (1988). A method for the routine analysis of
        pitch-and-roll buoy wave data. Journal of physical oceanography, 18(7), 1020-1034.

        :param fmin: minimum frequency, inclusive
        :param fmax: maximum frequency, inclusive
        :param directional_unit: units of the output angle, one of 'degree' or 'radians'
        :param directional_convention: convention of the output angle, one of 'mathematical' or 'oceanographical'
            or 'meteorological'
        :return: mean direction
        """

        return wave_mean_direction(
            self.mean_a1(fmin, fmax),
            self.mean_b1(fmin, fmax),
            directional_unit,
            directional_convention,
            "mean_direction",
        )

    def mean_directional_spread(
        self,
        fmin: float = 0.0,
        fmax: float = np.inf,
        directional_unit: DirectionalUnit = "degree",
    ) -> DataArray:
        """
        Mean directional spread of the spectrum. Calculated according to Kuik et al. (1988), their equation 33,
        using weighted directional moments (see, mean_a1, mean_b1).

        Kuik, A. J., Van Vledder, G. P., & Holthuijsen, L. H. (1988). A method for the routine analysis of
        pitch-and-roll buoy wave data. Journal of physical oceanography, 18(7), 1020-1034.

        :param fmin: minimum frequency, inclusive
        :param fmax: maximum frequency, inclusive
        :param directional_unit: units of the output angle, one of 'degree' or 'radians'
        :return: peak direction
        """
        return wave_directional_spread(
            self.mean_a1(fmin, fmax),
            self.mean_b1(fmin, fmax),
            directional_unit,
            "mean_directional_spread",
        )

    def mean_a1(self, fmin: float = 0.0, fmax: float = np.inf) -> DataArray:
        """
        Return the spectral weighted mean moment a1m defined as

        .. math:: a_{1m} = \\frac{\\int_{f_{min}}^{f_{max}} a_1(f) E(f) df}{M_0}

        where :math:`M_0` is the zeroth moment of the spectrum.
        :param fmin: minimum frequency, inclusive
        :param fmax: maximum frequency, inclusive
        :return:
        """
        return set_conventions(
            self._spectral_weighted(self.a1, fmin, fmax), "a1", overwrite=True
        )

    def mean_b1(self, fmin: float = 0.0, fmax: float = np.inf) -> DataArray:
        """
        Return the spectral weighted mean moment b1m defined as

        .. math:: b_{1m} = \\frac{\\int_{f_{min}}^{f_{max}} b_1(f) E(f) df}{M_0}

        :param fmin: minimum frequency, inclusive
        :param fmax: maximum frequency, inclusive
        :return:
        """
        return set_conventions(
            self._spectral_weighted(self.b1, fmin, fmax), "b1", overwrite=True
        )

    def mean_a2(self, fmin: float = 0.0, fmax: float = np.inf) -> DataArray:
        """
        Return the spectral weighted mean moment b1m defined as

        .. math:: a_{2m} = \\frac{\\int_{f_{min}}^{f_{max}} a_2(f) E(f) df}{M_0}

        :param fmin: minimum frequency, inclusive
        :param fmax: maximum frequency, inclusive
        :return:
        """
        return set_conventions(
            self._spectral_weighted(self.a2, fmin, fmax), "a2", overwrite=True
        )

    def mean_b2(self, fmin: float = 0.0, fmax: float = np.inf) -> DataArray:
        """
        Return the spectral weighted mean moment b1m defined as

        .. math:: b_{2m} = \\frac{\\int_{f_{min}}^{f_{max}} b_2(f) E(f) df}{M_0}

        :param fmin: minimum frequency, inclusive
        :param fmax: maximum frequency, inclusive
        :return:
        """
        return set_conventions(
            self._spectral_weighted(self.b2, fmin, fmax), "b2", overwrite=True
        )

    @property
    def depth(self) -> DataArray:
        """
        Water depth at the location of the spectra. If the depth is not available, the depth is set to infinity (deep
        water). Depth is used in calculation of spectral parameters such as the group speed.

        :return: Depth [m]
        """
        if NAME_DEPTH in self.dataset:
            depth = self.dataset[NAME_DEPTH]
            data_array = where(depth.isnull(), np.inf, depth)

        else:
            if len(self.dims_space_time) == 0:
                data_array = DataArray(np.inf)

            else:
                data_array = DataArray(
                    data=np.full(self.space_time_shape(), np.inf),
                    dims=self.dims_space_time,
                    coords=self.coords_space_time,
                    name="Depth [m]",
                )

        return set_conventions(data_array, NAME_DEPTH, overwrite=True)

    @property
    def groupspeed(self) -> DataArray:
        """
        Group speed of the waves for each frequency. The group speed is the speed at which the energy of the wave
        packet travels. Calculated based on the implementation in the `linearwavetheory' package.

        :return: group speed [m/s]
        """
        depth = self.depth.expand_dims(dim=NAME_F, axis=-1).values
        k = self.wavenumber.values

        # Construct the output coordinates and dimension of the data array
        return_dimensions = (*self.dims_space_time, NAME_F)
        coords = {}
        for dim in return_dimensions:
            coords[dim] = self.dataset[dim].values

        return set_conventions(
            DataArray(
                data=intrinsic_group_speed(
                    k,
                    depth,
                    physics_options=_as_physicsoptions_lwt(self.physics_options),
                ),
                dims=return_dimensions,
                coords=coords,
            ),
            "group_speed",
            overwrite=True,
        )

    @property
    def wavenumber(self) -> DataArray:
        """
        Determine the wavenumbers for the frequencies in the spectrum. Note that since the dispersion relation depends
        on depth the returned wavenumber array has the dimensions associated with the depth array by the frequency
        dimension.

        :return: wavenumbers
        """

        # For numba (used in the dispersion relation) we need raw numpy arrays of the correct dimension
        depth = self.depth.expand_dims(dim=NAME_F, axis=-1).values
        radian_frequency = self.angular_frequency.expand_dims(
            dim=self.depth.dims
        ).values

        # Construct the output coordinates and dimension of the data array
        return_dimensions = (*self.dims_space_time, NAME_F)
        coords = {}
        for dim in return_dimensions:
            coords[dim] = self.dataset[dim].values

        return set_conventions(
            DataArray(
                data=inverse_intrinsic_dispersion_relation(
                    radian_frequency,
                    depth,
                    physics_options=_as_physicsoptions_lwt(self.physics_options),
                ),
                dims=return_dimensions,
                coords=coords,
            ),
            "wavenumber",
            overwrite=True,
        )

    @property
    def wavelength(self) -> DataArray:
        return set_conventions(
            2 * np.pi / self.wavenumber, "wavelength", overwrite=True
        )

    @property
    def peak_wavenumber(self) -> DataArray:
        index = self.peak_index()
        # Construct the output coordinates and dimension of the data array
        coords = {}
        for dim in self.dims_space_time:
            coords[dim] = self.dataset[dim].values

        return set_conventions(
            DataArray(
                data=inverse_intrinsic_dispersion_relation(
                    self.angular_frequency[index].values,
                    self.depth.values,
                    physics_options=_as_physicsoptions_lwt(self.physics_options),
                ),
                dims=self.dims_space_time,
                coords=coords,
            ),
            "peak_wavenumber",
            overwrite=True,
        )

    @property
    def significant_waveheight(self) -> DataArray:
        """
        Significant wave height (Hm0) estimated based on the variance density spectrum. Estimated according to

                    hm0 = 4 * sqrt(m0)

        see Holthuijsen (2010), equation 4.2.24

        Holthuijsen, L. H. (2010). Waves in oceanic and coastal waters. Cambridge university press.

        :return: Data array containing the significant wave height for each spectrum.
        """
        return self.hm0()

    @property
    def mean_period(self) -> DataArray:
        """
        Mean period (Tm01) estimated based on the variance density spectrum. Estimated according to

            tm01 = m0 / m1

        see Holthuijsen (2010), equation 4.2.6

        :return: Data array containing the mean period for each spectrum.
        """
        return self.tm01()

    @property
    def zero_crossing_period(self) -> DataArray:
        """
        Zero crossing period (Tm02) estimated based on the variance density spectrum. Estimated according to

            tm02 = sqrt(m0 / m2)

        see Holthuijsen (2010), equation 4.2.5

        :return: Data array containing the zero crossing period for each spectrum.
        """
        return self.tm02()

    def cumulative_density_function(self) -> DataArray:
        """

        :return:
        """
        frequency_step = self.frequency_binwidth
        integration_frequencies = np.concatenate(
            ([0], np.cumsum(frequency_step.values))
        )
        integration_frequencies = (
            integration_frequencies
            - frequency_step.values[0] / 2
            + self.frequency.values[0]
        )
        values = (self.variance_density * frequency_step).values

        frequency_axis = self.dims.index(NAME_F)

        cumsum = np.cumsum(values, axis=frequency_axis)

        shape = list(cumsum.shape)
        shape[frequency_axis] = 1

        cumsum = np.concatenate((np.zeros(shape), cumsum), axis=frequency_axis)

        coords = {str(coor): self.coords[coor].values for coor in self.coords}
        coords[NAME_F] = integration_frequencies
        return set_conventions(
            DataArray(data=cumsum, dims=self.dims, coords=coords), "cdf", overwrite=True
        )

    def interpolate(
        self: _T, coordinates, extrapolation_value=0.0, method="linear", **kwargs
    ) -> _T:
        """
        Interpolate the spectrum to the given coordinates. The coordinates should be a dictionary with the dimension
        name as key and the coordinate as value. Uses the xarray interp method. Extrapolation is done by filling the
        NaN values with the given extrapolation value (0.0 by default).

        :param coordinates: dictionary with coordinates for each dimension
        :param extrapolation_value: value to use for extrapolation
        :param method: interpolation method to use (see xarray interp method)
        :return: interpolated spectrum object
        """
        if "time" in coordinates:
            coordinates["time"] = to_datetime64(coordinates["time"])

        spectrum = self.__class__(
            self.dataset.interp(
                coords=coordinates,
                method=method,
                kwargs={"fill_value": extrapolation_value},
            ),
            **kwargs,
        )
        spectrum.fillna(extrapolation_value)
        return spectrum

    def bandpass(
        self: _T,
        fmin: float = 0.0,
        fmax: float = np.inf,
        interpolate_to_limits: bool = True,
    ) -> _T:
        """
        Bandpass the spectrum to the given limits such that the frequency coordinates of the spectrum are given as

            fmin <= f <= fmax

        :param fmin: minimum frequency, inclusive (default 0)
        :param fmax: maximum frequency, inclusive (default np.inf)
        :param interpolate_to_limits: Exactly interpolate to the limits. If False, we merely return the frequency values
            that are within the limits. If True, we interpolate the spectrum to the limits.
        :return: Return a new spectrum object with the bandpassed spectrum.
        """
        dataset = Dataset()

        for name in self.dataset:
            if name in SPECTRAL_VARS:
                data = _bandpass(
                    self.dataset[name],
                    fmin,
                    fmax,
                    dim=NAME_F,
                    interpolate=interpolate_to_limits,
                )
                dataset = dataset.assign({name: data})
            else:
                dataset = dataset.assign({name: self.dataset[name]})
        cls = type(self)
        return cls(dataset)

    def interpolate_frequency(
        self: _T,
        new_frequencies,
        extrapolation_value: float = 0.0,
        method: str = "linear",
        **kwargs,
    ) -> _T:
        """
        Interpolate the spectrum to the given frequencies. Convenience method for interpolate. See interpolate for
        more information.

        :param new_frequencies: new frequencies to interpolate to
        :param extrapolation_value: extrapolation value
        :param method: interpolation method (see xarray interp method)
        :return: Interpolated spectrum
        """
        _object = self.interpolate(
            {NAME_F: new_frequencies},
            extrapolation_value=extrapolation_value,
            method=method,
            **kwargs,
        )
        _object.fillna(extrapolation_value)
        return _object

    def to_netcdf(self, path: str, **kwargs):
        """
        Save spectrum to a netcdf file. See xarray to_netcdf method for more information on use.

        :param path: path or path-like location to save the spectrum to.
        :return: None
        """
        return self.dataset.to_netcdf(path, **kwargs)

    @classmethod
    def from_netcdf(cls: _T, path: str, **kwargs) -> _T:
        """
        Load spectrum from a netcdf file. See xarray open_dataset method for more information on use.

        :param path: path or path-like location to load the spectrum from.
        :return: Spectrum object
        """
        return cls(open_dataset(path, **kwargs))

    @classmethod
    def from_dataset(cls: _T, dataset: Dataset, mapping=None) -> _T:
        """
        Create a spectrum object from a xarray dataset.

        :param dataset: xarray dataset
        :param mapping: dictionary mapping the xarray dataset names to the spectrum names
        :return: Spectrum object
        """
        if mapping is not None:
            dataset = dataset.copy(deep=False).rename(mapping)

        return cls(dataset)

    def multiply(
        self, array: np.ndarray, dimensions: List[str] = None, inplace=False
    ) -> _T:
        """
        Multiply the variance density with the given numpy array. Broadcasting is performed automatically if dimensions
        are provided. If no dimensions are provided the array needs to have the exact same shape as the variance
        density array.

        :param array: Array to multiply with variance density
        :param dimensions: Dimensions of the array
        :param inplace: Perform the operation inplace
        :return: self
        """
        output = self if inplace else self.copy()

        coords = {}
        shape = array.shape
        if dimensions is None:
            if shape != self.shape:
                raise ValueError(
                    "If no dimensions are provided the array must have the exact same shape as the"
                    "variance density array."
                )

            output._spectrum = self._spectrum * array
            return output

        if len(shape) != len(dimensions):
            raise ValueError(
                "The dimensions of the input array must match the number of dimension labels"
            )

        for length, dimension in zip(shape, dimensions):
            if dimension not in self.dims:
                raise ValueError(
                    f"Dimension {dimension} not a valid dimension of the spectral object."
                )
            coords[dimension] = self.dataset[dimension].values

            if len(self.dataset[dimension].values) != length:
                raise ValueError(
                    f"Array length along the dimension {dimension} does not match the length of the"
                    f"coordinate of the same name in the spectral object."
                )

        data = DataArray(data=array, coords=coords, dims=dimensions)
        output._spectrum = self._spectrum * data
        return output


########################################################################################################################
# Helper functions
########################################################################################################################


def _integrate(
    data: DataArray, dim: str, lower_limit=-np.inf, upper_limit=np.inf
) -> DataArray:
    """
    Integrate the data along the given dimension using the xarray integration method. We first bandpass the data
    to the given limits and then integrate to ensure we integrate over the requested limits
    """

    # Integrate dataset over frequencies. Make sure to fill any NaN entries with 0 before the integration.
    return (
        _bandpass(data, lower_limit, upper_limit, dim, interpolate=True)
        .fillna(0)
        .integrate(coord=dim)
    )


def _bandpass(
    data: DataArray, _min: Number, _max: Number, dim: str, interpolate: bool = False
) -> DataArray:
    coord = data[dim]

    if _min >= _max:
        raise ValueError(f"max must be larger than min, {_min} >= {_max}")

    # Check if there are actual bounds on the domain, if not, we can just return the data
    _domain_is_bounded = np.isfinite(_min) or np.isfinite(_max)
    if not _domain_is_bounded:
        return data

    if not interpolate:
        # Get frequency mask for [ fmin, fmax )
        _range = (coord.values >= _min) & (coord.values <= _max)

        return data.isel({dim: _range})

    _range = (coord.values > _min) & (coord.values < _max)
    coords = coord[_range]

    # We may have to interpolate the spectrum to get the correct frequency for the min and max cut-off
    # frequencies. To do the interpolation we
    # 1) get the frequencies such that  fmin <= f < fmax, and then
    # 2) fmin and fmax to the frequency array
    # 3) interpolate the spectrum to the new frequency array, if either fmin or fmax was added.
    # 4) calculate the integral.
    #
    if np.isfinite(_max):
        # If fmax is finite we may need to add it:
        coords = concat(
            [coords, DataArray([_max], dims=dim, coords={dim: [_max]})], dim=dim
        )

    if np.isfinite(_min):
        # If fmin is larger than 0 we may have to add it if the first frequency is larger than fmin.
        coords = concat(
            [DataArray([_min], dims=dim, coords={dim: [_min]}), coords], dim=dim
        )

    return data.interp({dim: coords})
