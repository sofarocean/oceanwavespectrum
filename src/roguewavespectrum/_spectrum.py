import typing

import numpy as np
from linearwavetheory import (
    inverse_intrinsic_dispersion_relation,
    intrinsic_group_speed,
    intrinsic_phase_speed,
)
from roguewavespectrum._time import to_datetime64
from roguewavespectrum._physical_constants import (
    PHYSICSOPTIONS,
    PhysicsOptions,
    _as_physicsoptions_lwt,
)
from ._directions import (
    DirectionalConvention,
    DirectionalUnit,
    wave_mean_direction,
    wave_directional_spread,
    convert_unit,
    convert_angle_convention,
    get_angle_convention_and_unit,
)

from roguewavespectrum._estimators.estimate import (
    estimate_directional_spectrum_from_moments,
    Estimators,
)

from typing import TypeVar, List, Mapping, Tuple, Union
from xarray import Dataset, DataArray, concat, where, open_dataset
from xarray.core.coordinates import DatasetCoordinates
from warnings import warn
from ._spline_interpolation import spline_peak_frequency
from numbers import Number
from ._variable_names import (
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
    NAME_D,
    NAME_E,
    NAME_e,
    NAME_a1,
    NAME_b1,
    NAME_a2,
    NAME_b2,
)

from ._spline_interpolation import cumulative_frequency_interpolation_1d_variable

_T = TypeVar("_T")
_number_or_dataarray = TypeVar("_number_or_dataarray", DataArray, Number)


class Spectrum:
    """
    Class for wave spectra. This class provides the basic functionality for 1D and 2D wave spectra. The class is
    based on xarray, and provides a number of convenience methods for working with wave spectra. To construct a
    spectrum object given numpy arrays, use the create_spectrum1d or create_spectrum2d functions, or initialize the
    class directly with a dataset object.
    """

    def __init__(
        self, dataset: Dataset, physics_options: PhysicsOptions = None, **kwargs
    ):
        """
        Initialize a spectrum object from a xarray dataset. The contents on the dataset determines if the spectrum is
        a 1D or 2D spectrum.

        1D spectrum required variables:
        - variance_density[...,frequency]: variance density spectrum
        - a1[...,frequency]: Fourier moment a1
        - b1[...,frequency]: Fourier moment b1
        - a2[...,frequency]: Fourier moment a2
        - b2[...,frequency]: Fourier moment b2
        - frequency[frequency]: frequency coordinates

        2D spectrum required variables:
        - directional_variance_density[...,frequency,direction]: variance density spectrum
        - frequency[frequency]: frequency coordinates
        - direction[direction]: direction coordinates. Direction must be in degrees, and span [0,360) degrees.

        The presence of either a `directional_variance_density` or `variance_density` variable determines if the
        spectrum is a 1D or 2D spectrum. If both are present, an error is raised as the spectrum is ambiguous (i.e.
        variance_density may or may not represent a directionally integrated version of directional_variance_density).

        Leading dimensions are optional, preserved in calculations and output, may be named or anonymous, and may be
        associated with coordinates. For example, if `variance_density` has dimensions `time` and `frequency`,
        then calculating significant waveheight will return waveheigth with dimensions `time`.

        We may optionally add, `time`, `latitude` or `longitude` as variables or as coordinates.

        Example 1D dataset specification where we have spectra as function of frequency and time, and we have latitude,
        longitude locations:
        ```python
        dataset = Dataset(
            data_vars={
                "variance_density": (["time","frequency"], variance_density),})
                "a1": (["time","frequency"], a1),
                "b1": (["time","frequency"], b1),
                "a2": (["time","frequency"], a2),
                "b2": (["time","frequency"], b2),
                "latitude": (["time"], latitude),
                "longitude": (["time"], longitude),
            },
            coords={"frequency": frequency,"time": time},
        ```

        Example 2D dataset specification where we have spectra as function of frequency:
        ```python
        dataset = Dataset(
            data_vars={
                "directional_variance_density": (["frequency","direction"], directional_variance_density),})
            },
            coords={"frequency": frequency,"direction": direction},
        ```

        Example 2D dataset specification where we have spectra as function of frequency, time, latitude and longitude (
        e.g. as from a model):
        ```python
        dataset = Dataset(
            data_vars={
                "directional_variance_density": (["time,latitude,longitude,frequency","direction"],
                    directional_variance_density),})
            },
            coords={"frequency": frequency,"direction": direction,"time": time, "latitude": latitude,
                "longitude": longitude},
        ```
        Note that in this case "latitude and longitude" are coordinates on the object (as opposed to variables in
        the 1D example above).

        :param dataset: Dataset containing the spectral data.
        :param physics_options: Options for calculating derived variables from linear wave theory.

        """

        self._physics_options = (
            physics_options if physics_options is not None else PHYSICSOPTIONS
        )  #: The physics options that are used to calculate derived variables from linear wave theory.

        # Make a shallow copy of the dataset to store as we will modify attributes / contents.
        self.dataset = dataset.copy(
            deep=False
        )  #: The dataset that contains the spectral data.

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

        if self.is_2d and self.is_1d:
            raise ValueError(
                "Only one of variance_density or directional_variance_density may be specified in the"
                "dataset."
            )

        # 1D or 2D spectra?
        if self.is_2d:
            required_variables = [NAME_F, NAME_D, NAME_E]
        elif self.is_1d:
            required_variables = [NAME_F, NAME_e]
        else:
            raise ValueError(
                "The dataset does not contain a variance_density or directional_variance_density variable."
            )

        for name in required_variables:
            if name not in dataset and name not in dataset.coords:
                raise ValueError(
                    f"Required variable/coordinate {name} is"
                    f" not specified in the dataset"
                )

    # Dunder methods
    # -----------------
    def __copy__(self: "Spectrum") -> "Spectrum":
        return self.__class__(self.dataset.copy(), self._physics_options)

    def __deepcopy__(self: "Spectrum", memodict) -> "Spectrum":
        return self.__class__(self.dataset.copy(deep=True), self._physics_options)

    # Private methods
    # -----------------
    def _directionally_integrate(self, data_array: DataArray) -> DataArray:
        return (data_array * self.direction_binwidth).sum(NAME_D, skipna=True)

    @property
    def _spectrum(self) -> DataArray:
        if self.is_2d:
            return self.dataset[NAME_E]
        else:
            return self.dataset[NAME_e]

    @_spectrum.setter
    def _spectrum(self, value):
        if self.is_2d:
            self.dataset[NAME_E] = value
        else:
            self.dataset[NAME_e] = value

    @property
    def _has_identifiers(self) -> bool:
        return "ids" in self.dataset

    # Operations
    # ------------

    def as_frequency_direction_spectrum(
        self: "Spectrum",
        number_of_directions: int = 36,
        method: Estimators = "mem2",
        solution_method="scipy",
    ) -> "Spectrum":
        """
        Construct a 2D directional energy spectrum based on the directional moments and a specified spectral
        reconstruction method.

        :param number_of_directions: number of directions to use in the reconstruction. The directions are given by
        np.linspace(0,360,number_of_directions,endpoint=False)

        :param method: Choose a method in ['mem','mem2']
            mem: maximum entrophy (in the Boltzmann sense) method
            Lygre, A., & Krogstad, H. E. (1986). Explicit expression and
            fast but tends to create narrow spectra anderroneous secondary peaks.

            mem2: use entrophy (in the Shannon sense) to maximize. Likely
            best method see- Benoit, M. (1993).

        :param solution_method: Only relevant for MeM2. Choose a solution method in ['scipy','newton'] This determines
            if we solve the nonlinear set of equations using scipy or a custom numba implemented newton-root finder.
            The newton solver is faster but may be less robust. For details we refer to comments in the code itself.

        :return: Spectrum object

        REFERENCES:
        Benoit, M. (1993). Practical comparative performance survey of methods
            used for estimating directional wave spectra from heave-pitch-roll data.
            In Coastal Engineering 1992 (pp. 62-75).

        Lygre, A., & Krogstad, H. E. (1986). Maximum entropy estimation of the
            directional distribution in ocean wave spectra.
            Journal of Physical Oceanography, 16(12), 2052-2060.

        """
        if self.is_2d:
            return self.copy()

        direction = np.linspace(0, 360, number_of_directions, endpoint=False)

        output_array = estimate_directional_spectrum_from_moments(
            self.variance_density.values,
            self.a1.values,
            self.b1.values,
            self.a2.values,
            self.b2.values,
            direction,
            method=method,
            solution_method=solution_method,
        )

        dims = list(self.dims_space_time) + [NAME_F, NAME_D]
        coords = {x: self.dataset[x].values for x in self.dims}
        coords[NAME_D] = direction

        data = {NAME_E: (dims, output_array)}
        for x in self.dataset:
            if x in SPECTRAL_VARS:
                continue
            data[x] = (self.dims_space_time, self.dataset[x].values)

        return Spectrum(
            Dataset(data_vars=data, coords=coords),
            physics_options=self._physics_options,
        )

    def as_frequency_spectrum(self) -> "Spectrum":
        """
        Return a 1D spectrum by integrating over the directions and return a Spectrum object. This will also include
        the a1,b1,a2,b2 moments as estimated from the 2D spectrum. If the object already is a 1D spectrum, the object
        returns a shallow copy of itself.
        :return: Spectrum object
        """
        if not self.is_2d:
            return self.copy()

        dataset = {
            "a1": self.a1,
            "b1": self.b1,
            "a2": self.a2,
            "b2": self.b2,
            "variance_density": self.variance_density,
        }
        for name in self.dataset:
            if name not in SPECTRAL_VARS:
                dataset[name] = self.dataset[name]
        return Spectrum(Dataset(dataset), physics_options=self._physics_options)

    def bandpass(
        self: "Spectrum",
        fmin: float = 0.0,
        fmax: float = np.inf,
        interpolate_to_limits: bool = True,
    ) -> "Spectrum":
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

    def copy(self: "Spectrum", deep=True) -> "Spectrum":
        """
        Return a copy of the object.
        :param deep: If True, perform a deep copy. Otherwise, perform a shallow copy.
        :return: copy of the object
        """
        if deep:
            return self.__deepcopy__({})
        else:
            return self.__copy__()

    def drop_invalid(self: "Spectrum") -> "Spectrum":
        """
        Drop invalid spectra from the dataset. Invalid spectra are defined as spectra with NaN values in all spectral
        :return: Returns a new WaveSpectrum object with the invalid spectra removed.
        """
        return self.where(self.is_valid())

    def fillna(self, value=0.0):
        """
        Fill NaN values in the dataset with a given value. Uses xarray's fillna method on a dataset object
        (see xarray documentation for more information).

        :param value: fill value
        :return: None
        """
        self.dataset = self.dataset.fillna(value)

    def flatten(self: "Spectrum", flattened_coordinate="index") -> "Spectrum":
        """
        Serialize the non-spectral dimensions creating a single leading dimension without a coordinate.
        """

        # Get the current dimensions and shape
        dims = self.dims_space_time
        coords = self.coords_space_time
        shape = self.space_time_shape
        if len(shape) == 0:
            length = 1
            shape = (1,)
        else:
            length = np.product(shape)

        # Calculate the flattened shape
        new_shape = (length,)
        new_spectral_shape = (length, *self.spectral_shape)
        new_dims = [flattened_coordinate] + list(self.dims_spectral)

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

    def interpolate(
        self: "Spectrum",
        coordinates,
        extrapolation_value=0.0,
        method="linear",
        **kwargs,
    ) -> "Spectrum":
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

        if self.is_2d:
            spectrum = Spectrum(
                self.dataset.interp(
                    coords=coordinates,
                    method=method,
                    kwargs={"fill_value": extrapolation_value},
                ),
                physics_options=self._physics_options,
                **kwargs,
            )
        else:
            # For 1D interpolation we have to take care to also interpolate the Fourier moments.
            _dataset = Dataset()
            _moments = [NAME_a1, NAME_b1, NAME_a2, NAME_b2]

            for name in self.dataset:
                _name = str(name)
                if _name in _moments:
                    _dataset = _dataset.assign(
                        {_name: getattr(self, _name) * self.variance_density}
                    )
                else:
                    _dataset = _dataset.assign({_name: self.dataset[_name]})

            interpolated_data = _dataset.interp(coords=coordinates, method=method)
            for name in _moments:
                interpolated_data[name] = (
                    interpolated_data[name] / interpolated_data[NAME_e]
                )

            spectrum = Spectrum(interpolated_data)

        spectrum.fillna(extrapolation_value)
        return spectrum

    def interpolate_frequency(
        self: "Spectrum",
        new_frequencies,
        extrapolation_value: float = 0.0,
        method: str = "linear",
        **kwargs,
    ) -> "Spectrum":
        """
        Interpolate the spectrum to the given frequencies. Convenience method for interpolate. See interpolate for
        more information.

        :param new_frequencies: new frequencies to interpolate to
        :param extrapolation_value: extrapolation value
        :param method: interpolation method (see xarray interp method)
        :return: Interpolated spectrum
        """
        if method == "spline":
            if self.is_2d:
                raise ValueError(
                    "Spline interpolation is only available for 1D spectra"
                )

            if isinstance(new_frequencies, DataArray):
                new_frequencies = new_frequencies.values

            self.fillna(0.0)
            frequency_axis = self.dims.index(NAME_F)
            interpolated_data = cumulative_frequency_interpolation_1d_variable(
                new_frequencies, self.dataset, frequency_axis=frequency_axis, **kwargs
            )
            spectrum = Spectrum(
                interpolated_data, physics_options=self._physics_options
            )
            spectrum.fillna(extrapolation_value)

        else:
            spectrum = self.interpolate(
                {NAME_F: new_frequencies},
                extrapolation_value=extrapolation_value,
                method=method,
                **kwargs,
            )
        return spectrum

    def isel(self: "Spectrum", *args, **kwargs) -> "Spectrum":
        dataset = Dataset()
        for var in self.dataset:
            dataset = dataset.assign({var: self.dataset[var].isel(*args, **kwargs)})
        return self.__class__(dataset=dataset)

    @property
    def is_2d(self) -> bool:
        if NAME_E in self.dataset:
            return True
        else:
            return False

    @property
    def is_1d(self) -> bool:
        if NAME_e in self.dataset:
            return True
        else:
            return False

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

    def sel(self: "Spectrum", *args, **kwargs) -> "Spectrum":
        dataset = Dataset()
        for var in self.dataset:
            dataset = dataset.assign({var: self.dataset[var].sel(*args, **kwargs)})
        return self.__class__(dataset=dataset)

    def where(self: "Spectrum", condition: DataArray) -> "Spectrum":
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

    # Coordinates, dimension and shape
    # ---------------------------------
    @property
    def angular_frequency(self) -> DataArray:
        """
        Angular frequency of the variance density spectra. The radial frequency is defined as:

        :return: Angular frequency - 2*pi*frequency
        """
        data_array = self.dataset[NAME_F] * 2 * np.pi
        return set_conventions(data_array, NAME_W, overwrite=True)

    @property
    def coords(self) -> DatasetCoordinates:
        """
        Return the coordinates of the spectrum dataarray.
        :return: dataset coordinates
        """
        return self.dataset.coords

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
    def dims(self) -> Tuple[str]:
        """
        Return a tuple of the dimensions of the variance density spectrum.

        :return: list[str] with names of dimensions.
        """
        return tuple(str(x) for x in self._spectrum.dims)

    @property
    def dims_space_time(self) -> Tuple[str]:
        """
        Return a tuple of the spatial and temporal dimensions of the variance density spectrum.

        :return: list[str] with names of spatial and temporal dimensions.
        """
        return tuple(str(x) for x in self._spectrum.dims if x not in SPECTRAL_DIMS)

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
                    data=np.full(self.space_time_shape, np.inf),
                    dims=self.dims_space_time,
                    coords=self.coords_space_time,
                    name="Depth [m]",
                )

        return set_conventions(data_array, NAME_DEPTH, overwrite=True)

    @depth.setter
    def depth(self, value: Union[Number, DataArray]):
        if np.isscalar(value):
            value = DataArray(
                data=np.full(self.space_time_shape, value),
                dims=self.dims_space_time,
                coords=self.coords_space_time,
                name="Depth [m]",
            )
        elif not isinstance(value, DataArray):
            raise ValueError("Depth must be a scalar or DataArray")
        self.dataset[NAME_DEPTH] = value

    @property
    def dims_spectral(self) -> Tuple[str]:
        """
        Return a tuple of the spectral dimensions of the variance density spectrum.

        :return: list[str] with names of spectral dimensions.
        """
        return tuple(str(x) for x in self._spectrum.dims if x in SPECTRAL_DIMS)

    def direction(
        self,
        directional_unit: DirectionalUnit = "degree",
        directional_convention: DirectionalConvention = "mathematical",
    ) -> DataArray:
        """
        Return the direction of the variance density spectrum in the specified convention and unit.
        :param directional_unit: Directional unit to return the direction in. Options are "degree" or "rad"
        :param directional_convention: Directional convention to return the direction in. Options are "oceanographical",
            "mathematical" or "meteorological"
        :return: directions as a dataset
        """
        if not self.is_2d:
            raise ValueError("Direction is only applicable for 2D spectra")
        from_convention, from_unit = get_angle_convention_and_unit(
            self.dataset[NAME_D],
            default_convention="mathematical",
            default_unit="degree",
        )
        angle = convert_unit(self.dataset[NAME_D], directional_unit, from_unit)
        angle = convert_angle_convention(
            angle, directional_convention, from_convention, units=directional_unit
        )
        angle = set_conventions(angle, [NAME_D, directional_convention], overwrite=True)
        angle.attrs["units"] = directional_unit
        return angle

    @property
    def direction_binwidth(self) -> DataArray:
        """
        Calculate the step size between the direction bins. Because the direction bins are circular, we use a modular
        difference estimate.
        :return:
        """
        if not self.is_2d:
            raise ValueError("Direction is only applicable for 2D spectra")

        direction = self.direction()
        unit = direction.attrs["units"]
        if unit == "rad":
            wrap = 2 * np.pi
        elif unit == "degree":
            wrap = 360
        else:
            raise ValueError(f"Unknown directional unit {unit}")

        forward = (
            np.diff(self.direction().values, append=self.direction()[0]) + wrap / 2
        ) % wrap - wrap / 2
        backward = (
            np.diff(self.direction().values, prepend=self.direction()[-1]) + wrap / 2
        ) % wrap - wrap / 2

        data_array = set_conventions(
            DataArray(
                data=(forward + backward) / 2,
                coords={NAME_D: self.direction().values},
                dims=[NAME_D],
            ),
            "direction_bins",
            overwrite=True,
        )
        data_array.attrs["units"] = unit
        return data_array

    @property
    def frequency(self) -> DataArray:
        """
        Frequency coordinates of the spectrum (Hz)

        :return: Frequencies (Hz)
        """
        return self.dataset[NAME_F]

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
    def ndims(self) -> int:
        """
        Calculate the number of dimensions of the spectrum.
        :return: number of dimensions
        """

        return len(self.dims)

    @property
    def number_of_directions(self) -> int:
        "Number of directions. Only applicable for 2D spectra"
        if not self.is_2d:
            raise ValueError("Number of directions is only applicable for 2D spectra")
        return len(self.direction())

    @property
    def number_of_frequencies(self) -> int:
        """
        Number of frequencies in the spectrum.
        :return: number of frequencies
        """
        return len(self.frequency)

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
    def radian_direction_mathematical(self) -> DataArray:
        if not self.is_2d:
            raise ValueError("Direction is only applicable for 2D spectra")

        return self.direction(
            directional_unit="rad", directional_convention="mathematical"
        )

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Tuple of array dimensions of the variance density spectrum

        See Also: np.ndarray.shape

        :return: Tuple of array dimensions.
        """
        return self._spectrum.shape

    @property
    def space_time_shape(self) -> tuple[int, ...]:
        """
        Tuple of array dimensions of the spatial/temporal dimensions of the variance density spectrum.

        See Also: np.ndarray.shape

        :return: Tuple of array dimensions.
        """
        number_of_spectral_dims = len(self.dims_spectral)
        return self.shape[:-number_of_spectral_dims]

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

    # Spectral properties
    # --------------------------------------------
    @property
    def a1(self) -> DataArray:
        """
        Normalized Fourier moment of the directional distribution function. See Kuik et al. (1988), eq A1.

        Kuik, A. J., Van Vledder, G. P., & Holthuijsen, L. H. (1988). A method for the routine analysis of
        pitch-and-roll buoy wave data. Journal of physical oceanography, 18(7), 1020-1034.

        :return: normalized Fourier moment cos(theta)
        """
        if self.is_2d:
            data_array = (
                self._directionally_integrate(
                    self._spectrum * np.cos(self.radian_direction_mathematical)
                )
                / self.variance_density
            )
            data_array = set_conventions(data_array, NAME_a1, overwrite=True)
        else:
            data_array = self.dataset[NAME_a1]

        return data_array

    @property
    def A1(self) -> DataArray:
        """
        Fourier moment of the directional distribution function (=a1(f)*e(f)).

        :return: Fourier moment cos(theta)
        """
        return set_conventions(self.a1 * self.variance_density, "A1", overwrite=True)

    @property
    def a2(self) -> DataArray:
        """
        Normalized Fourier moment of the directional distribution function. See Kuik et al. (1988), eq A1.

        Kuik, A. J., Van Vledder, G. P., & Holthuijsen, L. H. (1988). A method for the routine analysis of
        pitch-and-roll buoy wave data. Journal of physical oceanography, 18(7), 1020-1034.

        :return: normalized Fourier moment cos(2*theta)
        """
        if self.is_2d:
            data_array = (
                self._directionally_integrate(
                    self._spectrum * np.cos(2 * self.radian_direction_mathematical)
                )
                / self.variance_density
            )
            data_array = set_conventions(data_array, NAME_a2, overwrite=True)
        else:
            data_array = self.dataset[NAME_a2]
        return data_array

    @property
    def A2(self) -> DataArray:
        """
        Fourier moment of the directional distribution function (=a2(f)*e(f)).

        :return: Fourier moment cos(2*theta)
        """
        return set_conventions(self.a2 * self.variance_density, "A2", overwrite=True)

    @property
    def b1(self) -> DataArray:
        """
        Normalized Fourier moment of the directional distribution function. See Kuik et al. (1988), eq A1.

        Kuik, A. J., Van Vledder, G. P., & Holthuijsen, L. H. (1988). A method for the routine analysis of
        pitch-and-roll buoy wave data. Journal of physical oceanography, 18(7), 1020-1034.

        :return: normalized Fourier moment sin(theta)
        """
        if self.is_2d:
            data_array = (
                self._directionally_integrate(
                    self._spectrum * np.sin(self.radian_direction_mathematical)
                )
                / self.variance_density
            )
            data_array = set_conventions(data_array, NAME_b1, overwrite=True)
        else:
            data_array = self.dataset[NAME_b1]
        return data_array

    @property
    def B1(self) -> DataArray:
        """
        Fourier moment of the directional distribution function (=b1(f)*e(f)).

        :return: Fourier moment sin(theta)
        """
        return set_conventions(self.b1 * self.variance_density, "B1", overwrite=True)

    @property
    def b2(self) -> DataArray:
        """
        Normalized Fourier moment of the directional distribution function. See Kuik et al. (1988), eq A1.

        Kuik, A. J., Van Vledder, G. P., & Holthuijsen, L. H. (1988). A method for the routine analysis of
        pitch-and-roll buoy wave data. Journal of physical oceanography, 18(7), 1020-1034.

        :return: normalized Fourier moment sin(2*theta)
        """
        if self.is_2d:
            data_array = (
                self._directionally_integrate(
                    self._spectrum * np.sin(2 * self.radian_direction_mathematical)
                )
                / self.variance_density
            )
            data_array = set_conventions(data_array, NAME_b2, overwrite=True)
        else:
            data_array = self.dataset[NAME_b2]
        return data_array

    @property
    def B2(self) -> DataArray:
        """
        Fourier moment of the directional distribution function (=b2(f)*e(f)).

        :return: Fourier moment sin(2*theta)
        """
        return set_conventions(self.b2 * self.variance_density, "B2", overwrite=True)

    @property
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

    @property
    def directional_variance_density(self) -> DataArray:
        if not self.is_2d:
            raise ValueError(
                "Directional variance density is only applicable for 2D spectra"
            )
        return self.dataset[NAME_E]

    @property
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

    @property
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
            self.variance_density * wavenumber**2, "slope_spectrum", overwrite=True
        )
        data_array = data_array.assign_coords({NAME_K: wavenumber})
        return data_array

    @property
    def values(self) -> np.ndarray:
        """
        Get the numpy representation of the wave spectrum.

        :return: Numpy ndarray of the wave spectrum. [m^2/Hz or m^2/Hz/deg] depending on the type of spectrum.
        """
        return self._spectrum.values

    @property
    def variance_density(self) -> DataArray:
        """
        1D variance density spectrum as data array.

        :return: Variance density [m^2/Hz].
        """
        if self.is_2d:
            return set_conventions(
                self._directionally_integrate(self._spectrum), NAME_e, overwrite=True
            )
        else:
            return self.dataset[NAME_e]

    @property
    def wavenumber_directional_spectral_density(self) -> DataArray:
        """
        Wavenumber Spectral density. Conversion through multiplication with the Jacobian of the
        transformation such that

            E(f) df = E(k) dk

        with e the density as function of frequency (f) or wavenumber (k), and df and dk the differentials of the
        respective variables. Note that with w = 2 * pi * f, the Jacobian is equal to

        df/dk =   dw/dk * df/dw = groupspeed / ( 2 * pi)

        Note: requores that the spectra as directional information.

        :return: Wavenumber spectral density.
        """
        if not self.is_2d:
            raise ValueError(
                "Wavenumber directional spectral density can only be calculated for 2D spectra"
            )

        wavenumber = self.wavenumber
        data_array = set_conventions(
            self.variance_density * self.groupspeed / (np.pi * 2),
            "wavenumber_directional_variance_density",
            overwrite=True,
        )
        data_array = data_array.assign_coords({NAME_K: wavenumber})
        return data_array

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
            self.variance_density * self.groupspeed / (np.pi * 2),
            "wavenumber_variance_density",
            overwrite=True,
        )
        data_array = data_array.assign_coords({NAME_K: wavenumber})
        return data_array

    # Wave properties
    # --------------------------------------------

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
                    physics_options=_as_physicsoptions_lwt(self._physics_options),
                ),
                dims=return_dimensions,
                coords=coords,
            ),
            "group_speed",
            overwrite=True,
        )

    @property
    def wavelength(self) -> DataArray:
        return set_conventions(
            2 * np.pi / self.wavenumber, "wavelength", overwrite=True
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
                    physics_options=_as_physicsoptions_lwt(self._physics_options),
                ),
                dims=return_dimensions,
                coords=coords,
            ),
            "wavenumber",
            overwrite=True,
        )

    def wavespeed(self) -> DataArray:
        """
        Estimate the wave phase speed based on the dispersion relation. Return value is a DataArray with the same
        dimensions as the directionally integrated wave spectrum.

        :return: wave phase speed (m/s)
        """

        wavenumber = self.wavenumber
        wavespeed = intrinsic_phase_speed(
            wavenumber.values,
            self.depth.values,
            physics_options=_as_physicsoptions_lwt(self._physics_options),
        )
        return set_conventions(
            DataArray(wavespeed, dims=wavenumber.dims, coords=wavenumber.coords),
            "C",
            overwrite=True,
        )

    # Bulk properties
    # --------------------------------------------

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
            _integrate(
                self.variance_density * self.frequency**power, NAME_F, fmin, fmax
            ),
            "MN",
            overwrite=True,
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

    def hrms(self, fmin: float = 0.0, fmax: float = np.inf) -> DataArray:
        """
        Root mean square wave height estimated from the spectrum.

        :param fmin: minimum frequency, inclusive
        :param fmax: maximum frequency, inclusive
        :return: RMS wave height
        """
        return set_conventions(
            self.hm0(fmin, fmax) / np.sqrt(2), "Hrms", overwrite=True
        )

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

    def mean_a1(self, fmin: float = 0.0, fmax: float = np.inf) -> DataArray:
        """
        Return the spectral weighted mean moment a1m defined as

        .. math:: a_{1m} = \\frac{\\int_{f_{min}}^{f_{max}} a_1(f) E(f) df}{M_0}

        where :math:`M_0` is the zeroth moment of the spectrum.
        :param fmin: minimum frequency, inclusive
        :param fmax: maximum frequency, inclusive
        :return:
        """
        a1m = _integrate(self.a1 * self.variance_density, NAME_F, fmin, fmax) / self.m0(
            fmin, fmax
        )
        return set_conventions(a1m, "a1", overwrite=True)

    def mean_a2(self, fmin: float = 0.0, fmax: float = np.inf) -> DataArray:
        """
        Return the spectral weighted mean moment b1m defined as

        .. math:: a_{2m} = \\frac{\\int_{f_{min}}^{f_{max}} a_2(f) E(f) df}{M_0}

        :param fmin: minimum frequency, inclusive
        :param fmax: maximum frequency, inclusive
        :return:
        """
        a2m = _integrate(self.a2 * self.variance_density, NAME_F, fmin, fmax) / self.m0(
            fmin, fmax
        )
        return set_conventions(a2m, "a2", overwrite=True)

    def mean_b1(self, fmin: float = 0.0, fmax: float = np.inf) -> DataArray:
        """
        Return the spectral weighted mean moment b1m defined as

        .. math:: b_{1m} = \\frac{\\int_{f_{min}}^{f_{max}} b_1(f) E(f) df}{M_0}

        :param fmin: minimum frequency, inclusive
        :param fmax: maximum frequency, inclusive
        :return:
        """
        b1m = _integrate(self.b1 * self.variance_density, NAME_F, fmin, fmax) / self.m0(
            fmin, fmax
        )
        return set_conventions(b1m, "b1", overwrite=True)

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

    def mean_b2(self, fmin: float = 0.0, fmax: float = np.inf) -> DataArray:
        """
        Return the spectral weighted mean moment b1m defined as

        .. math:: b_{2m} = \\frac{\\int_{f_{min}}^{f_{max}} b_2(f) E(f) df}{M_0}

        :param fmin: minimum frequency, inclusive
        :param fmax: maximum frequency, inclusive
        :return:
        """
        b2m = _integrate(self.b2 * self.variance_density, NAME_F, fmin, fmax) / self.m0(
            fmin, fmax
        )
        return set_conventions(b2m, "b2", overwrite=True)

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
                    "The fmin and fmax parameters are ignored if use_spline is set to True"
                )

            data = spline_peak_frequency(
                self.frequency.values, self.variance_density.values, **kwargs
            )
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
        return self.variance_density.where(mask, 0).argmax(dim=NAME_F)

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

    def peak_wavespeed(
        self, fmin=0.0, fmax=np.inf, use_spline=False, **kwargs
    ) -> DataArray:
        """
        Wave phase speed at the peak frequency. Return value is a DataArray with the same dimensions as the
        space/time dimensions of the wave spectrum.

        :return: peak wave speed (m/s)
        """
        return set_conventions(
            2
            * np.pi
            * self.peak_frequency(fmin, fmax, use_spline, **kwargs)
            / self.peak_wavenumber(fmin, fmax, use_spline, **kwargs),
            "Cp",
            overwrite=True,
        )

    def peak_wavenumber(
        self, fmin=0.0, fmax=np.inf, use_spline=False, **kwargs
    ) -> DataArray:
        """
        Peak wavenumber of the spectrum.
        :param fmin:
        :param fmax:
        :param use_spline:
        :param kwargs:
        :return:
        """

        peak_frequency = self.peak_frequency(fmin, fmax, use_spline, **kwargs)
        # Construct the output coordinates and dimension of the data array
        coords = {}
        for dim in self.dims_space_time:
            coords[dim] = self.dataset[dim].values

        return set_conventions(
            DataArray(
                data=inverse_intrinsic_dispersion_relation(
                    2 * np.pi * peak_frequency.values,
                    self.depth.values,
                    physics_options=_as_physicsoptions_lwt(self._physics_options),
                ),
                dims=self.dims_space_time,
                coords=coords,
            ),
            "peak_wavenumber",
            overwrite=True,
        )

    def mean_period(self, fmin: float = 0.0, fmax: float = np.inf) -> DataArray:
        """
        Mean period, estimated as the inverse of the center of mass of the
        spectral curve under the 1d spectrum.

        :param fmin: minimum frequency, inclusive
        :param fmax: maximum frequency, inclusive
        :return: Mean period
        """
        return set_conventions(
            self.m0(fmin, fmax) / self.m1(fmin, fmax), "mean_period", overwrite=True
        )

    def energy_period(self, fmin: float = 0.0, fmax: float = np.inf) -> DataArray:
        """
        Wave energy period defined as the ratio of the zeroth and second frequency moment.

        :param fmin: minimum frequency, inclusive
        :param fmax: maximum frequency, inclusive
        :return: Mean period
        """
        return set_conventions(
            self.m1(fmin, fmax) / self.m0(fmin, fmax), "energy_period", overwrite=True
        )

    def zero_crossing_period(
        self, fmin: float = 0.0, fmax: float = np.inf
    ) -> DataArray:
        """
        Zero crossing period based on Rice's spectral estimate.

        :param fmin: minimum frequency, inclusive
        :param fmax: maximum frequency, inclusive
        :return: Zero crossing period
        """
        return set_conventions(
            np.sqrt(self.m0(fmin, fmax) / self.m2(fmin, fmax)),
            "zero_crossing_period",
            overwrite=True,
        )

    def waveage(self, windspeed: typing.Union[Number, DataArray]) -> DataArray:
        """
        Simple wave age estimate based on the ratio of the peak wave speed to the given wind speed scale.

        :param windspeed: windspeed scale in m/s
        :return: wave age [-]
        """
        return set_conventions(
            self.peak_wavespeed() / windspeed, "wave_age", overwrite=True
        )

    # ===================================================================================================================
    # IO
    # ===================================================================================================================
    def to_netcdf(self, path: str, **kwargs):
        """
        Save spectrum to a netcdf file. See xarray to_netcdf method for more information on use.

        :param path: path or path-like location to save the spectrum to.
        :return: None
        """
        return self.dataset.to_netcdf(path, **kwargs)

    # ===================================================================================================================
    # Class methods
    # ===================================================================================================================
    @classmethod
    def from_netcdf(cls: "Spectrum", path: str, **kwargs) -> "Spectrum":
        """
        Load spectrum from a netcdf file. See xarray open_dataset method for more information on use.

        :param path: path or path-like location to load the spectrum from.
        :return: Spectrum object
        """
        return cls(open_dataset(path, **kwargs))

    @classmethod
    def from_dataset(
        cls: "Spectrum", dataset: Dataset, mapping=None, deep=False
    ) -> "Spectrum":
        """
        Create a spectrum object from a xarray dataset. The dataset must either contain for

        Spectrum1D:
        - the variables: "variance_density", "a1", "b1", "a2", "b2"
        - coordinates: "frequency",

        Spectrum2D:
        - the variable: "directional_variance_density"
        - coordinates: "frequency", "direction"

        or a mapping must be provided that maps the dataset names to expected variable/coordinat names. Coordinate units
        are  assumed to be assumed to be degrees [0,360) and Hz (frequency>=0). Units of the spectrum are assumed to be
        m^2/Hz or m^2/Hz/degree. The default interpretation of the direction is mathematical (90 degrees is North), and
        direction indicates the direction the waves are going towards, and it is assumed directional moments are
        consistent with this interpretation.

        Note that by default we create a shallow copy of the dataset.

        :param dataset: xarray dataset
        :param mapping: dictionary mapping the xarray dataset names to the spectrum names
        :param deep: If True, create a deep copy of the input dataset.

        :return: Spectrum object
        """
        dataset = dataset.copy(deep=deep)
        if mapping is not None:
            dataset = dataset.rename(mapping)

        return cls(dataset)


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


class BuoySpectrum(Spectrum):
    """
    Class for wave spectra derived from (multiple) buoys. The class is a subclass of Spectrum and inherits all
    functionality from that class. The class adds functionality to select spectra by buoy identifier.
    """

    def __init__(self, dataset: Dataset, physics_options: PhysicsOptions = None):
        """
        Create a spectrum object from a xarray dataset. In addition to the requirements of the Spectrum class, the
        dataset must contain a leading index coordinate, and contain the variable "ids" that for each index indicates
        the buoy (or observation platforms) unique identifier. The unique identifiers are used to select spectra by
        buoy identifier.

        Example 1D dataset:
        ```python
        dataset = Dataset(
            data_vars={
                "variance_density": (["index","frequency"], variance_density),})
                "a1": (["index","frequency"], a1),
                "b1": (["index","frequency"], b1),
                "a2": (["index","frequency"], a2),
                "b2": (["index","frequency"], b2),
                "time": (["index"], latitude),
                "latitude": (["index"], latitude),
                "longitude": (["index"], longitude),
                "ids": (["index"], ids),
            },
            coords={"frequency": frequency,"index": index},
        ```

        :param dataset: Dataset containing the spectral data.
        :param physics_options: Options for calculating derived variables from linear wave theory.
        """
        super().__init__(dataset, physics_options=physics_options)

        if "ids" not in self.dataset:
            raise ValueError(
                "Buoy spectrum requires a dataset with the 'ids' variable that indicates the buoy id"
            )

        if "time" not in self.dataset:
            raise ValueError(
                "Buoy spectrum requires a dataset with the 'time' variable that indicates the time of each"
                "spectral observation"
            )

        if "unique_ids" not in dataset.coords:
            ids = np.unique(dataset["ids"])
            self.dataset = self.dataset.assign_coords(unique_ids=ids)

    def sel_by_id(self: "BuoySpectrum", item: str) -> "BuoySpectrum":
        """
        Select spectra by identifier.
        :param item: identifier of the spectra to select.
        :return: Spectrum object with the selected spectra.
        """
        spectrum = self.where(self.dataset["ids"] == item)
        dataset = spectrum.dataset.swap_dims({"index": "time"})
        return self.__class__(dataset)

    def __getitem__(self, item) -> "BuoySpectrum":
        return self.sel_by_id(item)

    def __contains__(self, item):
        return item in self.keys()

    def __iter__(self):
        return (id for id in self.keys())

    def keys(self) -> List[str]:
        return list(self.dataset["unique_ids"].values)

    def items(self):
        return ((id, self.sel_by_id(id)) for id in self.keys())

    @classmethod
    def from_trajectory_dataset(cls, dataset: Dataset, mapping=None) -> "BuoySpectrum":
        """
        Create a spectrum object from a xarray trajectory dataset.

        :param dataset: xarray dataset
        :param mapping: dictionary mapping the xarray dataset names to the spectrum names.
        :return: Spectrum object
        """
        if mapping is not None:
            dataset = dataset.copy(deep=False).rename(mapping)

        if "trajectory" in dataset:
            trajectory = dataset["trajectory"].values
            rowsizes = dataset["rowsize"].values
            labels = []
            for ordinal_index, rowsize in enumerate(rowsizes):
                labels += rowsize * [trajectory[ordinal_index]]

            dataset = dataset.assign({"ids": ("index", labels)})
            dataset = dataset.drop_vars(["trajectory", "rowsize"])

        else:
            raise ValueError(
                "creating GroupedFrequencySpectrum from a trajectory dataset requires trajectory "
                "information"
            )

        return cls(dataset)
