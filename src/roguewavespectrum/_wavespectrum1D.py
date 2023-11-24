import numpy
import numpy as np

from ._spectrum import Spectrum
from roguewavespectrum._estimators.estimate import (
    estimate_directional_spectrum_from_moments,
    Estimators,
)
from roguewavespectrum._time import to_datetime64
from roguewavespectrum._physical_constants import PhysicsOptions
from typing import TypeVar, Literal
from xarray import Dataset, DataArray, zeros_like, ones_like, concat
from ._spline_interpolation import cumulative_frequency_interpolation_1d_variable
from ._variable_names import (
    NAME_F,
    NAME_D,
    NAME_E,
    NAME_a1,
    NAME_b1,
    NAME_a2,
    NAME_b2,
    NAME_e,
    SPECTRAL_VARS,
)
from ._wavespectrum2D import Spectrum2D
from ._extrapolate import numba_fill_zeros_or_nan_in_tail

_T = TypeVar("_T")


class Spectrum1D(Spectrum):
    def __init__(
        self, dataset: Dataset, physics_options: PhysicsOptions = None, **kwargs
    ):
        super(Spectrum1D, self).__init__(
            dataset, physics_options=physics_options, **kwargs
        )
        for name in [NAME_F, NAME_e]:
            if name not in dataset and name not in dataset.coords:
                raise ValueError(
                    f"Required variable/coordinate {name} is"
                    f" not specified in the dataset"
                )

    @property
    def _spectrum(self) -> DataArray:
        return self.dataset[NAME_e]

    @_spectrum.setter
    def _spectrum(self, value):
        self.dataset[NAME_e] = value

    def interpolate_frequency(
        self: "Spectrum1D",
        new_frequencies,
        extrapolation_value=0.0,
        method: Literal["nearest", "linear", "spline"] = "linear",
        **kwargs,
    ) -> "Spectrum1D":

        if isinstance(new_frequencies, DataArray):
            new_frequencies = new_frequencies.values

        if method == "spline":
            self.fillna(0.0)
            frequency_axis = self.dims.index(NAME_F)
            interpolated_data = cumulative_frequency_interpolation_1d_variable(
                new_frequencies, self.dataset, frequency_axis=frequency_axis, **kwargs
            )
            object = Spectrum1D(interpolated_data)
            object.fillna(extrapolation_value)
            return object
        else:
            return self.interpolate(
                {NAME_F: new_frequencies},
                extrapolation_value=extrapolation_value,
                method=method,
            )

    def interpolate(
        self: _T, coordinates, extrapolation_value=0.0, method="linear", **kwargs
    ) -> _T:
        """
        Interpolate the spectrum to the given coordinates. The coordinates should be a dictionary with the dimension
        name as key and the coordinate as value. Uses the xarray interp method. Extrapolation is done by filling the
        NaN values with the given extrapolation value (0.0 by default).

        For physical reasons it is better to interpolate the scaled moments - as opposed to the normalized moments.
        For the dataset we interpolate we set a1: to A1 etc. Afterwards we scale the output back to the normalized
        state.

        :param coordinates: dictionary with coordinates for each dimension
        :param extrapolation_value: value to use for extrapolation
        :param method: interpolation method to use (see xarray interp method)
        :return: interpolated spectrum object
        """

        _dataset = Dataset()
        _moments = [NAME_a1, NAME_b1, NAME_a2, NAME_b2]

        if "time" in coordinates:
            coordinates["time"] = to_datetime64(coordinates["time"])

        for name in self.dataset:
            _name = str(name)
            if _name in _moments:
                _dataset = _dataset.assign({_name: getattr(self, _name) * self.e})
            else:
                _dataset = _dataset.assign({_name: self.dataset[_name]})

        interpolated_data = _dataset.interp(coords=coordinates, method=method)
        for name in _moments:
            interpolated_data[name] = (
                interpolated_data[name] / interpolated_data[NAME_e]
            )

        object = Spectrum1D(interpolated_data)
        object.fillna(extrapolation_value)
        return object

    @property
    def a1(self) -> DataArray:
        """
        Normalized Fourier moment of the directional distribution function. See Kuik et al. (1988), eq A1.

        Kuik, A. J., Van Vledder, G. P., & Holthuijsen, L. H. (1988). A method for the routine analysis of
        pitch-and-roll buoy wave data. Journal of physical oceanography, 18(7), 1020-1034.

        :return: normalized Fourier moment cos(theta)
        """
        return self.dataset[NAME_a1]

    @property
    def b1(self) -> DataArray:
        """
        Normalized Fourier moment of the directional distribution function. See Kuik et al. (1988), eq A1.

        Kuik, A. J., Van Vledder, G. P., & Holthuijsen, L. H. (1988). A method for the routine analysis of
        pitch-and-roll buoy wave data. Journal of physical oceanography, 18(7), 1020-1034.

        :return: normalized Fourier moment sin(theta)
        """
        return self.dataset[NAME_b1]

    @property
    def a2(self) -> DataArray:
        """
        Normalized Fourier moment of the directional distribution function. See Kuik et al. (1988), eq A1.

        Kuik, A. J., Van Vledder, G. P., & Holthuijsen, L. H. (1988). A method for the routine analysis of
        pitch-and-roll buoy wave data. Journal of physical oceanography, 18(7), 1020-1034.

        :return: normalized Fourier moment cos(2*theta)
        """
        return self.dataset[NAME_a2]

    @property
    def b2(self) -> DataArray:
        """
        Normalized Fourier moment of the directional distribution function. See Kuik et al. (1988), eq A1.

        Kuik, A. J., Van Vledder, G. P., & Holthuijsen, L. H. (1988). A method for the routine analysis of
        pitch-and-roll buoy wave data. Journal of physical oceanography, 18(7), 1020-1034.

        :return: normalized Fourier moment sin(2*theta)
        """
        return self.dataset[NAME_b2]

    @property
    def e(self):
        """
        Directionally integrated spectra. Synonym for variance_density for 1D spectra.

        :return: directionally integrated spectra.
        """
        return self._spectrum

    def extrapolate_tail(
        self,
        end_frequency,
        power=None,
        tail_energy=None,
        tail_bounds=None,
        tail_moments=None,
        tail_frequency=None,
    ) -> "Spectrum1D":
        """
        Extrapolate the tail using the given power
        :param end_frequency: frequency to extrapolate to
        :param power: power to use. If None, a best fit -4 or -5 tail is used.
        :return:
        """
        e = self.e
        a1 = self.a1
        b1 = self.b1
        a2 = self.a2
        b2 = self.b2

        frequency = self.frequency.values
        frequency_delta = frequency[-1] - frequency[-2]
        n = int((end_frequency - frequency[-1]) / frequency_delta) + 1

        fstart = frequency[-1] + frequency_delta
        fend = frequency[-1] + n * frequency_delta

        if tail_frequency is None:
            tail_frequency = numpy.linspace(fstart, fend, n, endpoint=True)

        tail_frequency = DataArray(
            data=tail_frequency, coords={"frequency": tail_frequency}, dims="frequency"
        )
        variance_density = concat(
            (e, e.isel(frequency=-1) * zeros_like(tail_frequency)), dim="frequency"
        )

        tail_a1 = a1.isel(frequency=-1) if tail_moments is None else tail_moments["a1"]
        tail_b1 = b1.isel(frequency=-1) if tail_moments is None else tail_moments["b1"]
        tail_a2 = a2.isel(frequency=-1) if tail_moments is None else tail_moments["a2"]
        tail_b2 = b2.isel(frequency=-1) if tail_moments is None else tail_moments["b2"]

        a1 = concat((a1, tail_a1 * ones_like(tail_frequency)), dim="frequency")
        b1 = concat((b1, tail_b1 * ones_like(tail_frequency)), dim="frequency")
        a2 = concat((a2, tail_a2 * ones_like(tail_frequency)), dim="frequency")
        b2 = concat((b2, tail_b2 * ones_like(tail_frequency)), dim="frequency")

        if tail_energy is not None:
            if isinstance(tail_energy, DataArray):
                tail_energy = tail_energy.values

            tail_information = (tail_bounds, tail_energy)
        else:
            tail_information = None

        variance_density = DataArray(
            data=numba_fill_zeros_or_nan_in_tail(
                variance_density.values,
                variance_density.frequency.values,
                power,
                tail_information=tail_information,
            ),
            dims=a1.dims,
            coords=a1.coords,
        )

        dataset = Dataset(
            {
                "variance_density": variance_density,
                "a1": a1,
                "b1": b1,
                "a2": a2,
                "b2": b2,
            }
        )

        for name in self.dataset:
            if name in SPECTRAL_VARS:
                continue
            else:
                dataset = dataset.assign({name: self.dataset[name]})

        return Spectrum1D(dataset)

    def downsample(self, frequencies: np.ndarray) -> "Spectrum1D":
        """
        Downsample the spectrum to the given frequencies. To downsample the spectrum we first calculate the cumulative
        density function and then sample the cumulative density function at the given frequencies. This ensures that
        the downsampled spectrum has equal energy within equal frequency intervals. For the directional moments we
        simply sample the moments at the nearest frequency.

        :param frequencies: ordered array of frequencies to downsample to.
        :return: Downsampled spectrum
        """
        cdf = self.cumulative_density_function
        diff = numpy.diff(
            frequencies,
            append=2 * frequencies[-1] - frequencies[-2],
            prepend=2 * frequencies[0] - frequencies[1],
        )
        frequency_step = diff[0:-1] * 0.5 + diff[1:] * 0.5

        sampling_frequencies = numpy.concatenate(([0], numpy.cumsum(frequency_step)))
        sampling_frequencies = (
            sampling_frequencies - frequency_step[0] / 2 + frequencies[0]
        )

        dims = self.dims
        sampled_cdf = cdf.sel({"frequency": sampling_frequencies}, method="nearest")
        data = {
            NAME_e: (dims, sampled_cdf.diff(dim="frequency").values / frequency_step),
            NAME_a1: (
                dims,
                self.a1.sel({"frequency": frequencies}, method="nearest").values,
            ),
            NAME_b1: (
                dims,
                self.b1.sel({"frequency": frequencies}, method="nearest").values,
            ),
            NAME_a2: (
                dims,
                self.a2.sel({"frequency": frequencies}, method="nearest").values,
            ),
            NAME_b2: (
                dims,
                self.b2.sel({"frequency": frequencies}, method="nearest").values,
            ),
        }

        coords = {x: self.dataset[x].values for x in self.dims}
        coords[NAME_F] = frequencies

        for x in self.dataset:
            if x in SPECTRAL_VARS:
                continue
            data[x] = (self.dims_space_time, self.dataset[x].values)

        return Spectrum1D(Dataset(data_vars=data, coords=coords))

    def as_frequency_direction_spectrum(
        self,
        number_of_directions,
        method: Estimators = "mem2",
        solution_method="scipy",
    ) -> "Spectrum2D":
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

        :return: Spectrum2D object

        REFERENCES:
        Benoit, M. (1993). Practical comparative performance survey of methods
            used for estimating directional wave spectra from heave-pitch-roll data.
            In Coastal Engineering 1992 (pp. 62-75).

        Lygre, A., & Krogstad, H. E. (1986). Maximum entropy estimation of the
            directional distribution in ocean wave spectra.
            Journal of Physical Oceanography, 16(12), 2052-2060.

        """

        direction = numpy.linspace(0, 360, number_of_directions, endpoint=False)

        output_array = estimate_directional_spectrum_from_moments(
            self.e.values,
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

        return Spectrum2D(Dataset(data_vars=data, coords=coords))

    @classmethod
    def from_dataset(cls: _T, dataset: Dataset, mapping=None, deep=False) -> _T:
        """
        Create a spectrum object from a xarray dataset. The dataset must either contain
        - the variables: "variance_density", "a1", "b1", "a2", "b2"
        - coordinates: "frequency",

        or a mapping must be provided that maps the dataset names to expected variable/coordinat names. Coordinate units
        are  assumed to be Hz (frequency>=0). Units of the spectrum are assumed to be
        m^2/Hz. The default interpretation of the direction is mathematical (90 degrees is North), and direction
        indicates the direction the waves are going towards, and it is assumed directional moments are consistent with
        this interpretation.

        Note that by default we create a shallow copy of the dataset.

        :param dataset: xarray dataset
        :param mapping: dictionary mapping the xarray dataset names to the spectrum names
        :param deep: If True, create a deep copy of the input dataset.

        :return: Spectrum1D object
        """
        dataset = dataset.copy(deep=deep)
        if mapping is not None:
            dataset = dataset.rename(mapping)

        return cls(dataset)
