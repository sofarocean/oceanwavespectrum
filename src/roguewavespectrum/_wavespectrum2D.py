import numpy as np
from typing import TypeVar
from xarray import (
    Dataset,
    DataArray,
)
from ._spectrum import Spectrum
from roguewavespectrum._physical_constants import PhysicsOptions
from ._directions import (
    DirectionalConvention,
    DirectionalUnit,
    convert_unit,
    convert_angle_convention,
    get_angle_convention_and_unit,
)

_T = TypeVar("_T")

from ._variable_names import (
    NAME_D,
    NAME_E,
    NAME_e,
    NAME_F,
    NAME_a2,
    NAME_b2,
    NAME_a1,
    NAME_b1,
    NAME_K,
    SPECTRAL_VARS,
    set_conventions,
)


class Spectrum2D(Spectrum):
    standard_name = "sea_surface_wave_variance_spectral_density"
    units = "m2 Hz-1 deg-1"

    def __init__(
        self, dataset: Dataset, physics_options: PhysicsOptions = None, **kwargs
    ):
        for name in [NAME_F, NAME_D, NAME_E]:
            if name not in dataset and name not in dataset.coords:
                raise ValueError(
                    f"Required variable/coordinate {name} is"
                    f" not specified in the dataset"
                )

        super(Spectrum2D, self).__init__(
            dataset, physics_options=physics_options, **kwargs
        )

    @property
    def _spectrum(self) -> DataArray:
        return self.dataset[NAME_E]

    @_spectrum.setter
    def _spectrum(self, value):
        self.dataset[NAME_E] = value

    @property
    def direction_step(self) -> DataArray:
        """
        Calculate the step size between the direction bins. Because the direction bins are circular, we use a modular
        difference estimate.
        :return:
        """
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
    def radian_direction_mathematical(self) -> DataArray:
        return self.direction(
            directional_unit="rad", directional_convention="mathematical"
        )

    def _directionally_integrate(self, data_array: DataArray) -> DataArray:
        return (data_array * self.direction_step).sum(NAME_D, skipna=True)

    @property
    def directional_variance_density(self) -> DataArray:
        return self._spectrum

    @property
    def e(self) -> DataArray:
        """
        Return the directionally integrated spectrum.

        :return: 1D spectral values (directionally integrated spectrum).
        """
        return set_conventions(
            self._directionally_integrate(self._spectrum), NAME_e, overwrite=True
        )

    @property
    def a1(self) -> DataArray:
        data_array = (
            self._directionally_integrate(
                self._spectrum * np.cos(self.radian_direction_mathematical)
            )
            / self.e
        )
        return set_conventions(data_array, NAME_a1, overwrite=True)

    @property
    def b1(self) -> DataArray:
        data_array = (
            self._directionally_integrate(
                self._spectrum * np.sin(self.radian_direction_mathematical)
            )
            / self.e
        )
        return set_conventions(data_array, NAME_b1, overwrite=True)

    @property
    def a2(self) -> DataArray:
        data_array = (
            self._directionally_integrate(
                self._spectrum * np.cos(2 * self.radian_direction_mathematical)
            )
            / self.e
        )
        return set_conventions(data_array, NAME_a2, overwrite=True)

    @property
    def b2(self) -> DataArray:
        data_array = (
            self._directionally_integrate(
                self._spectrum * np.sin(2 * self.radian_direction_mathematical)
            )
            / self.e
        )
        return set_conventions(data_array, NAME_b2, overwrite=True)

    def direction(
        self,
        directional_unit: DirectionalUnit = "degree",
        directional_convention: DirectionalConvention = "mathematical",
    ) -> DataArray:
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

    def as_frequency_spectrum(self):
        return _circular_dependency_workaround(self)

    @property
    def number_of_directions(self) -> int:
        return len(self.direction())

    @property
    def wavenumber_directional_spectral_density(self) -> DataArray:
        """
        Wavenumber Spectral density. Conversion through multiplication with the Jacobian of the
        transformation such that

            E(f) df = E(k) dk

        with e the density as function of frequency (f) or wavenumber (k), and df and dk the differentials of the
        respective variables. Note that with w = 2 * pi * f, the Jacobian is equal to

        df/dk =   dw/dk * df/dw = groupspeed / ( 2 * pi)

        :return: Wavenumber spectral density.
        """
        wavenumber = self.wavenumber
        data_array = set_conventions(
            self.variance_density * self.groupspeed / (np.pi * 2),
            "wavenumber_directional_variance_density",
            overwrite=True,
        )
        data_array = data_array.assign_coords({NAME_K: wavenumber})
        return data_array


def _circular_dependency_workaround(spectrum: Spectrum2D):
    from ._wavespectrum1D import Spectrum1D

    dataset = {
        "a1": spectrum.a1,
        "b1": spectrum.b1,
        "a2": spectrum.a2,
        "b2": spectrum.b2,
        "variance_density": spectrum.e,
    }
    for name in spectrum.dataset:
        if name not in SPECTRAL_VARS:
            dataset[name] = spectrum.dataset[name]

    return Spectrum1D(Dataset(dataset))
