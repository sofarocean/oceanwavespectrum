from ._spectrum import Spectrum
from ._variable_names import SPECTRAL_VARS
from typing import Sequence
from xarray import concat, Dataset, DataArray
from ._extrapolate import numba_fill_zeros_or_nan_in_tail
import numpy as np


def concatenate_spectra(spectra: Sequence[Spectrum], dim=None, **kwargs) -> Spectrum:
    """
    Concatenate along the given dimension. If the dimension does not exist a new dimension will be created. Under the
    hood this calls the concat function of xarray. Named arguments to that function can be applied here as well.

    If dim is set to None - we first flatten the spectral objects - and then join along the flattened dimension.

    :param spectra: A sequence of Frequency Spectra/Frequency Direction Spectra
    :param dim: the dimension to concatenate along
    :return: New combined spectral object.
    """

    # Concatenate the dataset in the spectral objects using the xarray concatenate function
    dataset = Dataset()

    if dim is None:
        dim = "index"
        spectra = [x.flatten(dim) for x in spectra]

    for variable_name in spectra[0].dataset:
        if variable_name == dim:
            continue

        dataset[variable_name] = concat(
            [x.dataset[variable_name] for x in spectra], dim=dim, **kwargs
        )

    # Get the class of the input spectra.
    cls = type(spectra[0])

    # Return a class instance.
    return cls(dataset)


def fill_zeros_or_nan_in_tail(
    spectrum: Spectrum,
    power=None,
    tail_energy=None,
    tail_bounds=None,
) -> Spectrum:
    variance_density = spectrum.variance_density
    a1 = spectrum.a1
    b1 = spectrum.b1
    a2 = spectrum.a2
    b2 = spectrum.b2

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

    for name in spectrum.dataset:
        if name in SPECTRAL_VARS:
            continue
        else:
            dataset = dataset.assign({name: spectrum.dataset[name]})

    return Spectrum(dataset)


def extrapolate_tail(
    spectrum: Spectrum,
    end_frequency: float,
    power=None,
    tail_energy=None,
    tail_bounds=None,
    tail_moments=None,
    tail_frequency=None,
) -> Spectrum:
    """
    Extrapolate the tail using the given power and return a 1D frequency spectrum.
    :param end_frequency: frequency to extrapolate to
    :param power: power to use. If None, a best fit -4 or -5 tail is used.
    :return:
    """
    e = spectrum.variance_density
    a1 = spectrum.a1
    b1 = spectrum.b1
    a2 = spectrum.a2
    b2 = spectrum.b2

    frequency = spectrum.frequency.values
    frequency_delta = frequency[-1] - frequency[-2]
    n = int((end_frequency - frequency[-1]) / frequency_delta) + 1

    fstart = frequency[-1] + frequency_delta
    fend = frequency[-1] + n * frequency_delta

    if tail_frequency is None:
        tail_frequency = np.linspace(fstart, fend, n, endpoint=True)

    tail_frequency = DataArray(
        data=tail_frequency, coords={"frequency": tail_frequency}, dims="frequency"
    )
    variance_density = concat(
        (e, e.isel(frequency=-1) * np.zeros_like(tail_frequency)), dim="frequency"
    )

    tail_a1 = a1.isel(frequency=-1) if tail_moments is None else tail_moments["a1"]
    tail_b1 = b1.isel(frequency=-1) if tail_moments is None else tail_moments["b1"]
    tail_a2 = a2.isel(frequency=-1) if tail_moments is None else tail_moments["a2"]
    tail_b2 = b2.isel(frequency=-1) if tail_moments is None else tail_moments["b2"]

    a1 = concat((a1, tail_a1 * np.ones_like(tail_frequency)), dim="frequency")
    b1 = concat((b1, tail_b1 * np.ones_like(tail_frequency)), dim="frequency")
    a2 = concat((a2, tail_a2 * np.ones_like(tail_frequency)), dim="frequency")
    b2 = concat((b2, tail_b2 * np.ones_like(tail_frequency)), dim="frequency")

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

    for name in spectrum.dataset:
        if name in SPECTRAL_VARS:
            continue
        else:
            dataset = dataset.assign({name: spectrum.dataset[name]})

    return Spectrum(dataset)
