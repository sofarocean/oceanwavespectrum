from .wavespectrum import (
    WaveSpectrum
)
from .variable_names import SPECTRAL_VARS
from .wavespectrum1D import FrequencySpectrum
from typing import Sequence
from xarray import concat, Dataset, DataArray
from .extrapolate import numba_fill_zeros_or_nan_in_tail

def concatenate_spectra(spectra: Sequence[WaveSpectrum], dim=None, **kwargs) -> WaveSpectrum:
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
    spectrum: WaveSpectrum,
    power=None,
    tail_energy=None,
    tail_bounds=None,
) -> FrequencySpectrum:
    variance_density = spectrum.e
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

    return FrequencySpectrum(dataset)