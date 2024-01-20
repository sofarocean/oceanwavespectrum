from ._spectrum import Spectrum
from typing import Sequence, List
from xarray import concat, Dataset, DataArray
from ._variable_names import NAME_E, NAME_e
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


def multiply(
    spectrum: Spectrum, array: np.ndarray, dimensions: List[str] = None, inplace=False
) -> Spectrum:
    """
    Multiply the variance density with the given numpy array. Broadcasting is performed automatically if dimensions
    are provided. If no dimensions are provided the array needs to have the exact same shape as the variance
    density array.

    :param array: Array to multiply with variance density
    :param dimension: Dimensions of the array
    :return: self
    """
    if inplace:
        output = spectrum
    else:
        output = spectrum.copy()

    coords = {}
    shape = array.shape

    if output.is_1d:
        target = NAME_e
    else:
        target = NAME_E

    if dimensions is None:
        if shape != spectrum.shape:
            raise ValueError(
                "If no dimensions are provided the array must have the exact same shape as the"
                "variance density array."
            )

        output.dataset[target] = output.dataset[target] * array
        return output

    if len(shape) != len(dimensions):
        raise ValueError(
            "The dimensions of the input array must match the number of dimension labels"
        )

    for length, dimension in zip(shape, dimensions):
        if dimension not in spectrum.dims:
            raise ValueError(
                f"Dimension {dimension} not a valid dimension of the spectral object."
            )
        coords[dimension] = output.dataset[dimension].values

        if len(output.dataset[dimension].values) != length:
            raise ValueError(
                f"Array length along the dimension {dimension} does not match the length of the"
                f"coordinate of the same name in the spctral object."
            )

    data = DataArray(data=array, coords=coords, dims=dimensions)
    output.dataset[target] = spectrum.dataset[target] * data
    return output
