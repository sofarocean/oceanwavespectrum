from roguewavespectrum._wavespectrum1D import Spectrum1D
from roguewavespectrum._variable_names import (
    NAME_F,
    NAME_E,
    NAME_a1,
    NAME_b1,
    NAME_a2,
    NAME_b2,
    NAME_e,
)
import numpy as np
from xarray import Dataset
from typing import Union, Mapping


def create_spectrum_1d(
    coordinates: Union[np.ndarray, Mapping],
    variance_density: np.ndarray,
    a1: np.ndarray,
    b1: np.ndarray,
    a2: np.ndarray,
    b2: np.ndarray,
    **kwargs,
) -> Spectrum1D:
    """
    Create a roguewavespectrum.Spectrum1D object for a given spectrum with a1/b1/a2/b2.

    :param coordinates: If a numpy array is given, it is assumed that the array contains the frequency coordinates.
        otherwise a mapping is expected where the key is the name of the coordinate and the value is the coordinate.
        Note that a frequency key is required, and that the final dimension of the variance_density/a1/b1/a2/b2 arrays
        should match the length of the frequency coordinate.
    :param variance: ndarray where the trailing dimension gives variance as function of frequency
    :param a1: ndarray where the trailing dimension gives a1 as function of frequency
    :param b1: ndarray where the trailing dimension gives b1 as function of frequency
    :param a2: ndarray where the trailing dimension gives a2 as function of frequency
    :param b2: ndarray where the trailing dimension gives b2 as function of frequency
    :param kwargs:
    :return: spectrum object
    """

    if isinstance(coordinates, Mapping):
        dims = tuple([key for key in coordinates.keys()])
        coords = coordinates
    else:
        if coordinates.ndim == 1:
            coords = {NAME_F: coordinates}
            dims = (NAME_F,)
        else:
            raise ValueError(
                f"If spectra have dimensions > 1, coordinates should be a mapping."
            )

    dataset = Dataset(
        data_vars={
            NAME_e: (dims, variance_density),
            NAME_a1: (dims, a1),
            NAME_b1: (dims, b1),
            NAME_a2: (dims, a2),
            NAME_b2: (dims, b2),
        },
        coords=coords,
    )
    return Spectrum1D(dataset, **kwargs)
