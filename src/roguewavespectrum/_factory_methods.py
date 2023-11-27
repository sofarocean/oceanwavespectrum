"""
Factory methods to help create spectra. Mostly for convinience
"""

from roguewavespectrum._spectrum import Spectrum
from roguewavespectrum._variable_names import (
    NAME_F,
    NAME_a1,
    NAME_b1,
    NAME_a2,
    NAME_b2,
    NAME_e,
    NAME_E,
    NAME_D,
    NAME_DEPTH,
)

import numpy as np
from xarray import Dataset
from typing import Union, Sequence, Literal
from roguewavespectrum.parametric import (
    create_dir_shape,
    create_freq_shape,
    parametric_directional_spectrum,
    parametric_spectrum,
)
from roguewavespectrum._operations import concatenate_spectra


def create_spectrum1d(
    coordinates: Union[
        np.ndarray, tuple[str, np.ndarray], Sequence[tuple[str, np.ndarray]]
    ],
    variance_density: np.ndarray,
    a1: np.ndarray,
    b1: np.ndarray,
    a2: np.ndarray,
    b2: np.ndarray,
    **kwargs,
) -> Spectrum:
    """
    Create a roguewavespectrum.Spectrum1D object for a given 1D spectrum with a1/b1/a2/b2.

    :param coordinates:  A list of coordinates naming the dimension and giving the coordinate values. For a 1D spectrum
    that has dimensions [dim1,...,dimN,frequency] The list should take the form:

        [(dim1, np.ndarray), (dim2, np.ndarray), ..., (dimN, np.ndarray), ("frequency", np.ndarray)]

    Note that the frequency coordinate is required, and MUST be the last coordinate.

    For convinience, if a numpy array is given, it is assumed that the array contains the frequency coordinates, and the
    coordinate is implicitly assumed to be [("frequency", coordinates)].

    If only the frequency is given, but the spectrum has more than 1 dimension, the other dimensions are given default
    names "dim_0", "dim_1", etc. and given integer coordinates.

    If more coordeniates than frequency are given, but the number of coordinates does not match the number of dimensions
    of the spectrum, a ValueError is raised.

    :param variance: ndarray where the trailing dimension gives variance as function of frequency
    :param a1: ndarray where the trailing dimension gives a1 as function of frequency
    :param b1: ndarray where the trailing dimension gives b1 as function of frequency
    :param a2: ndarray where the trailing dimension gives a2 as function of frequency
    :param b2: ndarray where the trailing dimension gives b2 as function of frequency
    :param kwargs:
    :return: spectrum object
    """

    if isinstance(coordinates, np.ndarray):
        coordinates = [(NAME_F, coordinates)]

    if isinstance(coordinates[0], str):
        coordinates = [coordinates]

    elif not isinstance(coordinates, Sequence):
        raise ValueError(
            "coordinates should be a numpy array or a sequence of tuples describing the coordinates"
        )

    if not (a1.shape == b1.shape == a2.shape == b2.shape == variance_density.shape):
        raise ValueError("a1,b1,a2,b2 and variance_density should have the same shape")

    if not coordinates[-1][0] == "frequency":
        raise ValueError("The last coordinate should be frequency")

    if len(coordinates) < variance_density.ndim:
        if len(coordinates) == 1:
            coordinates = [
                (f"dim_{ndim}", np.arange(variance_density.shape[ndim]))
                for ndim in range(variance_density.ndim - 1)
            ] + coordinates
        else:
            raise ValueError(
                "The number of coordinates should match the number of dimensions of the variance_density"
            )

    dims = [x[0] for x in coordinates]
    coords = {x[0]: x[1] for x in coordinates}

    if len(dims) != variance_density.ndim:
        raise ValueError(
            "The number of coordinates should match the number of dimensions of the variance_density"
        )

    shape = variance_density.shape
    for index, dim in enumerate(dims):
        coor = coords[dim]
        if not isinstance(coor, np.ndarray):
            raise ValueError(
                f"Coordinate number {index} associated with {dim} should be a numpy array"
            )

        if len(coor) != shape[index]:
            raise ValueError(
                f"Size of coordinate number {index} associated with {dim} does not match the shape of the "
                f"variance_density {shape}"
            )
    depth = kwargs.get("depth", np.inf)
    if np.isscalar(depth):
        depth = np.zeros(variance_density.shape[:-1]) + depth

    dataset = Dataset(
        data_vars={
            NAME_e: (dims, variance_density),
            NAME_a1: (dims, a1),
            NAME_b1: (dims, b1),
            NAME_a2: (dims, a2),
            NAME_b2: (dims, b2),
            NAME_DEPTH: (dims[:-1], depth),
        },
        coords=coords,
    )
    return Spectrum(dataset, **kwargs)


def create_spectrum2d(
    coordinates: Union[tuple[np.ndarray, np.ndarray], Sequence[tuple[str, np.ndarray]]],
    variance_density: np.ndarray,
    **kwargs,
) -> Spectrum:
    """
    Create a roguewavespectrum.Spectrum object for a given spectrum. Spectral units are [m^2/Hz/degree]. Frequency
    units are [Hz], direction units are [degrees] and are measured counter-clockwise from east, and refer to the
    direction the waves are travelling *to*.

    :param coordinates:  A list of coordinates naming the dimension and giving the coordinate values. For a 2D spectrum
    that has dimensions [dim1,...,dimN,frequency,direction] The list should take the form:

        [(dim1, np.ndarray), ..., (dimN, np.ndarray), ("frequency", np.ndarray),("direction", np.ndarray)]

    Note that the frequency amd direction coordinates are required, and MUST be the last coordinates (in that order).

    For convinience, if a tuple of numpy arrays is given, it is assumed that the array contains the
    frequency and direction coordinates (in that order), and the coordinate is implicitly assumed to be
    [("frequency", coordinates[0]),("direction", coordinates[1])].

    If only the frequency and direction are given, but the spectrum has more than 2 dimensions, the other dimensions are
    given default names "dim_0", "dim_1", etc. and given integer coordinates.

    If more coordeniates than frequency and direction are given, but the number of coordinates does not match the number
    of dimensions of the spectrum, a ValueError is raised.

    :param variance: ndarray where the trailing dimension gives variance as function of frequency
    :param a1: ndarray where the trailing dimension gives a1 as function of frequency
    :param b1: ndarray where the trailing dimension gives b1 as function of frequency
    :param a2: ndarray where the trailing dimension gives a2 as function of frequency
    :param b2: ndarray where the trailing dimension gives b2 as function of frequency
    :param kwargs:
    :return: spectrum object
    """

    if isinstance(coordinates, Sequence):
        if isinstance(coordinates[0], np.ndarray) and isinstance(
            coordinates[1], np.ndarray
        ):
            coordinates = [(NAME_F, coordinates[0]), (NAME_D, coordinates[1])]

    elif not isinstance(coordinates, Sequence):
        raise ValueError(
            "coordinates should be a sequence of numpy arrays or a sequence of tuples describing the "
            "coordinates"
        )

    if not coordinates[-2][0] == "frequency":
        raise ValueError("The second last coordinate should be frequency")

    if not coordinates[-1][0] == "direction":
        raise ValueError("The last coordinate should be direction")

    if len(coordinates) < variance_density.ndim:
        if len(coordinates) == 2:
            coordinates = [
                (f"dim_{ndim}", np.arange(variance_density.shape[ndim]))
                for ndim in range(variance_density.ndim - 2)
            ] + coordinates
        else:
            raise ValueError(
                "The number of coordinates should match the number of dimensions of the variance_density"
            )

    dims = [x[0] for x in coordinates]
    coords = {x[0]: x[1] for x in coordinates}

    if len(dims) != variance_density.ndim:
        raise ValueError(
            "The number of coordinates should match the number of dimensions of the variance_density"
        )

    shape = variance_density.shape
    for index, dim in enumerate(dims):
        coor = coords[dim]
        if not isinstance(coor, np.ndarray):
            raise ValueError(
                f"Coordinate number {index} associated with {dim} should be a numpy array"
            )

        if len(coor) != shape[index]:
            raise ValueError(
                f"Size of coordinate number {index} associated with {dim} does not match the shape of the "
                f"variance_density {shape}"
            )

    depth = kwargs.get("depth", np.inf)
    if np.isscalar(depth):
        depth = np.zeros(variance_density.shape[:-2]) + depth

    dataset = Dataset(
        data_vars={
            NAME_E: (dims, variance_density),
            NAME_DEPTH: (dims[:-2], depth),
        },
        coords=coords,
    )
    return Spectrum(dataset, **kwargs)


def create_parametric_spectrum1d(
    frequencies: np.ndarray,
    waveheight: Union[float, np.ndarray],
    period: Union[float, np.ndarray],
    shape_name: Literal["jonswap", "pm", "phillips", "gaussian"] = "jonswap",
) -> Spectrum:
    """
    Create a parametric 1D spectrum with a given shape. Shapes are:
    jonswap, pm, phillips, gaussian. See roguewavespectrum.parametric for details.

    :param frequencies: frequency array in Hz
    :param waveheight: significant waveheight [m]
    :param period: characteristic period [s] (peak period)
    :param shape_name: one of 'jonswap', 'pm', 'phillips', 'gaussian'
    :return: 1D spectrum
    """
    waveheight = np.atleast_1d(waveheight)
    period = np.atleast_1d(period)

    if not (len(waveheight) == len(period)):
        raise ValueError(
            "Waveheight and period should be equal length vectors, or scalars."
        )

    out = []
    for hs, tp in zip(waveheight, period):
        freq_shape = create_freq_shape(tp, hs, shape_name)
        out += [parametric_spectrum(frequencies, freq_shape)]

    spectra = concatenate_spectra(out)
    return spectra


def create_parametric_spectrum2d(
    frequencies: np.ndarray,
    number_of_directions: int,
    waveheight: Union[float, np.ndarray],
    period: Union[float, np.ndarray],
    direction: Union[float, np.ndarray],
    spread: Union[float, np.ndarray],
    frequency_shape_name: Literal["jonswap", "pm", "phillips", "gaussian"] = "jonswap",
    direction_shape_name: Literal["cosN", "cos2N"] = "cosN",
) -> Spectrum:
    """
    Create a parametric 2D spectrum with a given frequency and direction shape.
    :param frequencies: frequency array in Hz
    :param number_of_directions: number of directions. The directions are equally spaced between 0 and 360 degrees and
     given by np.linspace(0,360,number_of_directions,endpoint=False)
    :param waveheight: significant waveheight [m]
    :param period: characteristic period [s] (peak period)
    :param direction: mean direction [degrees]
    :param spread: mean spread [degrees]
    :param frequency_shape_name: one of 'jonswap', 'pm', 'phillips', 'gaussian'
    :param direction_shape_name: one of 'cosN', 'cos2N'
    :return: 2D spectrum
    """

    waveheight = np.atleast_1d(waveheight)  # type: np.ndarray
    period = np.atleast_1d(period)  # type: np.ndarray
    direction = np.atleast_1d(direction)  # type: np.ndarray
    spread = np.atleast_1d(spread)  # type: np.ndarray

    if not (len(waveheight) == len(period)) == len(direction) == len(spread):
        raise ValueError(
            "waveheight, period, direction and spread should either all be equal length vectors, "
            "or scalars"
        )

    out = []
    directions = np.linspace(0, 360, number_of_directions, endpoint=False)
    for hs, tp, _dir, sprd in zip(waveheight, period, direction, spread):
        freq_shape = create_freq_shape(tp, hs, frequency_shape_name)
        dir_shape = create_dir_shape(_dir, sprd, direction_shape_name)
        out += [
            parametric_directional_spectrum(
                frequencies, directions, freq_shape, dir_shape
            )
        ]

    spectrum = concatenate_spectra(out)
    return spectrum
