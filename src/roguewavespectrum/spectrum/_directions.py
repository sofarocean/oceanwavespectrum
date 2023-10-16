import numpy as np
from .variable_names import set_conventions
from typing import TypeVar, Literal

_T = TypeVar("_T")

DirectionalUnit = Literal["degree", "rad"]
DirectionalConvention = Literal["oceanographical", "mathematical", "meteorological"]


def wave_mean_direction(
    a1: _T,
    b1: _T,
    unit: DirectionalUnit = "degree",
    convention: DirectionalConvention = "mathematical",
    name=None,
) -> _T:
    angle = convert_unit(np.arctan2(b1, a1), unit, "rad")

    angle = convert_angle_convention(angle, convention, "mathematical", unit)

    if name is not None:
        angle = set_conventions(angle, [name, convention], overwrite=True)
        angle.attrs["units"] = unit
    return angle


def wave_directional_spread(
    a1: _T, b1: _T, unit: DirectionalUnit = "degree", name=None
) -> _T:
    spread = convert_unit(
        np.sqrt(2 - 2 * np.sqrt(a1**2 + b1**2)), to=unit, _from="rad"
    )
    if name is not None:
        spread = set_conventions(spread, name, overwrite=True)
        spread.attrs["units"] = unit
    return spread


def convert_unit(
    angle: _T, to: DirectionalUnit = "degree", _from: DirectionalUnit = "degree"
) -> _T:
    """
    Convert angle from one unit to another.
    :param angle: angle in rad or degree
    :param to: unit to convert to, one of 'degree', 'rad'
    :param _from: unit to convert from, one of 'degree', 'rad'
    :return:
    """
    if to == _from and to in ["degree", "rad"]:
        return angle

    elif to == "degree":
        return angle * 180 / np.pi

    elif to == "rad":
        return angle / 180 * np.pi

    else:
        if to not in ["degree", "rad"]:
            raise ValueError("Unknown unit to convert to")
        else:
            raise ValueError("Unknown unit to convert from")


def convert_angle_convention(
    angle: _T,
    to_convention: DirectionalConvention = "mathematical",
    from_convention: DirectionalConvention = "mathematical",
    units: DirectionalUnit = "rad",
) -> _T:
    """
    Convert angle from one convention to another. Conventions are:

    - mathematical: 0 degree / rad is east, going to, measured positive counterclockwise.
    - oceanographical: 0 degree / rad is north, going to, measured positive clockwise.
    - meteorological: 0 / rad degree is north, coming from, measured positive counterclockwise.

    :param angle: angle in rad or degree
    :param to_convention: convention to convert to, one of 'mathematical', 'oceanographical', 'meteorological'
    :param from_convention: convention to convert from, one of 'mathematical', 'oceanographical', 'meteorological'
    :param units: default 'rad', one of 'rad', 'degree'
    :return: angle in rad or degree (depending on units), in the new convention
    """

    if units == "degree":
        wrapping_length = 360

    elif units == "rad":
        wrapping_length = 2 * np.pi
    else:
        raise ValueError("Unknown units")

    if from_convention == "mathematical":
        pass
    elif from_convention == "oceanographical":
        angle = (wrapping_length / 4 - angle) % (wrapping_length)
    elif from_convention == "meteorological":
        angle = (3 * wrapping_length / 4 - angle) % (wrapping_length)
    else:
        raise ValueError("Unknown convention")

    if to_convention == "mathematical":
        return angle

    elif to_convention == "oceanographical":
        return (wrapping_length / 4 - angle) % (wrapping_length)

    elif to_convention == "meteorological":
        return (3 * wrapping_length / 4 - angle) % (wrapping_length)

    else:
        raise ValueError("Unknown convention")


def get_angle_convention_and_unit(
    dataarray, default_convention="mathematical", default_unit="degree"
):
    if "standard_name" in dataarray.attrs:
        if dataarray.attrs["standard_name"] == "sea_surface_wave_to_direction":
            convention = "oceanographical"
        elif dataarray.attrs["standard_name"] == "sea_surface_wave_from_direction":
            convention = "meteorological"
        else:
            raise ValueError("Unknown standard_name", dataarray.attrs["standard_name"])
    else:
        convention = default_convention

    if "units" in dataarray.attrs:
        unit = dataarray.attrs["units"]
    else:
        unit = default_unit
    return convention, unit
