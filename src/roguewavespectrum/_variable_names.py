from typing import Literal

import numpy as np
from xarray import DataArray
import warnings

NAME_F: Literal["frequency"] = "frequency"
NAME_K: Literal["wavenumber"] = "wavenumber"
NAME_W: Literal["angular_frequency"] = "angular_frequency"
NAME_D: Literal["direction"] = "direction"
NAME_T: Literal["time"] = "time"
NAME_E: Literal["directional_variance_density"] = "directional_variance_density"
NAME_e: Literal["variance_density"] = "variance_density"
NAME_a1: Literal["a1"] = "a1"
NAME_b1: Literal["b1"] = "b1"
NAME_a2: Literal["a2"] = "a2"
NAME_b2: Literal["b2"] = "b2"
NAME_LAT: Literal["latitude"] = "latitude"
NAME_LON: Literal["longitude"] = "longitude"
NAME_DEPTH: Literal["depth"] = "depth"

NAMES_2D = (NAME_F, NAME_D, NAME_T, NAME_E, NAME_LAT, NAME_LON, NAME_DEPTH)
NAMES_1D = (
    NAME_F,
    NAME_T,
    NAME_e,
    NAME_LAT,
    NAME_LON,
    NAME_a1,
    NAME_b1,
    NAME_a2,
    NAME_b2,
    NAME_DEPTH,
)
SPECTRAL_VARS = (NAME_E, NAME_e, NAME_a1, NAME_b1, NAME_a2, NAME_b2)
SPECTRAL_MOMENTS = (NAME_a1, NAME_b1, NAME_a2, NAME_b2)
SPECTRAL_DIMS = (NAME_F, NAME_D)
SPACE_TIME_DIMS = (NAME_T, NAME_LON, NAME_LAT)

cf_conventions = {
    NAME_E: {
        "units": "m^2/Hz/degree",
        "standard_name": "sea_surface_wave_directional_variance_spectral_density",
        "long_name": "Spectral density",
        "valid_min": 0,
        "missing_value": np.NAN,
    },
    NAME_e: {
        "units": "m^2/Hz",
        "standard_name": "sea_surface_wave_variance_spectral_density",
        "long_name": "Spectral density",
        "valid_min": 0,
        "missing_value": np.NAN,
    },
    NAME_a1: {
        "long_name": "First order cosine coefficient",
        "valid_min": -1,
        "valid_max": 1,
        "missing_value": np.NAN,
    },
    NAME_b1: {
        "long_name": "First order sine coefficient",
        "valid_min": -1,
        "valid_max": 1,
        "missing_value": np.NAN,
    },
    NAME_a2: {
        "long_name": "Second order cosine coefficient",
        "valid_min": -1,
        "valid_max": 1,
        "missing_value": np.NAN,
    },
    NAME_b2: {
        "long_name": "Second order sine coefficient",
        "valid_min": -1,
        "valid_max": 1,
        "missing_value": np.NAN,
    },
    NAME_LAT: {
        "units": "degree_north",
        "standard_name": "latitude",
        "long_name": "Latitude",
        "valid_min": -90,
        "valid_max": 90,
    },
    NAME_LON: {
        "units": "degree_east",
        "standard_name": "longitude",
        "long_name": "Longitude",
        "valid_min": -180,
        "valid_max": 360,
    },
    NAME_F: {
        "units": "Hz",
        "standard_name": "sea_surface_wave_frequency",
        "long_name": "Frequency",
        "missing_value": np.NAN,
        "valid_min": 0,
    },
    NAME_W: {
        "units": "rad/s",
        "standard_name": "sea_surface_wave_frequency",
        "long_name": "Frequency",
        "missing_value": np.NAN,
        "valid_min": 0,
    },
    NAME_DEPTH: {
        "units": "m",
        "standard_name": "sea_floor_depth_below_geoid",
        "long_name": "Depth",
        "missing_value": np.NAN,
    },
    NAME_T: {
        # To note- xarray automatically encodes this information when writing to netcdf. Otherwise, datetime64 encodes
        # the needed information already.
        "standard_name": "time",
        "long_name": "Time",
    },
    "Hm0": {
        "units": "m",
        "standard_name": "sea_surface_wave_significant_height",
        "long_name": "Significant wave height",
        "missing_value": np.NAN,
        "valid_min": 0,
    },
    "Tp": {
        "units": "s",
        "standard_name": "sea_surface_wave_mean_period",
        "long_name": "Mean wave period",
        "missing_value": np.NAN,
        "valid_min": 0,
    },
    "Tm01": {
        "units": "s",
        "standard_name": "sea_surface_wave_mean_period_from_variance_spectral_density_first_frequency_moment",
        "long_name": "Mean wave period",
        "missing_value": np.NAN,
        "valid_min": 0,
    },
    "Tm02": {
        "units": "s",
        "standard_name": "sea_surface_wave_mean_period_from_variance_spectral_density_second_frequency_moment",
        "long_name": "Mean wave period",
        "missing_value": np.NAN,
        "valid_min": 0,
    },
    "M2": {
        # Not an official cf-standard name
        "units": "m^2 Hz^2",
        "long_name": "Second frequency moment",
        "missing_value": np.NAN,
        "valid_min": 0,
    },
    "M1": {
        # Not an official cf-standard name
        "units": "m^2 Hz",
        "long_name": "First frequency moment",
        "missing_value": np.NAN,
        "valid_min": 0,
    },
    "M0": {
        # Not an official cf-standard name
        "units": "m^2",
        "long_name": "Variance",
        "missing_value": np.NAN,
        "valid_min": 0,
    },
    "MN": {
        # Not an official cf-standard name
        "units": "m^2 Hz^N",
        "long_name": "M frequency moment",
        "missing_value": np.NAN,
        "valid_min": 0,
    },
    "frequency_bins": {
        # Not an official cf-standard name
        "units": "Hz",
        "long_name": "frequency interval",
        "missing_value": np.NAN,
        "valid_min": 0,
    },
    "direction_bins": {
        # Not an official cf-standard name
        "units": "degree",
        "long_name": "direction interval",
        "missing_value": np.NAN,
        "valid_min": 0,
        "valid_max": 360,
    },
    "A1": {
        # Not an official cf-standard name
        "units": "m^2/Hz",
        "long_name": "First order cosine coefficient of the directional spectrum",
    },
    "B1": {
        # Not an official cf-standard name
        "units": "m^2/Hz",
        "long_name": "First order sine coefficient of the directional spectrum",
    },
    "A2": {
        # Not an official cf-standard name
        "units": "m^2/Hz",
        "long_name": "Second order cosine coefficient of the directional spectrum",
    },
    "B2": {
        # Not an official cf-standard name
        "units": "m^2/Hz",
        "long_name": "Secpmd order sine coefficient of the directional spectrum",
    },
    "C": {
        # Not an official cf-standard name
        "units": "m/s",
        "long_name": "Wave phase speed",
        "missing_value": np.NAN,
        "valid_min": 0,
    },
    "Cp": {
        # Not an official cf-standard name
        "units": "m/s",
        "long_name": "Peak wave phase speed",
        "missing_value": np.NAN,
        "valid_min": 0,
    },
    "group_speed": {
        # Not an official cf-standard name
        "units": "m/s",
        "long_name": "Group speed",
        "missing_value": np.NAN,
        "valid_min": 0,
    },
    "wave_age": {
        # Not an official cf-standard name
        "long_name": "Wave age",
        "missing_value": np.NAN,
        "valid_min": 0,
    },
    "wavelength": {
        # Not an official cf-standard name
        "units": "m",
        "long_name": "Wavelength",
        "missing_value": np.NAN,
        "valid_min": 0,
    },
    NAME_K: {
        "units": "rad/m",
        "long_name": "Wavenumber",
        "missing_value": np.NAN,
        "valid_min": 0,
    },
    "wavenumber_directional_variance_density": {
        "units": "m^3/rad/degree",
        "long_name": "Wavenuber spectral density",
        "missing_value": np.NAN,
        "valid_min": 0,
    },
    "wavenumber_variance_density": {
        "units": "m^3/rad",
        "long_name": "Wavenumber spectral density",
        "missing_value": np.NAN,
        "valid_min": 0,
    },
    "saturation_spectrum": {
        "units": "1/Hz/m",
        "long_name": "Saturation spectrum",
        "missing_value": np.NAN,
        "valid_min": 0,
    },
    "slope_spectrum": {
        "units": "1/Hz",
        "long_name": "Slope spectrum",
        "missing_value": np.NAN,
        "valid_min": 0,
    },
    "mean_squared_slope": {
        "long_name": "Mean squared slope",
        "missing_value": np.NAN,
        "valid_min": 0,
    },
    "peak_frequency": {
        "units": "Hz",
        "long_name": "Peak wave frequency",
        "standard_name": "sea_surface_wave_frequency_at_variance_spectral_density_maximum",
        "missing_value": np.NAN,
        "valid_min": 0,
    },
    "peak_angular_frequency": {
        "units": "rad/s",
        "long_name": "Peak wave angular frequency",
        "missing_value": np.NAN,
        "valid_min": 0,
    },
    "mean_direction": {
        "__default__": "mathematical",
        "meteorological": {
            "units": "degree",
            "long_name": "Wave direction",
            "missing_value": np.NAN,
            "valid_min": 0,
            "valid_max": 360,
            "standard_name": "sea_surface_wave_from_direction",
            "comment": "Direction the waves are coming from, measured clockwise from North",
        },
        "oceanographical": {
            "units": "degree",
            "long_name": "Wave direction",
            "missing_value": np.NAN,
            "valid_min": 0,
            "valid_max": 360,
            "standard_name": "sea_surface_wave_to_direction",
            "comment": "Direction the waves are going to, measured clockwise from North",
        },
        "mathematical": {
            "units": "degree",
            "long_name": "Wave direction",
            "missing_value": np.NAN,
            "valid_min": 0,
            "valid_max": 360,
            "comment": "Direction the waves are going to, measured anti-clockwise from East",
        },
    },
    NAME_D: {
        "__default__": "mathematical",
        "meteorological": {
            "units": "degree",
            "long_name": "Wave direction",
            "missing_value": np.NAN,
            "valid_min": 0,
            "valid_max": 360,
            "standard_name": "sea_surface_wave_from_direction",
            "comment": "Direction the waves are coming from, measured clockwise from North",
        },
        "oceanographical": {
            "units": "degree",
            "long_name": "Wave direction",
            "missing_value": np.NAN,
            "valid_min": 0,
            "valid_max": 360,
            "standard_name": "sea_surface_wave_to_direction",
            "comment": "Direction the waves are going to, measured clockwise from North",
        },
        "mathematical": {
            "units": "degree",
            "long_name": "Wave direction",
            "missing_value": np.NAN,
            "valid_min": 0,
            "valid_max": 360,
            "comment": "Direction the waves are going to, measured anti-clockwise from East",
        },
    },
    "peak_direction": {
        "__default__": "mathematical",
        "meteorological": {
            "units": "degree",
            "long_name": "Peak wave direction",
            "missing_value": np.NAN,
            "valid_min": 0,
            "valid_max": 360,
            "standard_name": "sea_surface_wave_from_direction_at_variance_spectral_density_maximum",
            "comment": "Direction the waves are coming from, measured clockwise from North",
        },
        "oceanographical": {
            "units": "degree",
            "long_name": "Peak wave direction",
            "missing_value": np.NAN,
            "valid_min": 0,
            "valid_max": 360,
            "standard_name": "sea_surface_wave_to_direction_at_variance_spectral_density_maximum",
            "comment": "Direction the waves are going to, measured clockwise from North",
        },
        "mathematical": {
            "units": "degree",
            "long_name": "Peak wave direction",
            "missing_value": np.NAN,
            "valid_min": 0,
            "valid_max": 360,
            "comment": "Direction the waves are going to, measured anti-clockwise from East",
        },
    },
    "mean_directional_spread": {
        "units": "degree",
        "long_name": "Wave directional spread",
        "missing_value": np.NAN,
        "valid_min": 0,
        "valid_max": 360,
        "standard_name": "sea_surface_wave_directional_spread",
    },
    "directional_spread": {
        "units": "degree",
        "long_name": "Wave directional spread",
        "missing_value": np.NAN,
        "valid_min": 0,
        "valid_max": 360,
        "standard_name": "sea_surface_wave_directional_spread",
    },
    "peak_directional_spread": {
        "units": "degree",
        "long_name": "Wave directional spread",
        "missing_value": np.NAN,
        "valid_min": 0,
        "valid_max": 360,
        "standard_name": "sea_surface_wave_directional_spread",
    },
    "cdf": {
        "units": "m^2",
        "long_name": "Cumulative distribution function",
        "missing_value": np.NAN,
        "valid_min": 0,
    },
    "peak_wavenumber": {
        "units": "rad/m",
        "long_name": "Peak wave wavenumber",
        "missing_value": np.NAN,
        "valid_min": 0,
    },
}


def get_conventions(variable):
    """
    Get CF conventions for a variable.
    :param variable: may be a list or a string. If a list we assume the attributes are found in a nested dictionary.
        where the first element of the list is the top level key, the second element is the second level key, etc. Note
        that we fail _silently_ if the key is not found.
    :return: dictionary of attributes
    """

    def _get_var(_dict, _key_list):
        """
        Get a variable from a nested dictionary.
        :param _dict: dictionary
        :param _key_list: key list
        :return: dictionary of attributes
        """
        if len(_key_list) == 0:
            if "__default__" in _dict:
                key = _dict["__default__"]
                return _get_var(_dict[key], [])

            return _dict
        else:
            if _key_list[0] not in _dict:
                raise KeyError(f"Key {_key_list[0]} not found in dictionary.")
            else:
                return _get_var(_dict[_key_list[0]], _key_list[1:])

    if variable[0] not in cf_conventions:
        return None
    else:
        return _get_var(cf_conventions, variable)


def set_conventions(
    data_array: DataArray, variable, overwrite=False, mutate=True
) -> DataArray:
    """
    Set CF conventions for a variable in a DataArray.
    :param data_array: data array
    :param variable: variable name
    :param overwrite: Overwrite existing attributes
    :param mutate: Mutate the data array.
    :return: return the data array. A copy is returned if mutate=False.
    """
    if mutate:
        da = data_array
    else:
        da = data_array.copy(deep=False)

    if isinstance(variable, str):
        variable = [variable]

    variable_name = variable[0]
    data_array.name = variable_name

    attributes = get_conventions(variable)
    if attributes is None:
        return da

    for attribute in attributes:
        target = attributes[attribute]
        if overwrite or attribute not in da.attrs:
            da.attrs[attribute] = target
        else:
            if da.attrs[attribute] != target:
                pass
                # warnings.warn(f'Attribute {attribute} of {variable_name} is set to {da.attrs[attribute]} '
                #               f'which differs from assumed {target}. Set overwrite=True to '
                #               f'overwrite.')

    return da


def make_ordinal(n):
    """
    from: https://stackoverflow.com/questions/9647202/ordinal-numbers-replacement

    Convert an integer into its ordinal representation::

        make_ordinal(0)   => '0th'
        make_ordinal(3)   => '3rd'
        make_ordinal(122) => '122nd'
        make_ordinal(213) => '213th'
    """
    n = int(n)
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = ["th", "st", "nd", "rd", "th"][min(n % 10, 4)]
    return str(n) + suffix
