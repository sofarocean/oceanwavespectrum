"""
Contents: Wind Estimator

Copyright (C) 2022
Sofar Ocean Technologies

Authors: Pieter Bart Smit
"""
import typing
import numpy
from typing import Literal
from xarray import DataArray
from ._physical_constants import PhysicsOptions

_direction_convention = Literal[
    "coming_from_clockwise_north", "going_to_counter_clockwise_east"
]


def friction_velocity(
    frequency: DataArray,
    variance_density: DataArray,
    a1: DataArray,
    b1: DataArray,
    physics_options: PhysicsOptions,
    power=4,
    directional_spreading_constant=2.5,
    beta=0.012,
) -> DataArray:

    e_peak, _, _ = equilibrium_range_values(
        frequency,
        variance_density,
        a1,
        b1,
        power=power,
    )
    emean = 8.0 * numpy.pi**3 * e_peak

    # Get friction velocity from spectrum
    return emean / physics_options.gravity / directional_spreading_constant / beta / 4


def estimate_wind_speed_from_wave_spectrum(
    frequency: DataArray,
    variance_density: DataArray,
    a1: DataArray,
    b1: DataArray,
    height_meter=10,
    power=4,
    directional_spreading_constant=2.5,
    phillips_constant_beta=0.0133,
    physics_options: PhysicsOptions = None,
    **kwargs,
) -> DataArray:
    #
    # =========================================================================
    # Required Input
    # =========================================================================
    #
    # f              :: frequencies (in Hz)
    # E              :: Variance densities (in m^2 / Hz )
    #
    # =========================================================================
    # Output
    # =========================================================================
    #
    # U10            :: in m/s
    # Direction      :: in degrees clockwise from North (where wind is *coming from)
    #
    # =========================================================================
    # Named Keywords (parameters to inversion algorithm)
    # =========================================================================
    # Npower = 4     :: exponent for the fitted f^-N spectral tail
    # I      = 2.5   :: Philips Directional Constant
    # beta   = 0.012 :: Equilibrium Constant
    # Kapppa = 0.4   :: Von Karman constant
    # Alpha  = 0.012 :: Constant in estimating z0 from u* in Charnock relation
    # grav   = 9.81  :: Gravitational acceleration
    #
    # =========================================================================
    # Algorithm
    # =========================================================================
    #
    # 1) Find the part of the spectrum that best fits a f^-4 shape
    # 2) Estimate the Phillips equilibrium level "Emean" over that range
    # 3) Use Emean to estimate Wind speed (using Charnock and LogLaw)
    # 4) Calculate mean direction over equilibrium range

    # Get friction velocity from spectrum
    ustar = friction_velocity(
        frequency,
        variance_density,
        a1,
        b1,
        physics_options,
        power,
        directional_spreading_constant,
        phillips_constant_beta,
    )

    # Find z0 from Charnock Relation
    z0 = charnock_roughness_length(ustar, physics_options, **kwargs)
    windspeed_at_height = (
        ustar / physics_options.vonkarman_constant * numpy.log(height_meter / z0)
    )
    return windspeed_at_height


def estimate_wind_direction_from_wave_spectrum(
    frequency: DataArray,
    variance_density: DataArray,
    a1: DataArray,
    b1: DataArray,
    power=4,
    **kwargs,
) -> DataArray:
    #

    e_peak, a1_peak, b1_peak = equilibrium_range_values(
        frequency,
        variance_density,
        a1,
        b1,
        power=power,
    )

    # Estimate direction from tail
    direction = (180.0 / numpy.pi * numpy.arctan2(b1_peak, a1_peak)) % 360
    return direction


def equilibrium_range_values(
    frequency: DataArray,
    variance_density: DataArray,
    a1: DataArray,
    b1: DataArray,
    power=4,
) -> typing.Tuple[DataArray, DataArray, DataArray]:
    """
    :param spectrum:
    :param frequency:
    :param variance_density:
    :param a1:
    :param b1:
    :param power:
    :return:
    """

    scaled_spec = variance_density * frequency**power
    scaled_spec = scaled_spec.fillna(0)
    indices = scaled_spec.argmax(dim="frequency")
    a1_peak = a1.isel({"frequency": indices})
    b1_peak = b1.isel({"frequency": indices})
    e_peak = scaled_spec.isel({"frequency": indices})
    return e_peak, a1_peak, b1_peak


def charnock_roughness_length(
    friction_velocity: DataArray, physics_options: PhysicsOptions, **kwargs
) -> DataArray:

    if not isinstance(friction_velocity, DataArray):
        friction_velocity = DataArray(data=friction_velocity)

    return (
        physics_options.charnock_constant
        * friction_velocity**2
        / physics_options.gravity
    )
