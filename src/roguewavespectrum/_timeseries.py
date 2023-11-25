"""
Private module for now - may expose eventually - but it needs work, and I feel this may better be handled in some
dependant package. Basic idea is to provide some simple tooling t provide time series realizations from spectra.
This is useful for testing and for generating synthetic data. May one day include second-order stuff we did in the
nonlinear kinematics paper.
"""

from roguewavespectrum import Spectrum
from numpy.random import default_rng
from numpy.fft import irfft
from numpy import sqrt, cos, sin, pi, exp, sum, linspace
from numpy.typing import NDArray
from typing import Literal, Tuple


def surface_timeseries(
    component: Literal["u", "v", "w", "x", "y", "z"],
    sampling_frequency: float,
    signal_length: int,
    spectrum: Spectrum,
    seed: int = None,
) -> Tuple[NDArray, NDArray]:
    """
    Create a timeseries for from a given power spectral density.

    :param component: Wave component to create a timeseries for: u,v,w,x,y,z.
    :param sampling_frequency: Sampling frequency of output signal in Hertz
    :param signal_length: Length of output signal
    :param spectrum: Input power spectrum
    :param seed: Input seed for the random number generator.
    :return:
    """

    nfft = (int(signal_length) // 2) * 2

    frequencies = linspace(0, 0.5 * sampling_frequency, nfft // 2, endpoint=False)

    time = linspace(0, nfft / sampling_frequency, nfft, endpoint=False)

    timeseries = nfft * irfft(
        create_fourier_amplitudes(component, spectrum, frequencies, seed)
    )

    return time, timeseries


def create_fourier_amplitudes(component, spectrum: Spectrum, frequencies, seed=None):
    spectrum = spectrum.interpolate_frequency(frequencies)

    if not spectrum.is_2d:
        radian_directions = 0.0
        radian_frequency = spectrum.angular_frequency.values
        area = spectrum.frequency_binwidth.values

    elif spectrum.is_2d:
        radian_directions = spectrum.radian_direction_mathematical.values[None, :]
        radian_frequency = spectrum.angular_frequency.values[:, None]
        area = (
            spectrum.frequency_binwidth.values[:, None]
            * spectrum.direction_binwidth.values[None, :]
        )
    else:
        raise ValueError("Not a spectrum")

    if component == "w":
        factor = 1j * radian_frequency

    elif component == "u":
        factor = radian_frequency * cos(radian_directions)

    elif component == "v":
        factor = radian_frequency * sin(radian_directions)

    elif component == "x":
        factor = -1j * cos(radian_directions)

    elif component == "y":
        factor = -1j * sin(radian_directions)

    else:
        factor = 1.0

    phases = default_rng(seed=seed).uniform(0, 2 * pi, spectrum.spectral_shape)
    amplitudes = sqrt(area * spectrum.values / 2) * exp(1j * phases) * factor

    if spectrum.is_2d:
        amplitudes = sum(amplitudes, axis=-1)

    return amplitudes
