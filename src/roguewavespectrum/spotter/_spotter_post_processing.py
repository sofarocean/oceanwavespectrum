from roguewavespectrum._spectrum import Spectrum
from roguewavespectrum._operations import multiply
from roguewavespectrum.spotter._spotter_extrapolate_tail import (
    extrapolate_tail,
    fill_zeros_or_nan_in_tail,
)
from roguewavespectrum.spotter._spotter_time_integration import complex_response
import numpy as np

LAST_BIN_WIDTH = 0.3
LAST_BIN_FREQUENCY_START = 0.5
LAST_BIN_FREQUENCY_END = 0.8
SPOTTER_FREQUENCY_RESOLUTION = 2.5 / 256


def post_process_api_spectrum(spectrum: Spectrum, **kwargs) -> Spectrum:
    """
    Spectra from the API may contain all 128 frequencies sampled at 2.5Hz, or only a reduced set of frequencies, where
    the last bin represents all energy from 0.5 to 0.8Hz. This function will:

    - ensure the spectrum is defined on a regular grid from 3*df to 81*df, with df representing the frequency
    resolution.
    - extrapolate the tail using the known energy in the last bin if the tail is missing (only 39 frequencies).
    - we correct for underflow in the tail.
    - we correct for integration errors in the tail.

    :param spectrum: input spectra.
    :param maximum_frequency: maximum frequency to extrapolate to.
    :return:
    """
    maximum_frequency = kwargs.get("maximum_frequency", LAST_BIN_FREQUENCY_END)
    maximum_index = int(maximum_frequency / SPOTTER_FREQUENCY_RESOLUTION)
    new_frequencies = (
        np.linspace(3, maximum_index, maximum_index - 2, endpoint=True)
        * SPOTTER_FREQUENCY_RESOLUTION
    )

    if len(spectrum.frequency) == 39:
        new_frequencies[0] = spectrum.frequency[0]

        # Get the energies and moments in the last bin
        last_bin_energy = spectrum.variance_density.values[..., -1] * LAST_BIN_WIDTH
        last_bin_moments = {
            "a1": spectrum.a1[..., -1],
            "b1": spectrum.b1[..., -1],
            "a2": spectrum.a2[..., -1],
            "b2": spectrum.b2[..., -1],
        }

        # interpolat the spectrum to the regular grid
        spectrum = spectrum.interpolate_frequency(new_frequencies)

        # chop off anything related to the last bin
        spectrum = spectrum.bandpass(
            fmax=LAST_BIN_FREQUENCY_START, interpolate_to_limits=False
        )

        # Correct for integration errors in the tail
        spectrum = spotter_frequency_response_correction(spectrum)

        # Extrapolate tail given the known energy in last bin. Also correct for potential underflow in the tail
        spectrum = extrapolate_tail(
            spectrum,
            maximum_frequency,
            tail_energy=last_bin_energy,
            tail_bounds=(LAST_BIN_FREQUENCY_START, LAST_BIN_FREQUENCY_END),
            tail_moments=last_bin_moments,
            tail_frequency=new_frequencies[new_frequencies > LAST_BIN_FREQUENCY_START],
        )
    else:
        # Chop to desired max freq
        spectrum = spectrum.bandpass(fmax=maximum_frequency)

        # Correct for integration errors in the tail
        spectrum = spotter_frequency_response_correction(spectrum)

        # Correct for potential underflow in the tail
        spectrum = fill_zeros_or_nan_in_tail(spectrum)

        # ensure we are on the regular grid
        spectrum = spectrum.interpolate_frequency(new_frequencies)

    return spectrum


def spotter_frequency_response_correction(
    spectrum: Spectrum, order=4, n=1, sampling_frequency=2.5
) -> Spectrum:
    """
    Correct for the spectral dampening/amplification caused by numerical integration of velocities.
    :param spectrum:
    :param order:
    :param n:
    :return:
    """
    amplification_factor = complex_response(
        spectrum.frequency.values / sampling_frequency, order, n
    )
    R = np.real(amplification_factor * np.conjugate(amplification_factor))
    return multiply(spectrum, 1 / R, ["frequency"])
