import numpy as np
from numba import njit
from xarray import DataArray, Dataset, concat, zeros_like, ones_like
from ._spotter_time_integration import integrated_response_factor_spectral_tail
from roguewavespectrum._variable_names import SPECTRAL_VARS
from roguewavespectrum._spectrum import Spectrum


@njit(cache=True)
def _fit(x, y, power):
    X = x**power
    coef = np.sum(X * y) / np.sum(X * X)
    goodness_of_fit = np.sum((coef * X - y) ** 2)
    return coef, goodness_of_fit


@njit(cache=True)
def tail_fit(x, y, power):
    if power is None:
        coef_4, goodness_of_fit_4 = _fit(x, y, -4)
        coef_5, goodness_of_fit_5 = _fit(x, y, -5)
        if goodness_of_fit_5 < goodness_of_fit_4:
            return coef_5, -5
        else:
            return coef_4, -4

    else:
        coef, goodness_of_fit = _fit(x, y, power)
        return coef, power


@njit(cache=True)
def transition_frequency(
    energy_last_resoved_frequency, last_resolved_frequency, integrated_tail
):
    """
    Estimate the transition frequency from a -4 to -5 tail given the last resolved frequency and the integrated energy
    in the tail.

    :param energy_last_resoved_frequency:
    :param last_resolved_frequency:
    :param integrated_tail:
    :return:
    """
    return (
        (energy_last_resoved_frequency * last_resolved_frequency**4)
        / (
            4 * energy_last_resoved_frequency * energy_last_resoved_frequency
            - 12 * integrated_tail
        )
    ) ** (1 / 3)


@njit(cache=True)
def numba_fill_zeros_or_nan_in_tail(
    variance_density: np.ndarray,
    frequencies: np.ndarray,
    power=None,
    zero=0.0,
    points_in_fit=10,
    tail_information=None,
):
    input_shape = variance_density.shape
    number_of_elements = 1
    for value in input_shape[:-1]:
        number_of_elements *= value

    number_of_frequencies = input_shape[-1]
    variance_density = variance_density.reshape((number_of_elements, input_shape[-1]))

    if tail_information is None:
        tail_energy = np.zeros(number_of_elements)
        tail_bounds = (frequencies[-1] + 1, frequencies[-1] + 2)
    else:
        tail_energy = tail_information[1]
        tail_bounds = tail_information[0]

    for ii in range(0, number_of_elements):

        for jj in range(number_of_frequencies - 1, -1, -1):
            if variance_density[ii, jj] > zero:
                # Note, for nan values this is also always false. No need to seperately check for that.
                index = jj
                break
        else:
            # no valid value found, we cannot extrapolate. Technically this is not needed as we catch this below as
            # well (since index=number_of_frequencies by default). But to make more explicit that we catch for this
            # scenario I leave it in.
            continue

        if index == number_of_frequencies - 1:
            continue
        elif index < points_in_fit:
            continue

        last_resolved_frequency = frequencies[index]
        last_resolved_energy = variance_density[ii, index]

        if tail_energy[ii] > 0.0:
            trans_freq, starting_energy = _compound_tail(
                last_resolved_frequency,
                last_resolved_energy,
                tail_energy[ii],
                tail_bounds,
            )

            for jj in range(index + 1, number_of_frequencies):
                if frequencies[jj] >= trans_freq:
                    variance_density[ii, jj] = (
                        starting_energy * trans_freq * frequencies[jj] ** -5
                    )
                else:
                    variance_density[ii, jj] = starting_energy * (frequencies[jj]) ** -4

        else:
            coef, fitted_power = tail_fit(
                x=frequencies[index - points_in_fit + 1 : index + 1],
                y=variance_density[ii, index - points_in_fit + 1 : index + 1],
                power=power,
            )

            for jj in range(index + 1, number_of_frequencies):
                variance_density[ii, jj] = coef * (frequencies[jj]) ** fitted_power

    return np.reshape(variance_density, input_shape)


@njit(cache=True)
def _compound_tail(
    last_resolved_frequency, last_resolved_energy, raw_tail_energy, tail_bounds
):
    transition_frequencies = np.linspace(tail_bounds[0], tail_bounds[1], 11)

    goodness_of_fit = 0.0
    transition_frequency = transition_frequencies[0]
    starting_energy = 0.0
    for index in range(transition_frequencies.shape[0]):
        tail_energy = raw_tail_energy * integrated_response_factor_spectral_tail(
            -4,
            tail_bounds[0],
            tail_bounds[1],
            2.5,
            transition_frequency=transition_frequencies[index],
        )
        freq_int = 1.0 / 3.0 * (
            tail_bounds[0] ** -3 - transition_frequencies[index] ** -3
        ) + 1.0 / 4.0 * transition_frequencies[index] * (
            transition_frequencies[index] ** -4 - tail_bounds[1] ** -4
        )
        current_starting_energy = tail_energy / freq_int
        current_fit = (
            current_starting_energy * last_resolved_frequency**-4
            - last_resolved_energy
        ) ** 2

        if index == 0:
            goodness_of_fit = (
                current_starting_energy * last_resolved_frequency**-5
                - last_resolved_energy
            ) ** 2
            starting_energy = current_starting_energy
            transition_frequency = transition_frequencies[index]

        else:
            if current_fit < goodness_of_fit:
                goodness_of_fit = current_fit
                starting_energy = current_starting_energy
                transition_frequency = transition_frequencies[index]

    return transition_frequency, starting_energy


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
        (e, e.isel(frequency=-1) * zeros_like(tail_frequency)), dim="frequency"
    )

    tail_a1 = a1.isel(frequency=-1) if tail_moments is None else tail_moments["a1"]
    tail_b1 = b1.isel(frequency=-1) if tail_moments is None else tail_moments["b1"]
    tail_a2 = a2.isel(frequency=-1) if tail_moments is None else tail_moments["a2"]
    tail_b2 = b2.isel(frequency=-1) if tail_moments is None else tail_moments["b2"]

    a1 = concat((a1, tail_a1 * ones_like(tail_frequency)), dim="frequency")
    b1 = concat((b1, tail_b1 * ones_like(tail_frequency)), dim="frequency")
    a2 = concat((a2, tail_a2 * ones_like(tail_frequency)), dim="frequency")
    b2 = concat((b2, tail_b2 * ones_like(tail_frequency)), dim="frequency")

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
