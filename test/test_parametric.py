from roguewavespectrum.parametric import (
    DirCosineN,
    FreqPiersonMoskowitz,
    FreqJonswap,
    parametric_directional_spectrum,
)
from numpy.testing import assert_allclose
from numpy import linspace, argmax, sum, trapezoid, isnan, any as np_any
import pytest


def test_raised_cosine():
    distribution = DirCosineN(mean_direction_degrees=20, width_degrees=28.64)
    assert_allclose(
        distribution.width_degrees_to_power(distribution.width_degrees),
        2,
        rtol=0.002,
        atol=0.003,
    )

    angles = linspace(0, 360, 36, endpoint=False)
    D = distribution.values(angles)
    assert argmax(D) == 2, f"maximum at {angles[argmax((D))]}, not at 20"
    assert_allclose(sum(D) * 10, 1, rtol=0.001, atol=0.001)


def test_pierson_moskowitz():
    distribution = FreqPiersonMoskowitz(
        peak_frequency_hertz=0.1, significant_waveheight_meter=4
    )
    frequency = linspace(0.01, 1, 100, endpoint=True)
    E = distribution.values(frequency)

    # Test that the peak is at the right location
    assert argmax(E) == 9, f"maximum at {frequency[argmax((E))]}, not at 0.1"

    # Test that it integrates to 1
    assert_allclose(trapezoid(E, frequency), 1, rtol=0.001, atol=0.001)


def test_create_parametric_spectrum():
    angles = linspace(0, 360, 36, endpoint=False)
    frequency = linspace(0.01, 1, 100, endpoint=True)
    dir_shape = DirCosineN(mean_direction_degrees=20, width_degrees=10)
    freq_shape = FreqPiersonMoskowitz(
        peak_frequency_hertz=0.1, significant_waveheight_meter=2
    )

    spectrum = parametric_directional_spectrum(frequency, angles, freq_shape, dir_shape)

    assert_allclose(spectrum.hm0().values, 2, rtol=0.001, atol=0.001)
    assert_allclose(spectrum.peak_frequency().values, 0.1, rtol=0.001, atol=0.001)
    assert_allclose(spectrum.mean_direction().values, 20, rtol=0.001, atol=0.001)

    # To note- the definition of spectral width is slightly different for the theoretical value used in the
    # shape. Hence the difference.
    assert_allclose(
        spectrum.mean_directional_spread().values, 10.116, rtol=0.001, atol=0.001
    )


def test_freq_shape_zero_waveheight_is_zero_not_nan():
    """A zero-waveheight frequency shape must renormalize to all-zero, not 0/0 NaN."""
    shape = FreqJonswap(peak_frequency_hertz=0.1, significant_waveheight_meter=0)
    frequency = linspace(0.01, 1, 100, endpoint=True)
    E = shape.values(frequency, renormalize=True)
    assert not np_any(isnan(E))
    assert_allclose(E, 0)


def test_freq_shape_raises_when_grid_does_not_resolve_peak():
    """A nonzero-waveheight shape whose peak falls entirely outside the sampled frequency grid
    must raise, not silently divide by a zero discretized integral (0/0 -> NaN)."""
    shape = FreqJonswap(peak_frequency_hertz=10.0, significant_waveheight_meter=1.0)
    frequency = linspace(0.01, 1, 100, endpoint=True)
    with pytest.raises(ValueError):
        shape.values(frequency, renormalize=True)


if __name__ == "__main__":
    test_raised_cosine()
    test_pierson_moskowitz()
    test_create_parametric_spectrum()
    test_freq_shape_zero_waveheight_is_zero_not_nan()
    test_freq_shape_raises_when_grid_does_not_resolve_peak()
