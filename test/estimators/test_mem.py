from roguewavespectrum.parametric import (
    parametric_directional_spectrum,
    DirCosineN,
    FreqPiersonMoskowitz,
)
import numpy
from roguewavespectrum import concatenate_spectra, Spectrum2D
from datetime import datetime, timezone
from numpy.testing import assert_allclose


def get_2d_spec() -> Spectrum2D:
    freq = numpy.linspace(0, 1, 20)
    dir = numpy.linspace(0, 360, 36, endpoint=False)
    time = datetime(2022, 1, 1, tzinfo=timezone.utc)

    frequency_shape = FreqPiersonMoskowitz(
        peak_frequency_hertz=0.1, significant_waveheight_meter=1
    )
    direction_shape = DirCosineN(mean_direction_degrees=35, width_degrees=20)
    spec = parametric_directional_spectrum(
        freq,
        dir,
        frequency_shape,
        direction_shape,
        time=time,
    )
    # spec = parametric_directional_spectrum(
    #     freq, frequency_shape="pm", peak_frequency_hertz=0.1, significant_waveheight_meter=1, direction_degrees=dir, direction_shape="raised_cosine",
    #     mean_direction_degrees=35, width_degrees=20, depth=numpy.inf, time=time
    # )
    return concatenate_spectra([spec], "time")


def test_mem():
    moments = ["a1", "b1", "a2", "b2"]

    spec2d = get_2d_spec()
    spec1d = spec2d.as_frequency_spectrum()
    reconstructed = spec1d.as_frequency_direction_spectrum(
        36, method="mem", solution_method="scipy"
    )

    for moment in moments:
        x = getattr(spec2d, moment)
        y = getattr(reconstructed, moment)
        assert_allclose(y, x, rtol=1e-2, atol=1e-2)
    pass


if __name__ == "__main__":
    test_mem()
