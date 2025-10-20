# Get directory of current file
import os
import pandas as pd
import roguewavespectrum
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
SPECTRUM = os.path.join(
    current_dir, "test_data", "wind_estimates", "spotter_spectrum.nc"
)
OUTPUT = os.path.join(current_dir, "test_data", "wind_estimates", "spotter_u10.csv")


def _get_target_data() -> pd.DataFrame:
    return pd.read_csv(OUTPUT, index_col=0, parse_dates=True)


def _get_spectrum() -> roguewavespectrum.Spectrum:
    return roguewavespectrum.Spectrum.from_netcdf(SPECTRUM)


def test_spotter_wind_estimate():
    spectrum = _get_spectrum()
    target = _get_target_data()

    u10 = spectrum.estimate_wind_speed_at_10_meter().values
    dir = spectrum.estimate_wind_direction().values

    u10_target = target["u10"].values
    dir_target = target["direction"].values

    np.testing.assert_allclose(u10, u10_target, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(dir, dir_target, rtol=1e-2, atol=1e-2)
