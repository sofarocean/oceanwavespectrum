from ._spotter_sdcard_data import read_spectral_csv as _read_spectral_csv
from ._spotter_wavefleet_data import fetch_spectrum as _fetch_spectrum

fetch_spectrum = _fetch_spectrum
read_spectral_csv = _read_spectral_csv
