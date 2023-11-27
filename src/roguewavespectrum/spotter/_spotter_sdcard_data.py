from roguewavespectrum._spectrum import Spectrum
from roguewavespectrum._factory_methods import create_spectrum1d
from datetime import datetime
import pandas as pd
import numpy as np
import os


def _read_spotter_spectral_file(file):
    if os.path.isfile(file):
        data = pd.read_csv(file).apply(pd.to_numeric, errors="coerce")
    else:
        raise FileNotFoundError(file)

    columns = list(data.columns)
    frequencies = np.array([float(x) for x in columns[8:]])
    values = data[columns[8:]].values

    time_tuple = data[columns[0:6]].values
    time = []
    for index in range(time_tuple.shape[0]):
        time.append(
            datetime(
                year=time_tuple[index, 0],
                month=time_tuple[index, 1],
                day=time_tuple[index, 2],
                hour=time_tuple[index, 3],
                minute=time_tuple[index, 4],
                second=time_tuple[index, 5],
            )
        )

    return {"time": np.array(time), "frequencies": frequencies, "values": values}


def read_spectral_csv(path: str, depth=np.inf, **kwargs) -> Spectrum:
    """
    Read parsed spotter spectral data from a given folder and return a Spectrum object.

    :param path: path to folder containing the parsed spotter spectral data. These files are created by the spotter
        sd-card parser and are typically named a1.csv, b1.csv, a2.csv, b2.csv, Szz.csv.
    :param depth: depth of the sensor (optional). This is used to estimate the wavenumber, wavespeed, groupspeed and
        other wave properties. If not provided, deep water conditions are assumed (when needed).
    :return: Spectrum object.
    """

    files = kwargs.get("files", None)
    if files is None:
        files = {
            "a1": "a1.csv",
            "b1": "b1.csv",
            "a2": "a2.csv",
            "b2": "b2.csv",
            "Szz": "Szz.csv",
        }

    data = {}
    for file_type, filename in files.items():
        file_location = os.path.join(path, filename)
        data[file_type] = _read_spotter_spectral_file(file_location)

    spectrum = create_spectrum1d(
        coordinates=[
            ("time", data["Szz"]["time"]),
            ("frequency", data["Szz"]["frequencies"]),
        ],
        variance_density=data["Szz"]["values"],
        a1=data["a1"]["values"],
        b1=data["b1"]["values"],
        a2=data["a2"]["values"],
        b2=data["b2"]["values"],
        depth=depth,
    )
    spectrum.fillna(0.0)
    return spectrum
