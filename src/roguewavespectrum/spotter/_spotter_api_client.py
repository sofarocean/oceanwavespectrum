import requests
import pandas as pd
from requests.adapters import HTTPAdapter, Retry
from pysofar.sofar import SofarApi
from pysofar.spotter import Spotter
from datetime import datetime
from typing import Literal
import numpy as np

session = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))

DATA_TYPES = Literal[
    "waves",
    "wind",
    "surfaceTemp",
    "barometerData",
    "frequencyData",
    "microphoneData",
    "smartMooringData",
]
MAX_LOCAL_LIMIT = 100
MAX_LOCAL_LIMIT_BULK = 500
NUMBER_OF_RETRIES = 2
MAX_DAYS_SMARTMOORING = 10


class SpectralApi:
    """ """

    def __init__(self, token=None):
        self._session = SofarApi(custom_token=token)

    def get_frequency_data(self):
        self.get_frequency_data()


def _download_data(
    var_name: str,
    start_date: datetime,
    end_date: datetime,
    spotter_id: str,
    session: SofarApi,
    **kwargs,
):
    """ """
    data = list(_unpaginate(var_name, start_date, end_date, spotter_id, session))

    return data


def _unpaginate(
    var_name: str, start_date: datetime, end_date: datetime, spotter: Spotter
):
    """
    Generator function to unpaginate data from the api.

    idiosyncrasies to handle:
    - Wavefleet sometimes returns multiple instances of the same record (same
      time). These are filtered through a all to _unique. This should
      be fixed in the future.
    - Some entries are broken (None for entries).
    - Not all Spotters will have (all) data for the given timerange.
    - Wavefleet sometimes times out on requests.

    :param spotter_id: Spotter id

    :param start_date: ISO 8601 formatted date string, epoch or datetime.
                       If not included defaults to beginning of spotters
                       history

    :param end_date: ISO 8601 formatted date string, epoch or datetime.
                     If not included defaults to end of spotter history

    :return: Data
    """
    while True:
        json_data = None
        for retry in range(0, NUMBER_OF_RETRIES + 1):
            try:
                json_data = spotter.grab_data(
                    limit=MAX_LOCAL_LIMIT,
                    start_date=start_date.isoformat(),
                    end_date=end_date.isoformat(),
                    include_frequency_data=var_name == "frequencyData",
                    include_directional_moments=var_name == "frequencyData",
                    include_barometer_data=var_name == "barometerData",
                    include_waves=var_name == "waves",
                    include_wind=var_name == "wind",
                    include_surface_temp_data=var_name == "surfaceTemp",
                    include_microphone_data=var_name == "microphoneData",
                )
            except Exception as e:
                if retry > NUMBER_OF_RETRIES:
                    raise e

        # Is the variable returned
        if var_name not in json_data:
            raise ExceptionNoDataForVariable(var_name)

        data = json_data[var_name]

        # Filter for Nones in latitude/longitude (bad data in api), and data prior to the requested
        # start_date
        data = list(
            filter(
                lambda x: (x["latitude"] is not None)
                and (x["longitude"] is not None)
                and (x["timestamp"] is not None)
                and (pd.Timestamp(x["timestamp"]) >= start_date),
                data,
            )
        )

        # and if so- did it contain any data?
        if not data:
            break

        for entry in data:
            yield entry

        start_date = pd.Timestamp(
            pd.Timestamp(data[-1]["timestamp"]) + np.timedelta64(1, "s")
        )


class ExceptionCouldNotDownloadData(Exception):
    """
    Query raised when no frequency data is available for the spotter
    """

    pass


class ExceptionNoDataForVariable(Exception):
    """
    Query raised when no frequency data is available for the spotter
    """

    pass
