"""
Contents: Routines to get data from the spotter api

Copyright (C) 2022
Sofar Ocean Technologies

Authors: Pieter Bart Smit
======================

Routines to get data from the spotter api

Functions:

- `get_spectrum`, function to download spectral data.
- `get_bulk_wave_data`, function to download bulk wave data.
- `get_data`, general function to download different data types (spectral,
    bulk wave, wind, SST, barometer).
- `search_circle`, get all available data within a given circle.
- `search_rectangle`, get all available data within a given rectangle.

"""

# 1) Imports
# =============================================================================
from datetime import datetime, timedelta
from pysofar.spotter import Spotter, SofarApi
from pysofar.wavefleet_exceptions import QueryError
from roguewavespectrum._spectrum import BuoySpectrum
from typing import Dict, List, Union, Sequence, Literal
import xarray as xr
import numpy as np
import pandas as pd

# 2) Constants & Private Variables
# =============================================================================
# Maximum number of spectra to retrieve from the Spotter API per API call. Note
# that 2- os a hard limit of the API. If set higher than 100 it will just
# return 100 (and the implementation will fail)
MAX_LOCAL_LIMIT = 100
MAX_LOCAL_LIMIT_BULK = 500

# Number of retry attemps if a download fails.
NUMBER_OF_RETRIES = 2

MAX_DAYS_SMARTMOORING = 10

DATA_TYPES = Literal[
    "waves",
    "wind",
    "surfaceTemp",
    "barometerData",
    "frequencyData",
    "microphoneData",
    "smartMooringData",
]


def fetch_spectrum(
    spotter_ids: Union[str, Sequence[str]],
    start_date: Union[datetime, int, float, str] = None,
    end_date: Union[datetime, int, float, str] = None,
    **kwargs,
) -> xr.Dataset:
    """
    Gets the requested frequency wave data for the spotter(s) in the given
    interval.

    :param spotter_ids: Can be either 1) a List of spotter_ids or 2) a single
    Spotter_id.

    :param start_date: ISO 8601 formatted date string, epoch or datetime.
                       If not included defaults to beginning of spotters
                       history

    :param end_date:   ISO 8601 formatted date string, epoch or datetime.
                       If not included defaults to end of spotter history

    :return: Data as a dictornary with spotter_id's as keys, and for each
    corresponding value a List that for each returned time contains a
    WaveSpectrum1D object.

    """
    return fetch_bulkdata(spotter_ids, "frequencyData", start_date, end_date, **kwargs)


def fetch_bulkdata(
    spotter_ids: Union[str, Sequence[str]],
    data_type: DATA_TYPES,
    start_date: Union[datetime, int, float, str] = None,
    end_date: Union[datetime, int, float, str] = None,
    session: SofarApi = None,
) -> xr.Dataset:
    """
    Gets the requested data for the spotter(s) in the given interval as either a dataframe containing
    all the data for the combined spotters in a single table (all datatypes except frequencyData) or
    a dictionary object that has the spotter_id as key and contains a frequency spectrum object as
    values.

    :param spotter_ids: Can be either 1) a List of spotter_ids or 2) a single
    Spotter_id.

    :param data_type: Literal string denoting the desired data type, options are
            data_type="waves", bulk wave data
            data_type="wind", wind estimates
            data_type="surfaceTemp", surface temperature (if available)
            data_type="barometerData", barometer data (if available)
            data_type="frequencyData", frequency data (if available) NOTE: does not return a datafrae
            data_type="microphoneData", microphone data if available
            data_type="smartMooringData", smartmooring data if available.

    :param start_date: ISO 8601 formatted date string, epoch or datetime.
                       If not included defaults to beginning of spotters
                       history

    :param end_date:   ISO 8601 formatted date string, epoch or datetime.
                       If not included defaults to end of spotter history

    :param session:    Active SofarApi session. If none is provided one will be
                       created automatically. This requires that an API key is
                       set in the environment.

    :param parallel_download: Use multiple requests to the Api to speed up
                       retrieving data. Only useful for large requests.

    :param cache: Cache requests. If True, returned data will be stored in
                        a file Cache on disk, and repeated calls with the
                        same arguments will use locally cached data. The cache
                        is a FileCache with a maximum of 2GB by default.

    :return:
        data_type="frequencyData": a dictionary with spoter ids as keys and FrequencySpectra as values
        data_type= ...  : a Pandas Dataframe with a spotter_id column that indicates to which spotter entries
            belong.
    """

    if session is None:
        session = _get_sofar_api()

    if spotter_ids is None:
        spotter_ids = fetch_spotter_ids(sofar_api=session)

    # Make sure we have a list object
    if not isinstance(spotter_ids, (list, tuple)):
        spotter_ids = [spotter_ids]

    # Ensure we have datetime aware objects
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    data_for_variable = []
    for spotter_id in spotter_ids:
        data_for_variable.append(
            _download_data(
                var_name=data_type,
                start_date=start_date,
                end_date=end_date,
                spotter_id=spotter_id,
                session=session,
            )
        )

    values = []
    for spotter_id, spotter_data in zip(spotter_ids, data_for_variable):
        #
        # Did we get any data for this spotter
        if spotter_data is not None:
            values += [spotter_data]

    data = xr.concat(values, dim="time")
    data = data.drop_indexes("time")
    data = data.reset_coords("time")
    data = data.rename_dims({"time": "index"})
    return data


def _download_data(
    var_name: str,
    start_date: datetime,
    end_date: datetime,
    spotter_id: str,
    session: SofarApi,
    **kwargs,
) -> xr.Dataset:
    """ """
    data = list(_unpaginate(var_name, start_date, end_date, spotter_id, session))

    # Postprocessing
    if len(data) < 1:
        data = None
    else:
        data = xr.concat(data, dim="time")
        data.drop_duplicates("time")
        data.sortby("time")

    return data


def _get_smart_mooring_data(
    start_date: str, end_date: str, spotter: Spotter, max_days=None
):
    # We have a self imposed limit that we return to avoid overloading wavefleet.
    if max_days is not None:
        start_date_dt = pd.Timestamp(start_date)
        end_date_dt = pd.Timestamp(end_date)

        # If the enddate exceeds the maximum number of days to download we reduce the enddate
        if end_date_dt - start_date_dt > timedelta(days=max_days):
            end_date_dt = start_date_dt + timedelta(days=max_days)

            # update the enddate
            end_date = end_date_dt.isoformat()

    params = {"spotterId": spotter.id, "startDate": start_date, "endDate": end_date}
    scode, data = spotter._session._get("sensor-data", params=params)

    if scode != 200:
        raise QueryError(data["message"])

    return data["data"]


def _unpaginate(
    var_name: str,
    start_date: datetime,
    end_date: datetime,
    spotter_id: str,
    session: SofarApi,
) -> xr.Dataset:
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
    spotter = Spotter(spotter_id, spotter_id, session=session)
    page = 0
    while True:
        json_data = None
        page += 1
        for retry in range(0, NUMBER_OF_RETRIES + 1):
            try:
                if var_name == "smartMooringData":
                    json_data = _get_smart_mooring_data(
                        start_date=start_date.isoformat(),
                        end_date=end_date.isoformat(),
                        spotter=spotter,
                        max_days=MAX_DAYS_SMARTMOORING,
                    )
                    json_data = {"smartMooringData": json_data}
                else:
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
                break
            except Exception as e:
                if retry < NUMBER_OF_RETRIES:
                    warning = (
                        f"Error downloading data for {spotter.id}, "
                        f"attempting retry {retry + 1}"
                    )
                    print(warning)
                else:
                    raise e

        #
        # If so- was it returned? If not- raise error
        if var_name not in json_data:
            raise ExceptionNoDataForVariable(var_name)

        # If no data - return
        if not json_data[var_name]:
            break

        # Filter for Nones.
        json_data[var_name] = _none_filter(json_data[var_name])
        if not json_data[var_name]:
            break

        objects = []
        for _object in [
            _get_class(var_name, data, spotter_id) for data in json_data[var_name]
        ]:
            date = _object.time.values

            if date < start_date:
                continue
            else:
                objects.append(_object)

        if len(objects) == 0:
            break

        last_object = None
        for _object in objects:
            last_object = _object
            yield _object

        start_date = pd.Timestamp(last_object.time.values + np.timedelta64(1, "s"))


def _get_class(key, data, spotter_id) -> xr.Dataset:
    if key == "frequencyData":
        dataset = xr.Dataset(
            data_vars={
                "variance_density": (
                    ("frequency",),
                    np.array(data["varianceDensity"]),
                ),
                "a1": (
                    ("frequency",),
                    np.array(data["a1"]),
                ),
                "b1": (
                    ("frequency",),
                    np.array(data["b1"]),
                ),
                "a2": (
                    ("frequency",),
                    np.array(data["a2"]),
                ),
                "b2": (
                    ("frequency",),
                    np.array(data["b2"]),
                ),
                "latitude": (
                    (),
                    np.array(data["latitude"]),
                ),
                "longitude": (
                    (),
                    np.array(data["longitude"]),
                ),
                "ids": ((), spotter_id),
            },
            coords={
                "time": pd.Timestamp(data["timestamp"]),
                "frequency": np.array(data["frequency"]),
            },
        )

    else:
        data_vars = {}
        for var_key, value in data.items():
            # If the name differs get the correct name, otherwise use the
            # name returned from the API
            if var_key == "timestamp":
                continue

            data_vars[var_key] = (
                (),
                np.array(value),
            )
            data_vars["ids"] = (
                (),
                spotter_id,
            )

        dataset = xr.Dataset(
            data_vars=data_vars,
            coords={
                "time": pd.Timestamp(data["timestamp"]),
            },
        )
    return dataset


# Helper functions
def fetch_spotter_ids(sofar_api: SofarApi = None) -> List[str]:
    """
    Get a list of Spotter ID's that are available through this account.

    :param sofar_api: valid SofarApi instance.
    :return: List of spotters available through this account.
    """
    if sofar_api is None:
        sofar_api = _get_sofar_api()
    return sofar_api.device_ids


# SofarAPI instance.
_API = None


def _get_sofar_api(token=None) -> SofarApi:
    """
    Gets a new sofar API object if requested. Returned object is essentially a
    Singleton class-> next calls will return the stored object instead of
    creating a new class. For module internal use only.

    :return: instantiated SofarApi object
    """
    global _API
    if _API is None:
        _API = SofarApi(custom_token=token)
    return _API


def _none_filter(data: Dict):
    """
    Filter for the occasional occurance of bad data returned from wavefleet.
    :param data:
    :return:
    """
    return list(
        filter(
            lambda x: (x["latitude"] is not None)
            and (x["longitude"] is not None)
            and (x["timestamp"] is not None),
            data,
        )
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


class SpectralApiClient:
    def __iter__(self, sofar_api: SofarApi = None):
        self.sofar_api = sofar_api if sofar_api else _get_sofar_api()

    def spotter_ids(self) -> List[str]:
        """
        Get a list of Spotter ID's that are available through this account.

        :param sofar_api: valid SofarApi instance.
        :return: List of spotters available through this account.
        """
        return self.sofar_api.device_ids

    def fetch_spectrum(
        self,
        spotter_ids: Union[str, Sequence[str]],
        start_date: Union[pd.Timestamp, datetime, int, float, str] = None,
        end_date: Union[pd.Timestamp, datetime, int, float, str] = None,
        **kwargs,
    ) -> BuoySpectrum:

        dataset = fetch_spectrum(
            spotter_ids, start_date, end_date, session=self.sofar_api, **kwargs
        )
        spectrum = BuoySpectrum(dataset)

        return spectrum
