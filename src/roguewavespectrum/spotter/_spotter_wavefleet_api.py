import requests
import requests_cache
from typing import Union, Optional, Tuple, Literal, List
from functools import cached_property
import json
from requests.adapters import HTTPAdapter, Retry
from datetime import datetime
import numpy as np
import pandas as pd
from multiprocessing.pool import ThreadPool
from roguewavespectrum._spectrum import Spectrum, BuoySpectrum
import tqdm
import xarray as xr
from ._spotter_post_processing import post_process_api_spectrum

DATE_TYPE = Union[str, datetime, pd.Timestamp, int, float]


def _default_session(cached=False, **kwargs):
    requests_cache_path = kwargs.get(
        "requests_cache_path", "spotter_api_requests_cache.db"
    )
    if cached:
        session = requests_cache.CachedSession(
            cache_name=requests_cache_path, backend="sqlite"
        )
    else:
        session = requests.Session()

    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


class QueryError(Exception):
    pass


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


def _get(
    session: Union[requests.Session, requests_cache.CachedSession],
    url,
    header,
    params: Optional[dict] = None,
):
    if params is None:
        response = session.get(url, headers=header)
    else:
        response = session.get(url, headers=header, params=params)

    status = response.status_code
    message = json.loads(response.text)

    if status != 200:
        raise QueryError(f"{status}, Error retrieving data: {message}")

    return message


class SofarConnection:
    """
    Sofar API Client.

    :param custom_token: Custom token to use for authentication.
    """

    endpoint_devices = "https://api.sofarocean.com/api/devices"
    endpoint_historical_data = "https://api.sofarocean.com/api/wave-data"

    def __init__(
        self,
        token: str,
        session: Union[str, requests.Session, requests_cache.CachedSession] = "default",
        **kwargs,
    ):

        self._token = token

        if isinstance(session, (requests.Session, requests_cache.CachedSession)):
            self._session = session
        elif isinstance(session, str):

            if session == "default":
                cached = False
            elif session == "cached":
                cached = True
            else:
                raise ValueError(
                    "session must either be 'default' or 'cached', or must be a requests.Session "
                    "or requests_cache.CachedSession object"
                )

            self._session = _default_session(cached=cached, **kwargs)
        else:
            raise ValueError(
                "session must either be 'default' or 'cached', or must be a requests.Session "
                "or requests_cache.CachedSession object"
            )

    @property
    def _header(self):
        return {"token": self._token, "Content-Type": "application/json"}


class SofarSpotter:
    def __init__(self, spotter_id: str, connection: SofarConnection):
        self._connection = connection
        self.spotter_id = spotter_id

    def _query(
        self,
        start_date: DATE_TYPE,
        end_date: DATE_TYPE,
        var_name: Literal[
            "waves",
            "wind",
            "surfaceTemp",
            "barometerData",
            "frequencyData",
            "microphoneData",
            "smartMooringData",
        ],
        **kwargs,
    ):
        limit = kwargs.get("limit", 100)

        params = {
            "startDate": pd.Timestamp(start_date).isoformat(),
            "endDate": pd.Timestamp(end_date).isoformat(),
            "spotterId": self.spotter_id,
            "limit": limit,
            "includeFrequencyData": str(var_name == "frequencyData").lower(),
            "includeDirectionalMoments": str(var_name == "frequencyData").lower(),
            "includeBarometerData": str(var_name == "barometerData").lower(),
            "includeWaves": str(var_name == "waves").lower(),
            "includeWindData": str(var_name == "wind").lower(),
            "includeSurfaceTempData": str(var_name == "surfaceTemp").lower(),
            "includeMicrophoneData": str(var_name == "microphoneData").lower(),
            "includeSpikes": "false",
            "includeNonObs": "false",
            "processingSources": "all",
        }

        data = _get(
            self._connection._session,
            self._connection.endpoint_historical_data,
            self._connection._header,
            params=params,
        )
        return data

    def _paged_query(
        self,
        start_date: DATE_TYPE,
        end_date: DATE_TYPE,
        var_name: Literal[
            "waves",
            "wind",
            "surfaceTemp",
            "barometerData",
            "frequencyData",
            "microphoneData",
            "smartMooringData",
        ],
        **kwargs,
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
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)

        if start_date.tz is None:
            start_date = start_date.tz_localize("UTC")

        if end_date.tz is None:
            end_date = end_date.tz_localize("UTC")

        while True:
            json_data = self._query(start_date, end_date, var_name, **kwargs)

            if "data" not in json_data:
                raise ExceptionCouldNotDownloadData(json_data["message"])

            # Is the variable returned
            if var_name not in json_data["data"]:
                raise ExceptionNoDataForVariable(var_name)

            data = json_data["data"][var_name]

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

    def _query_spectrum(self, start_date: DATE_TYPE, end_date: DATE_TYPE, **kwargs):
        return list(
            self._paged_query(start_date, end_date, var_name="frequencyData", **kwargs)
        )

    def get_spectrum(
        self, start_date: DATE_TYPE, end_date: DATE_TYPE, **kwargs
    ) -> Spectrum:
        raw_spectra = self._query_spectrum(start_date, end_date, **kwargs)

        post_process = kwargs.get("post_process", True)
        if len(raw_spectra) == 0:
            raise ExceptionNoDataForVariable()

        return _to_spectrum(raw_spectra, post_process)


class SofarFleet:
    """
    Session object for querying the Sofar API. This object is used to query the api for spectral data for one or more
    devices. It requires an API token, which can be obtained from the Sofar Ocean website. The token is used to
    authenticate with the API.

    The primary methods of this class are:
    - get_spectra: Query the api for spectral data for the specified device_ids and return a BuoySpectrum object.
    - device_ids: Return a tuple of device ids (or Spotters) associated with the token/account.
    """

    def __init__(
        self,
        token: str,
        session: Union[str, requests.Session, requests_cache.CachedSession] = "default",
        requests_cache_path: str = "spotter_api_requests_cache.db",
    ):
        """
        Create a new Fleet object for querying the Sofar API for spectral data. It requires an API token, which can be
        obtained from the Sofar Ocean website. The token is used to authenticate with the API.

        For non-repeating queries, setting the session parameter to default is is sufficient. In that case a
        default requests.Session object will be created that will retry requests up to 5 times. However, if you request
        similar data multiple times, it is recommended to use a session with caching enabled. This will avoid querying
        the API multiple times for the same data. To use a session with caching enabled, set the session parameter to
        'cached'. This will create a session with caching enabled. By default, the cache will be stored in a file named
        'spotter_api_requests_cache.db' in the current working directory. To change the location of the cache file, set
        the requests_cache_path parameter to the desired location.

        :param token: API token. Will be used to authenticate with the API.
        :param session: If 'default', a default session will be created. If 'cached', a session with caching enabled.
        If a requests.Session or requests_cache.CachedSession object, that session will be used.
        :param requests_cache_path: Path to the cache file. Only used if session is 'cached'.


        """
        self._connection = SofarConnection(
            token, session, requests_cache_path=requests_cache_path
        )

    @cached_property
    def device_ids(self) -> Tuple[str, ...]:
        """
        Return a tuple of device ids associated with the session. This is a cached property to avoid querying the api
        multiple times. In the event devices are added dynamically while running a session, this property will not
        update. To update the property, create a new session object.

        :return: Tuple of device ids as strings.
        """
        devices = _get(
            self._connection._session,
            self._connection.endpoint_devices,
            self._connection._header,
        )
        return tuple(device["spotterId"] for device in devices["data"]["devices"])

    def _devices(
        self, device_ids: Optional[Union[List[str], Tuple[str, ...], str]] = None
    ) -> List[SofarSpotter]:
        """
        Return a list of SofarSpotter objects for the specified device_ids. If device_ids is None, all devices
        associated with the session will be returned.

        :param device_ids: Optional list of device ids to return. If None, all devices associated with the session.
        :return: List of SofarSpotter objects
        """

        if device_ids is None:
            device_ids = self.device_ids
        elif isinstance(device_ids, (list, tuple, str)):
            if isinstance(device_ids, str):
                device_ids = [device_ids]
        else:
            raise ValueError("When provided device_ids must be a list, tuple or str")

        return [SofarSpotter(_id, self._connection) for _id in device_ids]

    def get_spectra(
        self,
        start_date: DATE_TYPE,
        end_date: DATE_TYPE,
        device_ids: Optional[List[str]] = None,
        **kwargs,
    ) -> BuoySpectrum:

        """
        Query the api for spectral data for the specified device_ids and return a BuoySpectrum object.
        :param start_date: start date of interval. Any date format accepted by pd.Timestamp
        :param end_date: end date of interval. Any date format accepted by pd.Timestamp
        :param device_ids: Optional list of device ids to return. If None, all devices associated with the session.
        :param kwargs:
        :return:
        """

        parallel = kwargs.get("parallel", True)
        num_threads = kwargs.get("num_threads", 10)
        display_progress_bar = kwargs.get("display_progress_bar", True)
        post_process = kwargs.get("post_process", True)

        def _worker(spotter: SofarSpotter):
            try:
                data = spotter.get_spectrum(start_date, end_date, post_process=False)
            except ExceptionNoDataForVariable:
                data = None
            return data

        _input = self._devices(device_ids)

        if parallel:
            with ThreadPool(processes=num_threads) as pool:
                output = list(
                    tqdm.tqdm(
                        pool.imap(_worker, _input, chunksize=5),
                        desc="Downloading spectra for Spotters",
                        total=len(_input),
                        disable=not display_progress_bar,
                    )
                )
        else:
            output = list(
                tqdm.tqdm(
                    map(_worker, _input),
                    desc="Downloading spectra for Spotters",
                    total=len(_input),
                    disable=not display_progress_bar,
                )
            )

        data = {}
        for spotter, spectra in tqdm.tqdm(
            zip(_input, output),
            desc="Processing spectra",
            total=len(_input),
            disable=not display_progress_bar,
        ):
            if spectra is None:
                continue
            else:
                if post_process:
                    spectra = post_process_api_spectrum(spectra)
                data[spotter.spotter_id] = spectra

        return BuoySpectrum.from_dictionary(data)


def _to_spectrum(api_data: List[dict], post_process) -> Spectrum:
    """
    Convert a dictionary created from a wavefleet api response into a spectral object.
    :param api_data: json blob from the api containing spectral data ("frequencyData")
    :param spotter_id: id of the spotter
    :return: a Spectrum object
    """
    if len(api_data) == 0:
        raise ValueError("No data in api_data")

    number_of_freqs = len(api_data[0]["frequency"])
    number_of_spec = len(api_data)

    # The frequency data SHOULD be the same for all spectra, so we only need to get it once
    data = {"frequency": np.array(api_data[0]["frequency"])}

    # Initialize (numpy) arrays for the data
    for key in ["varianceDensity", "a1", "b1", "a2", "b2"]:
        data[key] = np.empty((number_of_spec, number_of_freqs))

    for key in ["latitude", "longitude"]:
        data[key] = np.empty(number_of_spec)

    data["timestamp"] = np.empty(number_of_spec, dtype="datetime64[ns]")

    # Load data into arrays
    for i, entry in enumerate(api_data):
        assert len(entry["frequency"]) == number_of_freqs

        for key in ["varianceDensity", "a1", "b1", "a2", "b2"]:
            data[key][i, :] = np.array(entry[key])

        for key in ["latitude", "longitude"]:
            data[key][i] = entry[key]

        data["timestamp"][i] = pd.Timestamp(entry["timestamp"]).value

    # Create the dataset
    dataset = xr.Dataset(
        data_vars={
            "variance_density": (
                ("time", "frequency"),
                data["varianceDensity"],
            ),
            "a1": (
                ("time", "frequency"),
                data["a1"],
            ),
            "b1": (
                ("time", "frequency"),
                data["b1"],
            ),
            "a2": (
                ("time", "frequency"),
                data["a2"],
            ),
            "b2": (
                ("time", "frequency"),
                data["b2"],
            ),
            "latitude": (
                ("time",),
                data["latitude"],
            ),
            "longitude": (
                ("time",),
                data["longitude"],
            ),
        },
        coords={
            "time": data["timestamp"],
            "frequency": data["frequency"],
        },
    )
    spectrum = Spectrum(dataset)

    if post_process:
        spectrum = post_process_api_spectrum(spectrum)
    return spectrum
