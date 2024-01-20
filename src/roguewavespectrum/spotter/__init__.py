"""
Introduction
===========================

The classes and functions in this module are used to facilitate interaction with data from Sofar Spotter devices, or
from the Sofar API.

Spotter API
===========================
To interact with the Spotter API you need to have a acount with the associated valid token. To get a token, please
follow instructions at https://api-token.sofarocean.com/.

In general the process for using the API is as follows:
- Create an instance of the SofarApi class.
- Use the instance to make requests to the API.

Right now the SofarApi class support limited interaction with the API. Specifically, it supports getting a list of
spotter IDs associated with an account, and getting spectra for specific spotter IDs. If you need to make other types
of requests, we refer to the pysofar package.

Example: retrieving spectra
---------------------------
The following example shows how to get a list of all spotter IDs associated with an account,
and then get spectra for a single spotter.
```python
from roguewavespectrum.spotter import SofarFleet
import os

# Get token from environment variable. You can also hard code the token - but it is generally good practice to avoid
# this as it may inadvertently be committed to a public repository.
token = os.getenv('TOKEN')

# Create an instance of the SofarApi class.
api = SofarFleet(token=token)

# Get a list of all spotter IDs.
spotter_ids = api.device_ids

# Get spectra for a single spotter.
spectra = api.get_spectra(start_date='2024-01-01', end_date='2024-01-02', device_ids=spotter_ids[0])
```

Example: sessions, retries
---------------------------
SofarAPI uses the requests library to handle interaction with the API. On creating the class instance, by default a
requests session is created that will retry requests up to 5 times, specified as:
```python
import requests
from requests.adapters import HTTPAdapter, Retry
session = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
session.mount("https://", requests.adapters.HTTPAdapter(max_retries=retries))
```
If you want to use a different session, you can pass it to the SofarApi class on creation:
```python
from roguewavespectrum.spotter import SofarFleet
session = ... # your session
api = SofarFleet(token=token, session=session)
```
Example: caching
---------------------------
Setting a session specifically can be useful if you want to use a session that has a different retry policy,
or if you want to use a session that has caching enabled. For example, to use a session that caches requests to the API,
we can use the requests-cache package:
```python
from requests_cache import CachedSession
from roguewavespectrum.spotter import SofarFleet
requests_cache_path = '~/temp'
session = requests_cache.CachedSession(cache_name=requests_cache_path, backend='sqlite')
api = SofarFleet(token=token, session=session)
```
For more information on caching, see the documentation of the requests-cache package. Note that as a convenience, we
can also enable caching directly on the SofarApi class, by setting `session=cached`. In this case
the API will create a session with caching enabled, and use that session for all requests:
```python

from roguewavespectrum.spotter import SofarFleet
cache_file_path = '~/temp'
api = SofarFleet(token=token, session=cached, requests_cache_path=cache_file_path)
...
```
From here on, the API will cache the results of requests to the API in the specified file. Note that the
`requests_cache_path` argument is only used if `session=cached` is specified, and per default is set to
`spotter_api_requests_cache.db`. To re-use the cache in other scripts, you can create a new instance of the SofarApi
class with the same cache file path. However, if you want to have more control on cache behaviour
(e.g. retention policy), you have to create your own session and pass it to the SofarApi class.

Reading Spectral CSV data produced by the Spotter SD card parser
================================================================
Spotter can store data locally on an SD card. Once the card is retrieved from the spotter and extracted, the data can
be parsed into a csv format that contains spectral data using the Sofar parser script (see the Sofar documentation for
more information). The spectral data is stored in the following files: a1.csv, b1.csv, a2.csv, b2.csv, Szz.csv.

To read the spectral data from these files, use the `read_spectral_csv` function. This function returns a Spectrum
object.

Example: reading spectral data
------------------------------
The following example shows how to read spectral data from a folder containing the csv files. Note that all csv files
must be in the same folder.
```python
from roguewavespectrum.spotter import read_spectral_csv
spectrum = read_spectral_csv('path/to/spectral/data')
```
"""

from ._spotter_sdcard_data import read_spectral_csv as _read_spectral_csv
from ._spotter_wavefleet_api import SofarFleet as _SofarFleet

SofarFleet = _SofarFleet

read_spectral_csv = _read_spectral_csv
