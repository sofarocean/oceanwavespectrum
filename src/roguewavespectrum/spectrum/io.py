from xarray import open_dataset
from .wavespectrum2D import FrequencyDirectionSpectrum
from .wavespectrum1D import FrequencySpectrum
from typing import Union
from .variable_names import NAME_D

def load_spectrum_from_netcdf(
    filename_or_obj,
) -> Union[FrequencySpectrum, FrequencyDirectionSpectrum]:
    """
    Load a spectrum from a netcdf file. Depending on whether the netcdf file contains directions a frequency spectrum
    or a frequency-direction spectrum is returned.

    :param filename_or_obj: filename or file-like object
    :return: FrequencySpectrum or FrequencyDirectionSpectrum
    """
    dataset = open_dataset(filename_or_obj=filename_or_obj)
    if NAME_D in dataset.coords:
        return FrequencyDirectionSpectrum(dataset=dataset)
    else:
        return FrequencySpectrum(dataset=dataset)