"""
Rogue wave spectrum package.
===========================

The rogue wave spectrum package is principally designed to facilitate work with ocean wave spectra through the
implementation of a spectral object. The spectral object is designed to be flexible and can be used to represent
spectra in one, two or three dimensions. The spectral object is designed to be compatible with the xarray package,
and can be used to represent spectra in a way that is compatible with the CF conventions.
"""

from .spectrum import (
    Spectrum2D,
    Spectrum1D,
    Spectrum,
    DrifterSpectrum,
    concatenate_spectra,
)
