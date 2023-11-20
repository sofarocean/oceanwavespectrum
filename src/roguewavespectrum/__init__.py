"""
Rogue wave spectrum package.
===========================

The rogue wave spectrum package is principally designed to facilitate work with ocean wave spectra through the
implementation of a spectral object. The spectral object is designed to be flexible and can be used to represent
spectra in one, two or more dimensions. The spectral object is designed to be compatible with the xarray package,
and can be used to represent spectra in a way that is compatible with the CF conventions.

For a good introduction to the concept of a variance density spectrum we refer to Holthuijsen (2010),
and here we will tacitly assume that the reader is familiar with the concept.

Spectra
=======
We consider ocean surface-wave variance density spectra, either as:

- $E(\ldots,f,\\theta)$ (2D spectrum), units [m^2/Hz/degree]
- $e(\ldots,f)$ (1D spectrum), units [m^2/Hz]

with $\\theta$ direction in degrees, and $f$ frequency in Hertz. The 1d spectrum follows from the 2D spectrum after
integration over the circle

$$e(\ldots,f) = \int_0^{360} E(\ldots,f,\\theta)\,\mathrm{d}\\theta$$

Note that the spectra conceptually may have arbitrary leading dimensions. For example, $E(t,x,y,f)$ could be the
frequency-spectrum as a function of  time, latitude, longitude and frequency. We still refer to this as a 1D spectrum,
noting that "dimensionality" here refers to the spectral dimensions (frequency, direction). Further, we adopt the
convention that spectral dimensions are always the trailing dimensions, and if present $\\theta$ is always the
innermost dimension of the spectral objects.

Spectrum Object
---------------
The `roguewavespectrum.Spectrum` object contains all the logic that is shared between 1D and 2D spectra. It is
implemented as an abstract base class - and should *never* be instantiated directly. All properties/methods documented
as part of the Spectrum class are available on Spectrum1D and spectrum2D as well.

Spectrum1D $e(\\ldots,f)$
-----------------
The `roguewavespectrum.Spectrum1D` object implements all logic specific to 1D spectra. A Spectrum1D may be converted
into a 2D spectrum using the `roguewavespectrum.Spectrum1D.as_frequency_direction_spectrum` method _if_ directional
moments are available.

Spectrum2D $e(\\ldots,f,\\theta)$
-----------------
The `roguewavespectrum.Spectrum2D` object implements all logic specific to 2D spectra. A Spectrum2D may be converted
into a 1D spectrum using the `roguewavespectrum.Spectrum1D.as_frequency_spectrum` method.

Spectral properties
===================

Bulk properties
===============

Wave properties
===============


Definitions
===========
Directions
----------
When creating a spectrum, or when querying a directional property, directions are by default expressed with
"units" degrees, and with the convention that they are measured counter-clockwise from east, and refer to the direction
the waves are travelling *to*.

References
==========

    Holthuijsen, L. H. (2010). Waves in oceanic and coastal waters. Cambridge university press.

Most base functionality is implemented in the abstract `roguewavespectrum.Spectrum` class.

"""
# Some import shananigans to make sure pydoc documents these imports without having to expose the private package.
from ._spectrum import Spectrum as _Spectrum
from ._wavespectrum2D import Spectrum2D as _Spectrum2D
from ._wavespectrum1D import Spectrum1D as _Spectrum1D
from ._wavespectrum1D import DrifterSpectrum as _DrifterSpectrum

Spectrum = _Spectrum
Spectrum1D = _Spectrum1D
Spectrum2D = _Spectrum2D
DrifterSpectrum = _DrifterSpectrum

from ._operations import concatenate_spectra
from ._factory_methods import create_spectrum_1d
import roguewavespectrum.parametric as parametric
