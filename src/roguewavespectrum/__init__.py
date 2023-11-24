"""
Rogue wave spectrum package.
===========================

Introduction
------------
The rogue wave spectrum package is principally designed to facilitate work with ocean wave spectra through the
implementation of a spectral object. Specifically:
- The spectral object is designed to be flexible and can be used to represent spectra in one, two or more dimensions.
- The spectral object is designed to be compatible with the xarray package. In particular, the spectral object is
    effectively a thin wrapper around a xarray dataset, and all the functionality of the xarray package is available
    by operating directly on the dataset property of the spectral object.
- Operations on spectra are vectorized as much as possible. Though performance was not a primary concern in the design.
- Returned (bulk)properties are always returned as xarray datasets, contain the appropriate coordinates and metadata
    that is compatible with the CF conventions.
- Most common bulk properties are implemented, and can be calculated from the spectral object. These include e.g. the
    significant wave height, the mean wave direction, mean wave periods, etc.
- Functions are provided to e.g. easily create spectra from numpy arrays, to create spectra for common parametric
    shapes, to save/load spectra to/from cf-compliant netcdf files, and to estimate a 2D spectrum from a 1D spectrum.

For a good introduction to the concept of a variance density spectrum we refer to Holthuijsen (2010),
and here we will tacitly assume that the reader is familiar with the concept.

Why this package - and what is its future?
------------------------------------------
This package is merely a formalization of my personal workflow for working with ocean wave spectra within the Sofar
ecosystem. After sharing snippets of code with colleagues, and occasionaly outside collaborators, I decided to formalize
this into something that could be shared more easily (e.g. through a pip install). Note that while hosted on
SofarOcean's github, this package is not intended to be a SofarOcean product, and is not formally supported by
SofarOcean.

Note that I make no garantees about the correctness of the code, and I make no garantees about the future of this
package, nor any garantees about support.

Long term, I hope that this package will be useful to others, and that it may be used to facilitate collaboration. To
this end I made an effort to document the code, make some sane (at least to me) API choices, and to make it as easy as
possible to install and use.

Ideally I would love to make this a community effort, and I would love to see others contribute to this package and
become maintainers.

How well is this package tested?
--------------------------------
The package contains a number of unit tests, but those mostly test if the code runs, and if the code is consitent
with previous behaviour, not if the code is correct. Most of these routines have been battlehardened through use in
my own work (e.g. used in publications). With regard to domain knowledge, I am an expert in ocean waves, and I have
been working with ocean wave spectra for over a decade, and all calculations are based on well established theory.

Why the name?
--------------------------------
Mostly because the name was available, I am not very creative when it comes to naming, and I liked it better than
something more generic.

Spectra
=======
We consider ocean surface-wave variance density spectra, either as:

- $E(\\ldots,f,\\theta)$ (2D spectrum), units [m^2/Hz/degree]
- $e(\\ldots,f)$ (1D spectrum), units [m^2/Hz]

with $\\theta$ direction in degrees, and $f$ frequency in Hertz. The 1d spectrum follows from the 2D spectrum after
integration over the circle

$$e(\\ldots,f) = \\int_0^{360} E(\\ldots,f,\\theta)\\,\\mathrm{d}\\theta$$

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

I want to
===================
Create a spectrum from numpy arrays
-----------------------------------
For this we use the `roguewavespectrum.create_spectrum1d` and `roguewavespectrum.create_spectrum2d` factory methods,
depending on whether we want to create a 1D or 2D spectrum.

For example, lets assume we have a bunch of spectra as a function of time and frequency, and have numpy arrays for the
frequency, time, variance density, and the directional moments a1,b1,a2,and b2 (here we will use random nonphysical
data)
>>> import numpy as np
>>> from pandas import Timestamp
>>> frequency = np.linspace(0,1,100)
>>> time = np.array([Timestamp("2020-01-01 00:00:00"),Timestamp("2020-01-01 01:00:00")])
>>> variance_density = np.random.rand(2,100)
>>> a1 = np.random.rand(2,100)
>>> b1 = np.random.rand(2,100)
>>> a2 = np.random.rand(2,100)
>>> b2 = np.random.rand(2,100)

Note that the frequency MUST be the last dimension of the variance_density, a1,b1,a2,b2 arrays, and all arrays must have
the same shape.

In this case we can create a spectrum using the `roguewavespectrum.create_spectrum1d` factory method:
>>> from roguewavespectrum import create_spectrum1d
>>> spectrum = create_spectrum1d( [("time",time),("frequency",frequency)], variance_density, a1, b1, a2, b2)

The first argument to the `roguewavespectrum.create_spectrum1d` method is a list of tuples that specify the name and
values of the coordinates. The first tuple specifies the name and values of the first coordinate, the second tuple
specifies the name and values of the second coordinate, etc. The last tuple must always be the frequency coordinate.

The processes for creating a 2D spectrum is similar, though now we need to specify the frequency and directin - i.e.
>>> from roguewavespectrum import create_spectrum2d
>>> spectrum = create_spectrum2d( [("time",time),("frequency",frequency),("direction",direction)], variance_density)

In this case moments are no longer required - but we do require a direction coordinate (and it MUST be the last
dimension).

For further details see the documentation of the `roguewavespectrum.create_spectrum1d` method.

Create a spectrum from a xarray dataset
---------------------------------------
For this we use the `roguewavespectrum.Spectrum1D.from_dataset` and `roguewavespectrum.Spectrum2D.from_dataset` methods.
The datasets must contain the appropriate variables and coordinates (see the documentation for details).

Note that the date in the xarray dataset may be located remomtely e.g. as a zarr archive on s3, and loading of the data
will in generally occur lazily.

Note; the spectrum object for all intents and purposes is a thin wrapper around the xarray dataset object, and the
(shallow) copy of the dataset is stored as the `roguewavespectrum.Spectrum.dataset` attribute.

Estimate a 2D spectrum from a 1D spectrum
-----------------------------------------
For this we use the `roguewavespectrum.Spectrum1D.as_frequency_direction_spectrum` method. This method requires that
the directional moments a1,b1,a2,b2 are available. If they are not available, the method will raise a ValueError. Given
a `spectrum1d` we can estimate the 2D spectrum as follows
>>> spectrum2d = spectrum1d.as_frequency_direction_spectrum(
number_of_directions=36,method='mem2',solution_method='scipy')

Here we have specified that we want to estimate the 2D spectrum using the MEM2 method, and that we want to use the
scipy solver to solve the nonlinear set of equations in MeM2. The number of directions is specified by the
`number_of_directions` argument such that the directions are given by
`np.linspace(0,360,number_of_directions,endpoint=False)`

For further details see the documentation of the `roguewavespectrum.Spectrum1D.as_frequency_direction_spectrum` method.

Calculate a 1D spectrum from a 2D spectrum
-----------------------------------------
For this we use the `roguewavespectrum.Spectrum2D.as_frequency_spectrum` method. This will return a 1D spectrum,
including the a1,b1,a2,b2 moments as estimated from the 2D spectrum.

Create a Parametric Spectrum
----------------------------
We refer to the `roguewavespectrum.parametric` module for this, and examples in the documentation. This module contains
a number of parametric spectral shapes, and a function to create a parametric spectrum from a frequency and direction
coordinate.

Save/Load a spectrum to/from a cf-compliant netcdf file
---------------------------------------------
For this we use the `roguewavespectrum.Spectrum.to_netcdf` method. This will save the spectrum to a netcdf file that
is compliant with the CF conventions.

Conversely, we can load a spectrum from a netcdf file callung the `from_netcdf` method on a Spectrum1D or Spectrum2D
class. Note that presently prior knowledge of the type of spectrum is required, i.e. if we have a 2D spectrum we need to
call the method on the Spectrum2D class.

WARNING: cf_convensions are _implemented_, but presently not well tested outside of the package itself (i.e. things are
internally consistent, but errors may still exist in naming convention).

References
==========

    Holthuijsen, L. H. (2010). Waves in oceanic and coastal waters. Cambridge university press.

"""
# Some import shananigans to make sure pydoc documents these imports without having to expose the private package.
from ._spectrum import Spectrum as _Spectrum
from ._wavespectrum2D import Spectrum2D as _Spectrum2D
from ._wavespectrum1D import Spectrum1D as _Spectrum1D
from ._buoyspectrum import BuoySpectrum1D as _BuoySpectrum
from ._factory_methods import create_spectrum1d as _create_spectrum1d
from ._factory_methods import create_spectrum2d as _create_spectrum2d
from ._factory_methods import (
    create_parametric_spectrum1d as _create_parametric_spectrum1d,
)
from ._factory_methods import (
    create_parametric_spectrum2d as _create_parametric_spectrum2d,
)
from ._operations import concatenate_spectra

import roguewavespectrum.parametric as parametric

create_spectrum1d = _create_spectrum1d
create_spectrum2d = _create_spectrum2d
create_parametric_spectrum1d = _create_parametric_spectrum1d
create_parametric_spectrum2d = _create_parametric_spectrum2d

Spectrum = _Spectrum
Spectrum1D = _Spectrum1D
Spectrum2D = _Spectrum2D
BuoySpectrum = _BuoySpectrum
