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
with previous behaviour, not if the code is correct. Most of these routines have been in use in
my own work and calculations are based on well established theory.

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
The `roguewavespectrum.Spectrum` object implements a data container for and methods applicable to spectra. It has been
implemented as a thin wrapper around the xarray dataset object. Very specifically; the only state of the object is
stored in the xarray dataset object (which is given), and all methods are implemented as methods on the xarray dataset
object. In principle this should not be visible to the user, but may be useful if you want to extend the functionality
of the Spectrum object (or extract information otherwise not available). It also means that xarray functionally (e.g.
using remote zarr stores) is available to the user. While the spectrum object is thus essentially an extended Dataset
object, I decided against inheriting from the xarray dataset object, as not all methods are applicable to the spectral
object, and I wanted to avoid confusion (also following composition over inheritance).

BuoySpectrum Object
-------------------
The `roguewavespectrum.BuoySpectrum` object is a subclass of the `roguewavespectrum.Spectrum` object, and is designed
to specifically work with data from multiple different buoys under the assumption that the buoys have similar frequency
ranges, but may not report at the same time (or have the same number of time entries) so that a 3D representation of the
buoy dataset [buoy,time,frequency] is not possible (ragged arrays - we cannot use time as a coordinate). Instead, we
represent all spectra as a single concatenated 2D dataset [index,frequency], and keep track of the buoy id's in a
separate array. To inquire about the buoy id's we use the `buoy_spectrum.keys()` property, so that we may get a spectrum
for an individual buoy as `buoy_spectrum[buoy_spectrum.keys()[0]]`. Essentially access acts as a dictionary, though
assignment is not possible.

Example
-------
To give an example of a spectrum, and its usage, we can create a parametric 1d spectrum as follows
```python
from roguewavespectrum import create_parametric_spectrum1d
frequencies = np.linspace(0,1,100)
spec1d = create_parametric_spectrum1d(
    frequencies,
    waveheight=[2,3],
    period=[10,11],
    shape_name='jonswap'
    )
```

where we have created a spectrum object with two spectra, one with waveheight 2 and period 10, and one with waveheight 3
and peak period 11. To calculate (say) mean periods or recalculate significant wave height we can do
```python
mean_period = spec1d.mean_period()
significant_waveheight = spec1d.hm0()
```

Note that for all bulk properties we return xarray datasets, with the appropriate coordinates and metadata to be CF
compliant (if possible).

Since 1D and 2D spectra only differ in dimensionallity- one object represents both type of spectra. Hence,
```python
from roguewavespectrum import create_parametric_spectrum2d
spec2d = create_parametric_spectrum2d(
    frequencies,
    number_of_directions=36,
    waveheight=2,
    period=10,
    direction=0.0,
    spread=30.0,
    frequency_shape_name='jonswap',
    direction_shape_name='cosN'
    )
```

returns a Spectrum object as well. However, some properties or operations are only available if the underlying spectrum
is a 2d spectrum. For example
```python
directions = spec.direction(directional_unit='degree',directional_convention='oceanographical')
```

will raise an error if the underlying spectrum is a 1d spectrum (but work for a 2d spectrum). As an easy check we can
do `spec.is_2d` to check if the spectrum is a 2d spectrum.


Bulk Properties
===============
Bulk properties are those that depend on a moment (or other weighted integrated property) of the spectrum. When
considering weighting across frequencies, these moments ar of the form
$$ m_n = \\int_{f_\\text{min}}^{f_\\text{min}} f^n e(f)\\,\\mathrm{d}f$$

where $n$ is an integer, and $f_\\text{min}$ and $f_\\text{max}$ are the minimum and maximum frequencies over which we
consider the moment. For example, the significant wave height is given by $4\\sqrt{m_0}$. Typically $f_\\text{min}=0$
and $f_\\text{max}=\\infty$. Because we often wish to calculate bulk properties over a subset of the frequencies (e.g.
swell significant wave heigth) all bulk properties take `fmin` and `fmax` as arguments. Here we calculate moments
(and frequency integrals in general) numerically using the trapizonal rule. If fmin and fmax are given, we first
bandpass the spectrum, and ensuring that fmin and fmax are actual bounds through interpolation if need be, and then
calculate the moment.

Frequency Moments
-----------------
Defined as
$$ m_n = \\int_{f_\\text{min}}^{f_\\text{min}} f^n e(f)\\,\\mathrm{d}f$$

and calculated by `roguewavespectrum.Spectrum.frequency_moment`. (for details see the documentation of the method).

significance waveheight $H_{m_{0}}$
-----------------------------------
Defined as
$$ h_{m_{0}} = 4\\sqrt{m_0}$$

and calculated by `roguewavespectrum.Spectrum.hm0`. (for details see the documentation of the method). Note that we
follow usual practice and set the proportionality coefficient to 4, (not 4.004).

mean wave period $T_{m_{0,1}}$
---------------------------------
Defined as
$$ T_{m_{0,1}} = \\frac{m_0}{m_{1}}$$

Technically this is the period associated with the mean frequency. Calculated by
`roguewavespectrum.Spectrum.mean_period`.

zero-crossing period $T_{m_{0,2}}$
-------------------------------------
Defined as
$$ T_{m_{0,2}} = \\sqrt{\\frac{m_0}{m_{2}}}$$

Sometimes also referred to as the zero-crossing period. Calculated by `roguewavespectrum.Spectrum.tm02`.

wave energy period $T_{m_{-1,0}}$
----------------------------------
Defined as
$$ T_{m_{-1,0}} = \\frac{m_{-1}}{m_{0}}$$

Spectrally weighted mean period referred to as the wave energy period. Calculated by
`roguewavespectrum.Spectrum.energy_period`.

peak period $T_{p}$
----------------------------------
Defined as
$$ T_{p} = \\text{argmax}_T \\,e(f=1/T)$$

and calculated by `roguewavespectrum.Spectrum.peak_period`. Note that by default the peak period is merely the inverse
of the frequency of the bin with the highest energy, but more sophisticated methods are available (see the
documentation of the method).

Mean wave direction $\\theta_m$
----------------------------------
Regardless whether the spectrum is a 1D or 2D spectrum, the mean wave direction is defined according to Kuik et al.,
using the mean directional moments $\\bar{a}_1$ and $\\bar{b}_1$ as
$$ \\theta_m =  \\frac{180}{\\pi}\\text{arctan2}\\, \\left(\\bar{b}_1,\\bar{a}_1\\right)$$

where arctan2 denotes the multivalued arc-tangent function to resolve the full 360 circle, and
$\\bar{a}_1$ and $\\bar{b}_1$ are weighted mean directional moments (see `roguewavespectrum.Spectrum.mean_a1` and
`roguewavespectrum.Spectrum.mean_b1`).Calculated by `roguewavespectrum.Spectrum.mean_direction`.

Note that per default returned angles are in degrees, $\\theta_m \\in [0,360)$, and the direction convention is
mathematical (going to, measured counter-clockwise from east). Oceanographical, Meteorological and radian convention
are available as options.

I want to ...
===================
... create a spectrum from numpy arrays
-----------------------------------
For this we use the `roguewavespectrum.create_spectrum1d` and `roguewavespectrum.create_spectrum2d` factory methods,
depending on whether we want to create a 1D or 2D spectrum.

For example, lets assume we have a bunch of spectra as a function of time and frequency, and have numpy arrays for the
frequency, time, variance density, and the directional moments a1,b1,a2,and b2 (here we will use random nonphysical
data)
```python
import numpy as np
from pandas import Timestamp
frequency = np.linspace(0,1,100)
time = np.array([Timestamp("2020-01-01 00:00:00"),Timestamp("2020-01-01 01:00:00")])
variance_density = np.random.rand(2,100)
a1 = np.random.rand(2,100)
b1 = np.random.rand(2,100)
a2 = np.random.rand(2,100)
b2 = np.random.rand(2,100)
```

Note that the frequency MUST be the last dimension of the variance_density, a1,b1,a2,b2 arrays, and all arrays must have
the same shape.

In this case we can create a spectrum using the `roguewavespectrum.create_spectrum1d` factory method:
```pythob
from roguewavespectrum import create_spectrum1d
spectrum = create_spectrum1d( [("time",time),("frequency",frequency)], variance_density, a1, b1, a2, b2)
```

The first argument to the `roguewavespectrum.create_spectrum1d` method is a list of tuples that specify the name and
values of the coordinates. The first tuple specifies the name and values of the first coordinate, the second tuple
specifies the name and values of the second coordinate, etc. The last tuple must always be the frequency coordinate.

The processes for creating a 2D spectrum is similar, though now we need to specify the frequency and direction - i.e.
```python
from roguewavespectrum import create_spectrum2d
spectrum = create_spectrum2d(
    [ ("time",time),("frequency",frequency),("direction",direction) ], variance_density)
```

In this case moments are no longer required - but we do require a direction coordinate (and it MUST be the last
dimension).

For further details see the documentation of the `roguewavespectrum.create_spectrum1d` method.

... create a spectrum from a xarray dataset
---------------------------------------
For this we use the `roguewavespectrum.Spectrum.from_dataset` methods.
The datasets must contain the appropriate variables and coordinates (see the documentation for details).

Note that the date in the xarray dataset may be located remomtely e.g. as a zarr archive on s3, and loading of the data
will in generally occur lazily.

Note; the spectrum object for all intents and purposes is a thin wrapper around the xarray dataset object, and the
(shallow) copy of the dataset is stored as the `roguewavespectrum.Spectrum.dataset` attribute.

... estimate a 2D spectrum from a 1D spectrum
-----------------------------------------
For this we use the `roguewavespectrum.Spectrum.as_frequency_direction_spectrum` method. This method requires that
the directional moments a1,b1,a2,b2 are available. If they are not available, the method will raise a ValueError. Given
a `spectrum` we can estimate the 2D spectrum as follows
```python
spectrum2d = spectrum1d.as_frequency_direction_spectrum(number_of_directions=36,method='mem2',solution_method='scipy')
```
Here we have specified that we want to estimate the 2D spectrum using the MEM2 method, and that we want to use the
scipy solver to solve the nonlinear set of equations in MeM2. The number of directions is specified by the
`number_of_directions` argument such that the directions are given by
`np.linspace(0,360,number_of_directions,endpoint=False)`

For further details see the documentation of the `roguewavespectrum.Spectrum.as_frequency_direction_spectrum` method.

... calculate a 1D spectrum from a 2D spectrum
-----------------------------------------
For this we use the `roguewavespectrum.Spectrum.as_frequency_spectrum` method. This will return a 1D spectrum,
including the a1,b1,a2,b2 moments as estimated from the 2D spectrum.

... create a Parametric Spectrum
----------------------------
We refer to the `roguewavespectrum.parametric` module for this, and examples in the documentation. This module contains
a number of parametric spectral shapes, and a function to create a parametric spectrum from a frequency and direction
coordinate.

... save/Load a spectrum to/from a cf-compliant netcdf file
---------------------------------------------
For this we use the `roguewavespectrum.Spectrum.to_netcdf` method. This will save the spectrum to a netcdf file that
is compliant with the CF conventions. Conversely, we can load a spectrum from a netcdf file callung the `from_netcdf`
class method on the  Spectrum class.

WARNING: cf_convensions are _implemented_, but presently not well tested outside of the package itself (i.e. things are
internally consistent, but errors may still exist in naming convention).

References
==========

    Holthuijsen, L. H. (2010). Waves in oceanic and coastal waters. Cambridge university press.

    Kuik, A. J., Van Vledder, G. P., & Holthuijsen, L. H. (1988). A method for the routine analysis of pitch-and-roll
    buoy wave data. Journal of physical oceanography, 18(7), 1020-1034.

"""
# Some import shananigans to make sure pydoc documents these imports without having to expose the private package.
from ._spectrum import Spectrum as _Spectrum
from ._spectrum import BuoySpectrum as _BuoySpectrum
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
import roguewavespectrum.spotter

create_spectrum1d = _create_spectrum1d
create_spectrum2d = _create_spectrum2d
create_parametric_spectrum1d = _create_parametric_spectrum1d
create_parametric_spectrum2d = _create_parametric_spectrum2d

Spectrum = _Spectrum
BuoySpectrum = _BuoySpectrum
