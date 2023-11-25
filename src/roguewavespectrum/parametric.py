"""
Parametric Spectra
==================
This module contains routines for easily generating parametric wave spectra and return them as WaveSpectrum objects.
The primary function are `parametric_directional_spectrum` and ` parametric_spectrum`,
which generate a parametric frequency-direction spectrum and a parametric frequency spectrum, respectively. Curently
the following frequency spectra are supported: Pierson-Moskowitz, JONSWAP, Phillips, and Gaussian. For the directional
shape the options are a raised cosine distribution (Pierson et al., 1952, Longuet-Higgins, 1963), either defined on the
half plane (CosineN) or on the full plane (DirCosineN). A good reference for the parametric spectra and their
expressions is Holthuijsen (2010), sections 6.3.3 and 6.3.4.

Use
===

Create Frequency-directional spectra (2d)
------------------------------------------
To create a parametric 2d spectrum, use the `parametric_directional_spectrum` function. This function will require a
frequency and direction grid as input, as well as a frequency and directional shape that specify the parametric shape.

Create Frequency spectra (1d)
-----------------------------
To create a parametric 1d spectrum, use the `parametric_spectrum` function. This function will require a
frequency grid as input, as well as a frequency shape that specifies the parametric shape.

Create a frequency shape
------------------------
A frequency shape is a class that describes the shape of the spectrum. Currently the following frequency shapes are
supported:
- Pierson-Moskowitz, `FreqPiersonMoskowitz`
- JONSWAP, `FreqJonswap`
- Phillips, `FreqPhillips`
- Gaussian, `FreqGaussian`

for example, to create a JONSWAP spectrum with a peak frequency of 0.1 Hz and a significant wave height of 2 m, use:
>>> freq_shape = FreqJonswap( peak_frequency_hertz=0.1, significant_waveheight_meter=2)

Create a directional shape
--------------------------
A directional shape is a class that describes the directional shape of the spectrum. Currently the following directional
shapes are supported:
- Raised cosine, `DirCosineN`
- Raised cosine 2N, `DirCosine2N`

for example, to create a raised cosine distribution with a mean direction of 35 degrees and a width of 20 degrees, use:
>>> dir_shape = DirCosineN( mean_direction_degrees=35, width_degrees=20)

Examples
========

Create and plot a Frequency spectrum (1d)
-----------------------------------------
>>> import numpy as np
>>> from matplotlib.pyplot as plt
>>> from roguewavespectrum.parametric import parametric_spectrum, FreqJonswap
>>>
>>> frequency = np.linspace(0.05, 0.5, 100)
>>> shape = FreqJonswap( peak_frequency_hertz=0.1, significant_waveheight_meter=1)
>>> spectrum = parametric_spectrum( frequency, shape )
>>> plt.plot( spectrum.frequency, spectrum.variance_density )
>>> plt.show()

References
==========

    Pierson Jr, W. J., & Moskowitz, L. (1964). A proposed spectral form for fully developed wind seas based on the
    similarity theory of SA Kitaigorodskii. Journal of geophysical research, 69(24), 5181-5190.

    Pierson Jr, W. J., Tuttell, J. J., & Woolley, J. A. (1952). The theory of the refraction of a short crested Gaussian
    sea surface with application to the northern New Jersey coast. Coastal Engineering Proceedings, (3), 8-8.

    Hasselmann, K., Barnett, T. P., Bouws, E., Carlson, H., Cartwright, D. E., Enke, K., ... & Walden, H. (1973).
    Measurements of wind-wave growth and swell decay during the Joint North Sea Wave Project (JONSWAP).
    Ergaenzungsheft zur Deutschen Hydrographischen Zeitschrift, Reihe A.

    Hasselmann, D. E., Dunckel, M., & Ewing, J. A. (1980). Directional wave spectra observed during JONSWAP 1973.
    Journal of physical oceanography, 10(8), 1264-1280.

    Holthuijsen, L. H. (2010). Waves in oceanic and coastal waters. Cambridge university press.

    Longuet-Higgins, M. S. (1963). Observation of the directional spectrum of sea waves using the motions of a floating
    buoy. Oc. Wave Spectra.

    Phillips, O. M. (1958). The equilibrium range in the spectrum of wind-generated waves.
    Journal of Fluid Mechanics, 4(4), 426-434.
"""
from abc import ABC, abstractmethod
from scipy.special import gamma
import numpy
from ._time import to_datetime64
from xarray import Dataset
from ._physical_constants import GRAVITATIONAL_ACCELERATION
from ._variable_names import (
    NAME_F,
    NAME_D,
    NAME_E,
    NAME_LAT,
    NAME_LON,
    NAME_T,
    NAME_DEPTH,
)
from typing import Literal
from ._spectrum import Spectrum


def parametric_directional_spectrum(
    frequency_hertz: numpy.ndarray,
    direction_degrees: numpy.ndarray,
    frequency_shape: "FreqShape",
    direction_shape: "DirShape",
    renormalize: bool = True,
    **kwargs,
) -> Spectrum:
    """
    Create a parametrized directional frequency spectrum according to a given frequency or directional distribution.

    :param frequency_hertz: Frequencies to resolve
    :param peak_frequency_hertz:  Desired peak frequency of the spectrum
    :param frequency_shape:  Frequency shape object, see create_frequency_shape
    :param direction_shape:  Directional shape object, see create_directional_shape

    :return: FrequencyDirectionSpectrum object.
    """
    nat = numpy.datetime64().astype("<M8[ns]")
    time = to_datetime64(kwargs.get("time", nat))
    latitude = kwargs.get("latitude", numpy.nan)
    longitude = kwargs.get("longitude", numpy.nan)
    depth = kwargs.get("depth", numpy.nan)

    if frequency_hertz is None:
        fp = frequency_shape.peak_frequency_hertz
        frequency_hertz = numpy.linspace(0.5 * fp, 3 * fp, 26)

    if direction_degrees is None:
        direction_degrees = numpy.linspace(0, 360, 36, endpoint=False)

    D = direction_shape.values(direction_degrees, renormalize=renormalize)
    E = frequency_shape.values(frequency_hertz, renormalize=renormalize)

    dataset = Dataset(
        data_vars={
            NAME_E: ((NAME_F, NAME_D), E[:, None] * D[None, :]),
            NAME_DEPTH: ((), depth),
        },
        coords={
            NAME_F: frequency_hertz,
            NAME_D: direction_degrees,
            NAME_LAT: latitude,
            NAME_LON: longitude,
            NAME_T: time,
        },
    )

    return Spectrum(dataset)


def parametric_spectrum(
    frequency_hertz: numpy.ndarray,
    frequency_shape: "FreqShape",
    renormalize: bool = True,
    **kwargs,
) -> Spectrum:
    # We create a 1d spectrum from an integrated 2d spectrum with assumed raised cosine shape. This allows us to
    # add the a1/b1 parameters easily.
    direction_degrees = numpy.linspace(0, 360, 36, endpoint=False)

    direction_shape = kwargs.get("direction_shape", DirCosineN(0, 30))

    spec2d = parametric_directional_spectrum(
        frequency_hertz,
        direction_degrees,
        frequency_shape,
        direction_shape,
        renormalize=renormalize,
        **kwargs,
    )
    return spec2d.as_frequency_spectrum()


class DirShape(ABC):
    """
    Abstract base class for the directional shape of the wave spectrum. Note: Never instantiate this class directly,
    but use one of the subclasses.
    """

    def __init__(self, mean_direction_degrees: float = 0, width_degrees: float = 28.64):
        """
        Create a directional shape with the given mean direction and width.

        :param mean_direction_degrees: Mean direction of the waves in degrees in the assumed coordinate system.
        Default is 0.

        :param width_degrees: Width of the directional distribution in degrees
        """
        self.width_degrees = (
            width_degrees  #: Width of the directional distribution in degrees
        )
        self.mean_direction_degrees = mean_direction_degrees  #: Mean direction of the waves in degrees in the assumed
        # coordinate system.

    def values(
        self, direction_degrees: numpy.ndarray, renormalize: bool = False
    ) -> numpy.ndarray:
        """
        Return the value of the directional distribution (in degrees**-1) at the given angles in the assumed coordinate
        system. If renormalize is True, the distribution is renormalized so that the discrete integral (midpoint rule)
        over the distribution over the given directions is 1.

        :param direction_degrees: directions in degrees
        :param renormalize: renormalize the distribution so that the discrete integral over the distribution is 1.
        :return: numpy array with the values of the directional distribution
        """
        data = self._values(direction_degrees)
        if renormalize:
            # Renormalize so that the discretely integrated distribution is 1. First we need to estimate the bin size
            # for the given direction vector. We do this by calculating the forward and backward difference and taking
            # the average of the two. We need take into account that the direction vector is cyclic, so we need to
            # append the first value to the end and prepend the last value to the beginning and use modulo arithmetic.
            wrap = 360
            forward_diff = (
                numpy.diff(direction_degrees, append=direction_degrees[0]) + wrap / 2
            ) % wrap - wrap / 2
            backward_diff = (
                numpy.diff(direction_degrees, prepend=direction_degrees[-1]) + wrap / 2
            ) % wrap - wrap / 2
            bin_size = (forward_diff + backward_diff) / 2

            # With the binsize known, renormalize to 1
            data = data / numpy.sum(data * bin_size)

        return data

    @abstractmethod
    def _values(self, direction_degrees: numpy.ndarray) -> numpy.ndarray:
        ...


class DirCosineN(DirShape):
    """
    Raised cosine directional shape, $D(\\theta)=A \\cos^n(\\theta)$ where the normalization constant $A$ is chosen such
    that $\\int_0^{2\\pi} D(\\theta) d\\theta = 1$ (see Holthuijsen, 2010, section 6.3.3).
    """

    def __init__(self, mean_direction_degrees: float = 0, width_degrees: float = 28.64):
        """
        Create a raised cosine directional shape $D(\\theta)=A \\cos^n(\\theta-\\theta_0)$ with the given mean
        direction $\\theta_0$ and width $\\sigma_\\theta$.

        :param mean_direction_degrees: Mean direction of the waves in degrees in the assumed coordinate system.
        Default is 0.

        :param width_degrees: Width of the directional distribution in degrees. The power $n$ in the distribution is
        estimated such that the directional width corresponds to the given width. Default is 28.64 degrees, which
        corresponds to n=2.

        """
        super(DirCosineN, self).__init__(mean_direction_degrees, width_degrees)

    def _normalization(self, power):
        return (
            numpy.pi
            / 180
            * gamma(power / 2 + 1)
            / (gamma(power / 2 + 1 / 2) * numpy.sqrt(numpy.pi))
        )

    def _values(self, direction_degrees: numpy.ndarray) -> numpy.ndarray:
        """
        Calculate the directional distribution value at the given angles.
        :param direction_degrees: Direction in degrees in the assumed coordinate system.
        :param renormalize: If True, renormalize so that the discretely integrated distribution is 1. This is useful
        if the direction_degrees array is coarsely sampled, but we want to ensure that the discretely integrated
        distribution is 1.

        :return: Directional distribution value (degree^-1) at the given angles.
        """
        angle = (direction_degrees - self.mean_direction_degrees + 180) % 360 - 180
        power = self.width_degrees_to_power(self.width_degrees)
        with numpy.errstate(invalid="ignore", divide="ignore"):
            data = numpy.where(
                numpy.abs(angle) <= 90,
                self._normalization(power) * numpy.cos(angle * numpy.pi / 180) ** power,
                0,
            )
        return data

    @staticmethod
    def width_degrees_to_power(width_degrees):
        """
        Calculate power that gives the given width of the directional distribution. See Holthuijsen, 2010, section
        6.3.4. Note - this expression is adapted from eq 6.3.26 to use with a raised cosine distribution instead of a
        $\\cos^{2n}(2\\theta)$ distribution.

        :param width_degrees: Width of the directional distribution in degrees.
        """
        return 4 / ((numpy.pi * width_degrees / 90) ** 2) - 2

    @staticmethod
    def power_to_width_degrees(power):
        """
        Calulate width of the directional distribution for the given power. See Holthuijsen, 2010, section 6.3.4.
        Note - this expression is adapted from eq 6.3.26 to use with a raised cosine distribution instead of a
        $\\cos^{2n}(2\\theta)$ distribution.
        """
        return numpy.sqrt(4 / (power + 2)) * 90 / numpy.pi


class DirCosine2N(DirShape):
    """
    Raised cosine directional shape proposed by Longuet-Higgins,  1963, $D(\\theta)=A \\cos^2n(\\theta/2)$ where the
    normalization constant $A$ is chosen such that $\\int_0^{2\\pi} D(\\theta) d\\theta = 1$
    (see Holthuijsen, 2010, section 6.3.4).
    """

    def __init__(self, mean_direction_degrees: float = 0, width_degrees: float = 28.64):
        """
        Create a raised cosine directional shape $D(\\theta)=A \\cos^n(\\theta-\\theta_0)$ with the given mean
        direction $\\theta_0$ and width $\\sigma_\\theta$.

        :param mean_direction_degrees: Mean direction of the waves in degrees in the assumed coordinate system.
        Default is 0.

        :param width_degrees: Width of the directional distribution in degrees. The power $n$ in the distribution is
        estimated such that the directional width corresponds to the given width. Default is 28.64 degrees, which
        corresponds to n=2.

        """
        super(DirCosine2N, self).__init__(mean_direction_degrees, width_degrees)

    def _normalization(self, power):
        # See Holthuijsen, 2010, section 6.3.3
        return (
            numpy.pi
            / 180
            * gamma(power + 1)
            / (gamma(power + 1 / 2) * 2 * numpy.sqrt(numpy.pi))
        )

    def _values(self, direction_degrees: numpy.ndarray) -> numpy.ndarray:
        """
        Calculate the directional distribution value at the given angles.
        :param direction_degrees: Direction in degrees in the assumed coordinate system.
        :param renormalize: If True, renormalize so that the discretely integrated distribution is 1. This is useful
        if the direction_degrees array is coarsely sampled, but we want to ensure that the discretely integrated
        distribution is 1.

        :return: Directional distribution value (degree^-1) at the given angles.
        """
        angle = (direction_degrees - self.mean_direction_degrees + 180) % 360 - 180
        power = self.width_degrees_to_power(self.width_degrees)
        with numpy.errstate(invalid="ignore", divide="ignore"):
            data = numpy.where(
                numpy.abs(angle) <= 180,
                self._normalization(power)
                * numpy.cos(angle / 2 * numpy.pi / 180) ** (2 * power),
                0,
            )
        return data

    @staticmethod
    def width_degrees_to_power(width_degrees):
        """
        Calculate power that gives the given width of the directional distribution. See Holthuijsen, 2010, eq 6.3.26.

        :param width_degrees: Width of the directional distribution in degrees.
        """
        return 2 / (width_degrees * numpy.pi / 180) ** 2 - 1

    @staticmethod
    def power_to_width_degrees(power):
        """
        Calulate width of the directional distribution for the given power. See Holthuijsen, 2010, eq 6.3.26.
        """
        return numpy.sqrt(2 / (power + 1)) * 180 / numpy.pi


class FreqShape(ABC):
    """
    Abstract base class for the frequency shape of the wave spectrum. Note: Never instantiate this class directly,
    but use one of the subclasses.
    """

    def __init__(
        self, peak_frequency_hertz, significant_waveheight_meter: float = 1, **kwargs
    ):
        """
        Create a frequency shape with the given peak frequency and variance density.

        :param peak_frequency_hertz: Peak frequency of the spectrum
        :param significant_waveheight_meter: significant waveheight defined as $4*\\sqrt(m0)$, where m0 is the variance
            of the spectrum.
        """
        self.peak_frequency_hertz = peak_frequency_hertz
        self.significant_waveheight_meter = significant_waveheight_meter

    @property
    def m0(self) -> float:
        """
        Variance of the spectrum.
        :return:
        """
        return (self.significant_waveheight_meter / 4) ** 2

    def values(
        self, frequency_hertz: numpy.ndarray, renormalize=False
    ) -> numpy.ndarray:
        """
        Calculate the variance density (in m**2/Hz) at the given frequencies. If renormalize is True, the variance
        density is renormalized so that the discretely integrated spectrum (trapezoidal rule) will yield the specified
        significant waveheight exactly.

        :param frequency_hertz: frequencies in Hertz
        :param renormalize: renormalize so that the discretely integrated spectrum (trapezoidal rule) will yield the
            specified significant waveheight exactly.
        :return: numpy array with variance density (in m**2/Hz) at the given frequencies.
        """
        values = self._values(frequency_hertz)
        if renormalize:
            values = values * self.m0 / numpy.trapz(values, frequency_hertz)

        return values

    @abstractmethod
    def _values(self, frequency_hertz: numpy.ndarray) -> numpy.ndarray:
        ...


class FreqGaussian(FreqShape):
    """
    Gaussian frequency shape. Useful for testing and to model swell (with very narrow standard deviation).
    """

    def __init__(
        self, peak_frequency_hertz, significant_waveheight_meter: float = 1, **kwargs
    ):
        """
        Create a Gaussian frequency shape with the given peak frequency and significant wave height. The standard
        deviation of the Gaussian is set to 1/10 of the peak frequency (narrow), but may be overridden by the
        standard_deviation_hertz keyword argument.

        :param peak_frequency_hertz: Peak frequency of the spectrum in Hertz.
        :param significant_waveheight_meter: Significant waveheight defined as $4\\sqrt(m0)$, where m0 is the variance
        :param kwargs:
        """

        super(FreqGaussian, self).__init__(
            peak_frequency_hertz, significant_waveheight_meter, **kwargs
        )
        self.standard_deviation_hertz = kwargs.get(
            "standard_deviation_hertz", peak_frequency_hertz / 10
        )

    def _values(self, frequency_hertz: numpy.ndarray) -> numpy.ndarray:
        return (
            self.m0
            / self.standard_deviation_hertz
            / numpy.sqrt(2 * numpy.pi)
            * numpy.exp(
                -0.5
                * (frequency_hertz - self.peak_frequency_hertz) ** 2
                / self.standard_deviation_hertz**2
            )
        )


class FreqPhillips(FreqShape):
    """
    Phillips frequency shape as proposed by Phillips (1958).
    """

    def __init__(
        self, peak_frequency_hertz, significant_waveheight_meter: float = 1, **kwargs
    ):
        """
        Create a Phillips frequency shape with the given peak frequency and significant wave height.
        :param peak_frequency_hertz: peak frequency of the spectrum in Hertz.
        :param significant_waveheight_meter: significant waveheight defined as $4\\sqrt(m0)$, where m0 is the variance
        :param kwargs:
        """
        super(FreqPhillips, self).__init__(
            peak_frequency_hertz, significant_waveheight_meter, **kwargs
        )
        self._g = kwargs.get("g", GRAVITATIONAL_ACCELERATION)

    @property
    def _alpha(self):
        """
        Scale parameter of the Phillips spectrum.
        """
        return (
            self.m0
            * 8
            * (numpy.pi) ** 4
            * self.peak_frequency_hertz**4
            / self._g**2
        )

    def _values(self, frequency_hertz: numpy.ndarray) -> numpy.ndarray:
        """
        Phillips variance-density spectrum with frequency in Hz as
        dependent variable.

        :return:
        """
        values = numpy.zeros(len(frequency_hertz))
        msk = frequency_hertz >= self.peak_frequency_hertz
        values[msk] = (
            self._alpha
            * self._g**2
            * (2 * numpy.pi) ** -4
            * frequency_hertz[msk] ** -5
        )
        return values


class FreqPiersonMoskowitz(FreqShape):
    """
    Pierson Moskowitz frequency shape as proposed by Pierson and Moskowitz (1964). Commonly used for wind-generated
    waves in deep water.
    """

    def __init__(
        self, peak_frequency_hertz, significant_waveheight_meter: float = 1, **kwargs
    ):
        """
        Create a Pierson Moskowitz frequency shape with the given peak frequency and significant wave height.
        :param peak_frequency_hertz: peak frequency of the spectrum in Hertz.
        :param significant_waveheight_meter: significant waveheight defined as $4\\sqrt(m0)$, where m0 is the variance
        :param kwargs:
        """
        super(FreqPiersonMoskowitz, self).__init__(
            peak_frequency_hertz, significant_waveheight_meter, **kwargs
        )
        self._g = kwargs.get("g", GRAVITATIONAL_ACCELERATION)

    @property
    def _alpha(self):
        return (
            self.m0 * 5 * (2 * numpy.pi * self.peak_frequency_hertz) ** 4 / self._g**2
        )

    def _values(self, frequency_hertz: numpy.ndarray) -> numpy.ndarray:
        """
        Pierson Moskowitz variance-density spectrum with frequency in Hz as
        dependant variable. See e.g. Holthuijsen "Waves in Oceanic Water."

        :param frequency: frequency in Hz (scalar or array)
        :param peak_frequency: peak frequency in Hz
        :param alpha: Phillips constant (default 0.0081)
        :param g: gravitational acceleration (default 9.81)
        :return:
        """
        values = numpy.zeros(len(frequency_hertz))
        msk = frequency_hertz > 0
        values[msk] = (
            self._alpha
            * self._g**2
            * (2 * numpy.pi) ** -4
            * frequency_hertz[msk] ** -5
            * numpy.exp(
                -5 / 4 * (self.peak_frequency_hertz / frequency_hertz[msk]) ** 4
            )
        )
        return values


class FreqJonswap(FreqShape):
    """
    JONSWAP frequency shape as proposed by Hasselmann et al. (1973). Commonly used for wind-generated waves in deep
    water.
    """

    def __init__(
        self, peak_frequency_hertz, significant_waveheight_meter: float = 1, **kwargs
    ):
        """
        Create a JONSWAP frequency shape with the given peak frequency and significant wave height. The peakeness
        parameter $\\gamma$ is set to 3.3 by default, but may be overridden by the gamma keyword argument.
        :param peak_frequency_hertz: peak frequency of the spectrum in Hertz.
        :param significant_waveheight_meter: significant waveheight defined as $4\\sqrt(m0)$, where m0 is the variance
        :param kwargs:
        """
        super(FreqJonswap, self).__init__(
            peak_frequency_hertz, significant_waveheight_meter, **kwargs
        )
        self._g = kwargs.get("g", GRAVITATIONAL_ACCELERATION)
        self.gamma = kwargs.get("gamma", 3.3)

        # Hardcoded JONSWAP parameters until we have a better way of handling alpha
        self._sigma_a = 0.07  # kwargs.get("sigma_a", 0.07)
        self._sigma_b = 0.09  # kwargs.get("sigma_b", 0.09)

    @property
    def _alpha(self):
        # Approximation by Yamaguchi (1984), "Approximate expressions for integral properties of the JONSWAP
        # spectrum" Proc. Japanese Society of Civil Engineers, 345/II-1, 149–152 [in Japanese]. Taken from Holthuijsen
        # "waves in oceanic and coastal waters". Not valid if sigma_a or sigma_b are chanegd from defaults. Otherwise
        # accurate to within 0.25%
        #
        return (
            self.m0
            * (2 * numpy.pi * self.peak_frequency_hertz) ** 4
            / self._g**2
            / (0.06533 * self.gamma**0.8015 + 0.13467)
        )

    def _values(self, frequency_hertz: numpy.ndarray) -> numpy.ndarray:
        """
        Jonswap variance-density spectrum with frequency in Hz as
        dependant variable. See e.g. Holthuijsen "Waves in Oceanic Water."

        :param frequency: frequency in Hz (scalar or array)
        :param peak_frequency: peak frequency in Hz
        :param alpha: Phillips constant (default 0.0081)
        :param g: gravitational acceleration (default 9.81)
        :return:
        """
        values = numpy.zeros(len(frequency_hertz))
        msk = frequency_hertz > 0

        sigma = numpy.where(
            frequency_hertz <= self.peak_frequency_hertz, self._sigma_a, self._sigma_b
        )
        peak_enhancement = self.gamma ** numpy.exp(
            -1 / 2 * ((frequency_hertz / self.peak_frequency_hertz - 1) / sigma) ** 2
        )

        values[msk] = (
            self._alpha
            * self._g**2
            * (2 * numpy.pi) ** -4
            * frequency_hertz[msk] ** -5
            * numpy.exp(
                -5 / 4 * (self.peak_frequency_hertz / frequency_hertz[msk]) ** 4
            )
            * peak_enhancement[msk]
        )
        return values


def create_freq_shape(
    period: float,
    waveheight: float,
    shape_name: Literal["jonswap", "pm", "phillips", "gaussian"],
) -> FreqShape:
    """
    Create a frequency shape object.
    :param period: Period measure in seconds
    :param waveheight: Wave height
    :param shape_name: frequency shape, one of 'jonswap','pm','phillips','gaussian'
    :return: FreqShape object
    """
    if shape_name == "jonswap":
        return FreqJonswap(
            peak_frequency_hertz=1 / period, significant_waveheight_meter=waveheight
        )
    elif shape_name == "pm":
        return FreqPiersonMoskowitz(
            peak_frequency_hertz=1 / period, significant_waveheight_meter=waveheight
        )
    elif shape_name == "phillips":
        return FreqPhillips(
            peak_frequency_hertz=1 / period, significant_waveheight_meter=waveheight
        )
    elif shape_name == "gaussian":
        return FreqGaussian(
            peak_frequency_hertz=1 / period, significant_waveheight_meter=waveheight
        )
    else:
        raise ValueError(f"Unknown frequency shape {shape_name}")


def create_dir_shape(
    direction: float, spread: float, shape_name: Literal["cosN", "cos2N"]
) -> DirShape:
    """
    Create a directional shape object.
    :param direction: Mean direction
    :param spread: Directional spread
    :param shape_name: Directional shape, one of 'cosN','cos2N'
    :return: DirShape object
    """
    if shape_name == "cosN":
        return DirCosineN(mean_direction_degrees=direction, width_degrees=spread)
    elif shape_name == "cos2N":
        return DirCosine2N(mean_direction_degrees=direction, width_degrees=spread)
    else:
        raise ValueError(f"Unknown directional shape {shape_name}.")
