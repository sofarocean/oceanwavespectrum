"""
Maximum entropy method
======================
This module contains functions that implement the maximum entropy method by Lygre and Krogstadt (1986,JPO)  to estimate
the directional shape of the spectrum. Specifically if $\\theta$ is the angle and $D((\\theta)$ the directional
distribution we minimize
    $$\\int - D * log(D) \\, d\\theta$$

such that the resulting distribution $D(\\theta)$ reproduces the observed moments. I.e.
    $$\\int  D \\, d\\theta= 1$$
    $$\\int \\cos(\\theta) D \\, d\\theta = a_1 $$
    $$\\int \\sin(\\theta) D \\, d\\theta = b_1 $$
    $$\\int \\cos(2\\theta) D \\, d\\theta = a_2$$
    $$\\int \\sin(2\\theta) D \\, d\\theta = b_2$$
and
    $$ D(\\theta) \\geq 0$$

References:

    Lygre, A., & Krogstad, H. E. (1986). Maximum entropy estimation of the directional
    distribution in ocean wave spectra. Journal of Physical Oceanography, 16(12), 2052-2060.

"""
import numpy


def mem(
    directions_radians: numpy.ndarray,
    a1: numpy.ndarray,
    b1: numpy.ndarray,
    a2: numpy.ndarray,
    b2: numpy.ndarray,
    progress,
    **kwargs
) -> numpy.ndarray:
    """
    Estimate the directional distribution $D(\theta)$ of the waves at each frequency using the maximum entropy method.

    :param directions_radians: 1d array of wave directions in radians,
    length[number_of_directions]. (going to, anti-clockswise from east)

    :param a1: 1d array of cosine directional moment as function of frequency,
    length [number_of_frequencies]

    :param b1: 1d array of sine directional moment as function of frequency,
    length [number_of_frequencies]

    :param a2: 1d array of double angle cosine directional moment as function
    of frequency, length [number_of_frequencies]

    :param b2: 1d array of double angle sine directional moment as function of
    frequency, length [number_of_frequencies]

    :return: array with shape [number_of_frequencies,number_of_direction]
    representing the directional distribution of the waves at each frequency.

    Maximize the enthrophy of the solution with entrophy defined as:

           integrate log(D) over directions

    such that the resulting distribution D reproduces the observed moments.

    :return: Directional distribution as a numpy array

    Note that:
    d1 = a1; d2 =b1; d3 = a2 and d4=b2 in the defining equations 10.

    """

    number_of_frequencies = a1.shape[-1]
    number_of_points = a1.shape[0]

    directional_distribution = numpy.zeros(
        (number_of_points, number_of_frequencies, len(directions_radians))
    )

    for ipoint in range(0, number_of_points):
        progress.update(1)
        directional_distribution[ipoint, :, :] = _mem(
            directions_radians,
            a1[ipoint, :],
            b1[ipoint, :],
            a2[ipoint, :],
            b2[ipoint, :],
        )
    return directional_distribution


def _mem(
    directions_radians: numpy.ndarray,
    a1: numpy.ndarray,
    b1: numpy.ndarray,
    a2: numpy.ndarray,
    b2: numpy.ndarray,
) -> numpy.ndarray:
    """
    Private implementation of MEM; see mem() for details.
    """

    number_of_directions = len(directions_radians)

    c1 = a1 + 1j * b1
    c2 = a2 + 1j * b2
    #
    # Eq. 13 L&K86
    #
    Phi1 = (c1 - c2 * numpy.conj(c1)) / (1 - c1 * numpy.conj(c1))
    Phi2 = c2 - Phi1 * c1
    #
    e1 = numpy.exp(-directions_radians * 1j)
    e2 = numpy.exp(-directions_radians * 2j)

    numerator = 1 - Phi1 * numpy.conj(c1) - Phi2 * numpy.conj(c2)
    denominator = (
        numpy.abs(1 - Phi1[:, None] * e1[None, :] - Phi2[:, None] * e2[None, :]) ** 2
    )

    D = numpy.real(numerator[:, None] / denominator) / numpy.pi / 2

    # Normalize to 1. in discrete sense
    integralApprox = numpy.sum(D, axis=-1) * numpy.pi * 2.0 / number_of_directions
    D = D / integralApprox[:, None]

    return numpy.squeeze(D)
