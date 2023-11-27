"""

Maximum entropy method 2
========================
This module contains functions that implements the MEM2 method (see Kim1995 and references therein) and is used to
estimate the directional distribution that maximizes the enthrophy of the solution with entrophy defined as
    $$\\int - D * log(D) \\, d\\theta$$

such that the resulting distribution $D(\theta)$ reproduces the observed moments. I.e.
    $$\\int  D \\, d\\theta= 1$$
    $$\\int \\cos(\\theta) D \\, d\\theta = a_1 $$
    $$\\int \\sin(\\theta) D \\, d\\theta = b_1 $$
    $$\\int \\cos(2\\theta) D \\, d\\theta = a_2$$
    $$\\int \\sin(2\\theta) D \\, d\\theta = b_2$$
and
    $$ D(\\theta) \\geq 0$$

References:

    Kim, T., Lin, L. H., & Wang, H. (1995). Application of maximum entropy method
    to the real sea data. In Coastal Engineering 1994 (pp. 340-355).

    link: https://icce-ojs-tamu.tdl.org/icce/index.php/icce/article/download/4967/4647
    (working as of May 29, 2022)

"""

import numpy as np
from scipy.optimize import root
import typing
from numba import njit, prange
from numba.typed import Dict as NumbaDict
from numba.core import types
from numpy.linalg import norm
from numba_progress import ProgressBar

# Settings for numba JIT compilation- whether to use fast math and parallel optimizations when possible.
_FASTMATH = True
_PARALLEL = False

# Numerical settings used in solving for the mem2 distribution
_NUMERICS = {
    # absolute tolerence stopping criterium. let moment = [ a1,b1,a2,b2] and let iterate_moment contain the moments
    # calculated from the current estmitaed distribution. The stopping criterium is:
    #     norm( moment-iterate_moment ) < atol
    "atol": 0.01,
    # Maximum number of iterations
    "max_iter": 100,
    # Maximum number of subiterations in the line search algorithm. Typically deep line search activates only when
    # the convergence is poor anyway.
    "max_line_search_depth": 8,
    # If we fall back to least squares estimate of the newton update we have an ill-conditioned system, and solve
    # the system approximately removing the smallest singular values. rcond it the ration of smallest divided by largest
    # singular value.
    "rcond": 1e-6,
    # Convergence is mostly (based on limited testing) poor for narrow distributions (large lagrange multipliers). If
    # we fail to converge we fall back to the mem estimate which has no such issues. For narrow distributions this is
    # hopefully fine.
    "use_mem_when_failing_to_converge": True,
}

# Entry Function
# =============================================================================


def mem2(
    directions_radians: np.ndarray,
    a1: np.ndarray,
    b1: np.ndarray,
    a2: np.ndarray,
    b2: np.ndarray,
    progress_bar: ProgressBar = None,
    solution_method="newton",
    solver_config=None,
) -> np.ndarray:
    """
    Estimate the directional distribution from the Fourier moments using the MEM2 method.

    :param directions_radians: 1d array of wave directions in radians,
    length[number_of_directions]

    :param a1: 1d array of cosine directional moment as function of position and frequency,
        shape = ( number_of_points,number_of_frequencies)

    :param b1: 1d array of sine directional moment as function of position and frequency,
        shape = ( number_of_points,number_of_frequencies)

    :param a2: 1d array of double angle cosine directional moment as function of position and frequency,
        shape = ( number_of_points,number_of_frequencies)

    :param b2: 1d array of double angle sine directional moment as function of position and frequency,
        shape = ( number_of_points,number_of_frequencies)

    :param progress_bar: Progress bar instance if updates are desired.

    :param solution_method: Method used to solve the nonlinear system of equations. Can be one of: "scipy", "newton".
    The scipy method is a wrapper around scipy.optimize.root. The newton method is a custom implementation of the
    newton method in numba. The newton method is faster than the scipy method but occasionally fails to converge. When
    this happens the method falls back to the MEM method.

    :param solver_config: Dictionary of solver settings. See _NUMERICS for default values.

    :return: array with shape [numbet_of_points, number_of_frequencies,number_of_direction]
    representing the directional distribution of the waves at each frequency.


    """

    if solver_config is None:
        solver_config = _NUMERICS

    else:
        solver_config = _NUMERICS | solver_config

    if solution_method == "scipy":
        func = _mem2_scipy_root_finder
        kwargs = {}

    elif solution_method == "newton":
        func = _mem2_newton
        numba_solver_config = NumbaDict.empty(
            key_type=types.unicode_type, value_type=types.float64
        )
        for key in solver_config:
            numba_solver_config[key] = solver_config[key]

        kwargs = {"config": numba_solver_config}

    elif solution_method == "approximate":
        func = _mem2_newton
        kwargs = {"approximate": True}

    else:
        raise ValueError("Unknown method")

    return func(directions_radians, a1, b1, a2, b2, progress_bar, **kwargs)


# Scipy Implementation
# =============================================================================
def _mem2_scipy_root_finder(
    directions_radians: np.ndarray,
    a1: typing.Union[np.ndarray, float],
    b1: typing.Union[np.ndarray, float],
    a2: typing.Union[np.ndarray, float],
    b2: typing.Union[np.ndarray, float],
    progress_bar,
    **kwargs
) -> np.ndarray:
    """
    Return the directional distribution that maximizes Shannon [ - D log(D) ]
    enthrophy constrained by given observed directional moments,

    :param directions_radians: 1d array of wave directions in radians,
    length[number_of_directions]

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

           integrate - D * log(D) over directions

    such that the resulting distribution D reproduces the observed moments.

    """

    number_of_frequencies = a1.shape[-1]
    number_of_points = a1.shape[0]

    directional_distribution = np.zeros(
        (number_of_points, number_of_frequencies, len(directions_radians))
    )

    direction_increment = _get_direction_increment(directions_radians)

    twiddle_factors = np.empty((4, len(directions_radians)))
    twiddle_factors[0, :] = np.cos(directions_radians)
    twiddle_factors[1, :] = np.sin(directions_radians)
    twiddle_factors[2, :] = np.cos(2 * directions_radians)
    twiddle_factors[3, :] = np.sin(2 * directions_radians)

    guess = _initial_value(a1, b1, a2, b2)
    for ipoint in range(0, number_of_points):
        if progress_bar is not None:
            progress_bar.update(1)

        for ifreq in range(0, number_of_frequencies):
            #
            moments = np.array(
                [
                    a1[ipoint, ifreq],
                    b1[ipoint, ifreq],
                    a2[ipoint, ifreq],
                    b2[ipoint, ifreq],
                ]
            )

            if np.any(np.isnan(guess[ipoint, ifreq, :])):
                continue

            res = root(
                _moment_constraints,
                guess[ipoint, ifreq, :],
                args=(twiddle_factors, moments, direction_increment),
                method="lm",
            )
            lambas = res.x

            directional_distribution[ipoint, ifreq, :] = _mem2_directional_distribution(
                lambas, direction_increment, twiddle_factors
            )

    return directional_distribution


# Numba Implementation
# =============================================================================


# To note; enabling caching seems to not play nice with paralel
@njit(parallel=_PARALLEL, cache=(not _PARALLEL))
def _mem2_newton(
    directions_radians: np.ndarray,
    a1: np.ndarray,
    b1: np.ndarray,
    a2: np.ndarray,
    b2: np.ndarray,
    progress_bar: ProgressBar = None,
    config: NumbaDict = None,
    approximate: bool = False,
) -> np.ndarray:
    """
    Return the directional distribution that maximizes Shannon [ - D log(D) ]
    enthrophy constrained by given observed directional moments.

    :param directions_radians: 1d array of wave directions in radians,
    length[number_of_directions]

    :param a1: 1d array of cosine directional moment as function of position and frequency,
        shape = ( number_of_points,number_of_frequencies)

    :param b1: 1d array of sine directional moment as function of position and frequency,
        shape = ( number_of_points,number_of_frequencies)

    :param a2: 1d array of double angle cosine directional moment as function of position and frequency,
        shape = ( number_of_points,number_of_frequencies)

    :param b2: 1d array of double angle sine directional moment as function of position and frequency,
        shape = ( number_of_points,number_of_frequencies)

    :param progress_bar: Progress bar instance if updates are desired.

    :return: array with shape [numbrt_of_points, number_of_frequencies,number_of_direction]
    representing the directional distribution of the waves at each frequency.

    Maximize the enthrophy of the solution with entrophy defined as:

           integrate - D * log(D) over directions

    such that the resulting distribution D reproduces the observed moments.

    """

    number_of_frequencies = a1.shape[-1]
    number_of_points = a1.shape[0]

    directional_distribution = np.zeros(
        (number_of_points, number_of_frequencies, len(directions_radians))
    )

    direction_increment_downward_difference = (
        directions_radians - np.roll(directions_radians, 1) + np.pi
    ) % (2 * np.pi) - np.pi

    direction_increment_upward_difference = (
        -(directions_radians - np.roll(directions_radians, -1) + np.pi) % (2 * np.pi)
        - np.pi
    )

    direction_increment = (
        direction_increment_downward_difference + direction_increment_upward_difference
    ) / 2

    # Calculate the needed Fourier transform twiddle factors to calculate moments.
    twiddle_factors = np.empty((4, len(directions_radians)))
    twiddle_factors[0, :] = np.cos(directions_radians)
    twiddle_factors[1, :] = np.sin(directions_radians)
    twiddle_factors[2, :] = np.cos(2 * directions_radians)
    twiddle_factors[3, :] = np.sin(2 * directions_radians)

    guess = _initial_value(a1, b1, a2, b2)
    for ipoint in prange(0, number_of_points):
        if progress_bar is not None:
            progress_bar.update(1)

        # Note; entries to directional_distribution[ipoint, :, :] is modified in the call below. This avoids creation
        # of memory for the resulting array at the expense of allowing for side-effects.
        _mem2_newton_point(
            directional_distribution[ipoint, :, :],
            a1[ipoint, :],
            b1[ipoint, :],
            a2[ipoint, :],
            b2[ipoint, :],
            guess[ipoint, :, :],
            direction_increment,
            twiddle_factors,
            config,
            approximate,
        )

    return directional_distribution


# frequency iteration
# ----------------------


@njit(cache=True)
def _mem2_newton_point(
    out,
    a1,
    b1,
    a2,
    b2,
    guess,
    direction_increment,
    twiddle_factors,
    config=None,
    approximate=False,
):
    """

    :param out: a (view) of the array that will containt the output
    :param a1: 1d array of cosine directional moment as function of frequency,
    :param b1: 1d array of sine directional moment as function of frequency,
    :param a2: 1d array of double angle cosine directional moment as function of frequency,
    :param b2: 1d array of double angle sine directional moment as function of frequency,
    :param guess: initial guess of the lagrange multipliers
    :param direction_increment: directional stepsize used in the integration, nd-array
    :param twiddle_factors: [sin theta, cost theta, sin 2*theta, cos 2*theta] as a 4 by ndir array
    :param config: numerical settings, see description at NUMERICS at top of file.
    :param approximate: whether or not to use the approximate relations.
    :return: None - we use side-effects to pass the results back to the caller (modifying out)
    """
    number_of_frequencies = a1.shape[0]
    for ifreq in range(0, number_of_frequencies):
        #
        moments = np.array([a1[ifreq], b1[ifreq], a2[ifreq], b2[ifreq]])
        out[ifreq, :] = _mem2_newton_solver(
            moments,
            guess[ifreq, :],
            direction_increment,
            twiddle_factors,
            config,
            approximate,
        )


# mem2 numerical solver
# ----------------------


@njit(cache=True)
def _mem2_newton_solver(
    moments: np.ndarray,
    guess: np.ndarray,
    direction_increment: np.ndarray,
    twiddle_factors: np.ndarray,
    config=None,
    approximate=False,
) -> np.ndarray:
    """
    Newton iteration to find the solution to the non-linear system of constraint equations defining the lagrange
    multipliers in the MEM2 method. Because the Lagrange multipliers enter the equations as exponents the system can
    be unstable to solve numerically.

    :param moments: the normalized directional moments [a1,b1,a2,b2]
    :param guess: first guess for the lagrange multipliers (ndarray, length 4)
    :param direction_increment: directional stepsize used in the integration, nd-array
    :param twiddle_factors: [sin theta, cost theta, sin 2*theta, cos 2*theta] as a 4 by ndir array
    :param config: numerical settings, see description at NUMERICS at top of file.
    :param approximate: whether or not to use the approximate relations.
    :return:
    """
    if config is None:
        max_iter = 100
        rcond = 1e-6
        atol = 0.01
        max_line_search_depth = 8
        use_mem_when_failing_to_converge = True

    else:
        max_iter = config["max_iter"]
        rcond = config["rcond"]
        atol = config["atol"]
        max_line_search_depth = config["max_line_search_depth"]
        use_mem_when_failing_to_converge = (
            config["use_mem_when_failing_to_converge"] > 0.0
        )

    directional_distribution = np.empty(len(direction_increment))
    if np.any(np.isnan(guess)):
        directional_distribution[:] = 0
        return directional_distribution

    if approximate:
        directional_distribution[:] = _mem2_directional_distribution(
            guess, direction_increment, twiddle_factors
        )
        return directional_distribution

    current_iterate = guess
    current_func = _moment_constraints(
        current_iterate,
        twiddle_factors,
        moments,
        direction_increment,
    )

    jacobian = np.empty((4, 4))

    convergence = False
    for iter in range(0, max_iter):

        # Stopping criterium
        magnitude_cur_func_eval = norm(current_func)
        if magnitude_cur_func_eval < atol:
            convergence = True
            break

        #
        # Compute jacobian, and find newton iterate innovation as we solve for:
        #
        #       jacobian @ delta = - current_iterate_func_eval
        #
        # with:
        #
        #       delta = next_lagrange_multiplier_iterate-cur_lagrange_multiplier_iterate

        jacobian = _mem2_jacobian(
            current_iterate, twiddle_factors, direction_increment, jacobian
        )
        try:
            update_iterate = _solve_cholesky(jacobian, -current_func)
        except Exception:
            update_iterate = np.linalg.lstsq(jacobian, -current_func, rcond=rcond)[0]

        magnitude_current_iterate = norm(current_iterate)
        magnitude_update = norm(update_iterate)

        # Do a line search for the optimum decrease. This is intended to stabilize the algorithm
        # as the equations are ill-posed.
        line_search_factor = 1
        for ii in range(max_line_search_depth):
            next_iterate = current_iterate + line_search_factor * update_iterate
            next_func = _moment_constraints(
                next_iterate, twiddle_factors, moments, direction_increment
            )

            if norm(next_func) < magnitude_cur_func_eval:
                # If we are decreasing- continue
                current_func = next_func
                current_iterate = next_iterate
                break
            else:
                # The update may be too big as we are not decreasing the cost function magnitude. We will decrease the
                # step size we take - but keep the direction of the step the same.
                if magnitude_update == 0.0:
                    # We are stuck at a stationary point. We are done.
                    convergence = False
                    break
                inverse_relative_update = magnitude_current_iterate / magnitude_update
                line_search_factor = min(
                    inverse_relative_update, line_search_factor / 2
                )
        else:
            # The linesearch failed. We could not find a factor that ensures the next function estimate is closer
            # to 0.
            convergence = False
            break
    else:
        # We failed to converge after the maximum number of iterations.
        convergence = False

    if not convergence:
        if use_mem_when_failing_to_converge:
            directions = np.arctan2(twiddle_factors[1, :], twiddle_factors[0, :])
            directional_distribution[:] = _numba_mem(
                directions, moments[0], moments[1], moments[2], moments[3]
            )
        else:
            raise ValueError("we did not converge")

    directional_distribution[:] = _mem2_directional_distribution(
        current_iterate, direction_increment, twiddle_factors
    )

    return directional_distribution


# mem2 functions
# ----------------------


@njit(cache=True, fastmath=_FASTMATH)
def _moment_constraints(lambdas, twiddle_factors, moments, direction_increment):
    """
    Construct the nonlinear equations we need to solve for lambda. The constrainst are the difference between the
    desired moments a1,b1,a2,b2 and the moment calculated from the current distribution guess and for a perfect fit
    should be 0.

    To note: we differ from Kim et al here who formulate the constraints using unnormalized equations. Here we opt to
    use the normalized version as that allows us to cast the error / or mismatch directly in terms of an error in the
    moments.

    :param lambdas: the lagrange multipliers
    :param twiddle_factors: [sin theta, cost theta, sin 2*theta, cos 2*theta] as a 4 by ndir array
    :param moments: [a1,b1,a2,b2]
    :param direction_increment: directional stepsize used in the integration, nd-array
    :return: array (length=4) with the difference between desired moments and those calculated from the current
        approximate distribution
    """

    # Get the current estimate of the directional distribution
    dist = _mem2_directional_distribution(lambdas, direction_increment, twiddle_factors)
    out = np.zeros(4)
    for mm in range(0, 4):
        # note - the part after the "-" is just a discrete approximation of the Fourier sine/cosine amplitude (moment)
        out[mm] = moments[mm] - np.sum(
            (twiddle_factors[mm, :]) * dist * direction_increment
        )

    return out


@njit(cache=True, fastmath=_FASTMATH)
def _mem2_jacobian(lagrange_multiplier, twiddle_factors, direction_increment, jacobian):
    """
    Calculate the jacobian of the constraint equations. The resulting jacobian is a square and positive definite matrix

    :param lambdas: the lagrange multipliers
    :param twiddle_factors: [sin theta, cost theta, sin 2*theta, cos 2*theta] as a 4 by ndir array
    :param direction_increment: directional stepsize used in the integration, nd-array

    :return: a 4 by 4 matrix that is the Jacobian of the constraint equations.
    """
    inner_product = np.zeros(twiddle_factors.shape[1])
    for jj in range(0, 4):
        inner_product = inner_product + lagrange_multiplier[jj] * twiddle_factors[jj, :]

    # We subtract the minimum to ensure that the values in the exponent do not become too large. This amounts to
    # multiplyig with a constant - which is fine since we normalize anyway. Effectively- this avoids overflow errors
    # (or infinities) - at the expense of underflowing (which is less of an issue).
    #
    inner_product = inner_product - np.min(inner_product)

    normalization = 1 / np.sum(np.exp(-inner_product) * direction_increment)
    shape = np.exp(-inner_product)

    normalization_derivative = np.zeros(4)
    for mm in range(0, 4):
        normalization_derivative[mm] = normalization * np.sum(
            twiddle_factors[mm, :] * np.exp(-inner_product) * direction_increment
        )

    # To note- we have to multiply seperately to avoid potential underflow/overflow errors.
    normalization_derivative = normalization_derivative * normalization

    shape_derivative = np.zeros((4, twiddle_factors.shape[1]))
    for mm in range(0, 4):
        shape_derivative[mm, :] = -twiddle_factors[mm, :] * shape

    for mm in range(0, 4):
        # we make use of symmetry and only explicitly calculate up to the diagonal
        for nn in range(0, mm + 1):
            jacobian[mm, nn] = -np.sum(
                twiddle_factors[mm, :]
                * direction_increment
                * (
                    normalization * shape_derivative[nn, :]
                    + shape * normalization_derivative[nn]
                ),
                -1,
            )
            if nn != mm:
                jacobian[nn, mm] = jacobian[mm, nn]
    return jacobian


@njit(cache=True, fastmath=_FASTMATH)
def _mem2_directional_distribution(
    lagrange_multiplier,
    direction_increment,
    twiddle_factors,
) -> np.ndarray:
    """
    Given the solution for the Lagrange multipliers- reconstruct the directional
    distribution.
    :param lagrange_multiplier: the lagrange multipliers
    :param twiddle_factors: [sin theta, cost theta, sin 2*theta, cos 2*theta] as a 4 by ndir array
    :param direction_increment: directional stepsize used in the integration, nd-array
    :return: Directional distribution arrasy as a function of directions
    """
    inner_product = np.zeros(twiddle_factors.shape[1])
    for jj in range(0, 4):
        inner_product = inner_product + lagrange_multiplier[jj] * twiddle_factors[jj, :]

    inner_product = inner_product - np.min(inner_product)

    normalization = 1 / np.sum(np.exp(-inner_product) * direction_increment)
    return np.exp(-inner_product) * normalization


@njit(cache=True, fastmath=_FASTMATH)
def _initial_value(a1: np.ndarray, b1: np.ndarray, a2: np.ndarray, b2: np.ndarray):
    """
    Initial guess of the Lagrange Multipliers according to the "MEM AP2" approximation
    found im Kim1995

    :param a1: moment a1
    :param b1: moment b1
    :param a2: moment a2
    :param b2: moment b2
    :return: initial guess of the lagrange multipliers, with the same leading dimensions as input.
    """
    guess = np.empty((*a1.shape, 4))
    fac = 1 + a1**2 + b1**2 + a2**2 + b2**2
    guess[..., 0] = 2 * a1 * a2 + 2 * b1 * b2 - 2 * a1 * fac
    guess[..., 1] = 2 * a1 * b2 - 2 * b1 * a2 - 2 * b1 * fac
    guess[..., 2] = a1**2 - b1**2 - 2 * a2 * fac
    guess[..., 3] = 2 * a1 * b1 - 2 * b2 * fac
    return guess


@njit(cache=True, fastmath=True)
def _solve_cholesky(matrix, rhs):
    """
    Solve using cholesky decomposition according to the Cholesky–Banachiewicz algorithm.
    See: https://en.wikipedia.org/wiki/Cholesky_decomposition#The_Cholesky_algorithm
    """
    M, N = matrix.shape
    x = np.zeros(M)
    cholesky_decomposition = np.zeros((M, M))
    inv = np.zeros(M)

    for mm in range(0, M):
        forward_sub_sum = rhs[mm]
        for nn in range(0, mm):
            sum = matrix[mm, nn]
            for kk in range(0, nn):
                sum -= cholesky_decomposition[mm, kk] * cholesky_decomposition[nn, kk]

            cholesky_decomposition[mm, nn] = inv[nn] * sum
            forward_sub_sum += -cholesky_decomposition[mm, nn] * x[nn]

        sum = matrix[mm, mm]
        for kk in range(0, mm):
            sum -= cholesky_decomposition[mm, kk] ** 2

        if sum <= 0.0:
            raise ValueError(
                "Matrix not positive definite, likely due to finite precision errors."
            )

        cholesky_decomposition[mm, mm] = np.sqrt(sum)
        inv[mm] = 1 / cholesky_decomposition[mm, mm]
        x[mm] = forward_sub_sum * inv[mm]

    # Backward Substitution (in place)
    for mm in range(0, M):
        kk = M - mm - 1
        sum = x[kk]
        for nn in range(kk + 1, N):
            sum += -cholesky_decomposition[nn, kk] * x[nn]
        x[kk] = sum * inv[kk]
    return x


def _get_direction_increment(directions_radians: np.ndarray) -> np.ndarray:
    """
    calculate the stepsize used for midpoint integration. The directions
    represent the center of the interval - and we want to find the dimensions of
    the interval (difference between the preceeding and succsesive midpoint).

    :param directions_radians: array of radian directions
    :return: array of radian intervals
    """

    # Calculate the forward difference appending the first entry to the back
    # of the array. Use modular trickery to ensure the angle is in [-pi,pi]
    forward_diff = (
        np.diff(directions_radians, append=directions_radians[0]) + np.pi
    ) % (2 * np.pi) - np.pi

    # Calculate the backward difference prepending the last entry to the front
    # of the array. Use modular trickery to ensure the angle is in [-pi,pi]
    backward_diff = (
        np.diff(directions_radians, prepend=directions_radians[-1]) + np.pi
    ) % (2 * np.pi) - np.pi

    # The interval we are interested in is the average of the forward and backward
    # differences.
    return (forward_diff + backward_diff) / 2


@njit(cache=True)
def _numba_mem(
    directions_radians: np.ndarray,
    a1: float,
    b1: float,
    a2: float,
    b2: float,
) -> np.ndarray:
    """
    Numba implementation of the MEM function. We re-implement the MEM function here because we need to call it from
    within a numba jit function.

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

    number_of_directions = len(directions_radians)

    c1 = a1 + 1j * b1
    c2 = a2 + 1j * b2
    #
    # Eq. 13 L&K86
    #
    Phi1 = (c1 - c2 * np.conj(c1)) / (1 - c1 * np.conj(c1))
    Phi2 = c2 - Phi1 * c1
    #
    e1 = np.exp(-directions_radians * 1j)
    e2 = np.exp(-directions_radians * 2j)

    numerator = 1 - Phi1 * np.conj(c1) - Phi2 * np.conj(c2)
    denominator = np.abs(1 - Phi1 * e1 - Phi2 * e2) ** 2

    D = np.real(numerator / denominator) / np.pi / 2

    # Normalize to 1. in discrete sense
    integralApprox = np.sum(D, axis=-1) * np.pi * 2.0 / number_of_directions
    D = D / integralApprox

    return D
