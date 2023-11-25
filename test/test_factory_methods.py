from roguewavespectrum._factory_methods import (
    create_spectrum1d,
    create_spectrum2d,
    create_parametric_spectrum1d,
    create_parametric_spectrum2d,
)
import numpy as np
from test._tools import run_input_tests


def test_create_spectrum1d():
    """
    Test that the create_spectrum1d factory method works as expected.
    """
    # Create some dummy 1D data
    f = np.linspace(0, 1, 100)
    variance_density = np.random.rand(100)
    a1 = np.random.rand(100)
    b1 = np.random.rand(100)
    a2 = np.random.rand(100)
    b2 = np.random.rand(100)

    # Create the spectrum
    _ = create_spectrum1d(f, variance_density, a1, b1, a2, b2)

    # Create the using named coordinate
    _ = create_spectrum1d([("frequency", f)], variance_density, a1, b1, a2, b2)

    # Create some dummy 2D data
    f = np.linspace(0, 1, 100)
    variance_density = np.random.rand(3, 2, 100)
    a1 = np.random.rand(3, 2, 100)
    b1 = np.random.rand(3, 2, 100)
    a2 = np.random.rand(3, 2, 100)
    b2 = np.random.rand(3, 2, 100)

    coordinates_all = [
        ("dim_0", np.arange(3)),
        ("dim_1", np.arange(2)),
        ("frequency", f),
    ]
    coordinates_wrong_order = [
        ("frequency", f),
        ("dim_0", np.arange(3)),
        ("dim_1", np.arange(2)),
    ]
    coordinates_wrong_size = [
        ("dim_0", np.arange(2)),
        ("dim_1", np.arange(2)),
        ("frequency", f),
    ]

    # Create the spectrum with just a numpy array, we should create default leading dimensions
    spectrum = create_spectrum1d(f, variance_density, a1, b1, a2, b2)
    assert spectrum.dims == ("dim_0", "dim_1", "frequency")

    # Create the spectrum with the frequency dimension specified
    _ = create_spectrum1d(coordinates_all[-1], variance_density, a1, b1, a2, b2)

    # Name all dimensions
    _ = create_spectrum1d(coordinates_all, variance_density, a1, b1, a2, b2)

    # Naming only a subset is not allowed
    try:
        _ = create_spectrum1d(coordinates_all[1:], variance_density, a1, b1, a2, b2)
    except ValueError:
        pass

    # The last coordinate should be frequency
    try:
        _ = create_spectrum1d(coordinates_wrong_order, variance_density, a1, b1, a2, b2)
    except ValueError:
        pass

    # Coordinate size much match size of variance_density shape
    try:
        _ = create_spectrum1d(coordinates_wrong_size, variance_density, a1, b1, a2, b2)
    except ValueError:
        pass


def test_create_spectrum2d():
    """
    Test that the create_spectrum1d factory method works as expected.
    """
    # Create some dummy 1D data
    f = np.linspace(0, 1, 100)
    _dir = np.linspace(0, 360, 36, endpoint=False)
    variance_density = np.random.rand(100, 36)

    # Create the spectrum
    _ = create_spectrum2d((f, _dir), variance_density)

    # Create the using named coordinate
    _ = create_spectrum2d([("frequency", f), ("direction", _dir)], variance_density)

    # Create some dummy ND data
    variance_density = np.random.rand(3, 2, 100, 36)

    coordinates_all = [
        ("dim_0", np.arange(3)),
        ("dim_1", np.arange(2)),
        ("frequency", f),
        ("direction", _dir),
    ]
    coordinates_wrong_order = [
        ("frequency", f),
        ("dim_0", np.arange(3)),
        ("dim_1", np.arange(2)),
        ("direction", _dir),
    ]
    coordinates_wrong_size = [
        ("dim_0", np.arange(2)),
        ("dim_1", np.arange(2)),
        ("frequency", f),
        ("direction", _dir),
    ]

    # Create the spectrum with just a numpy array, we should create default leading dimensions
    spectrum = create_spectrum2d((f, _dir), variance_density)
    assert spectrum.dims == ("dim_0", "dim_1", "frequency", "direction")

    # Create the spectrum with the frequency dimension specified
    _ = create_spectrum2d(coordinates_all[-2:], variance_density)

    # Name all dimensions
    _ = create_spectrum2d(coordinates_all, variance_density)

    # Naming only a subset is not allowed
    try:
        _ = create_spectrum2d(coordinates_all[1:], variance_density)
    except ValueError:
        pass

    # Frequency and direction should be the trailing dimensions.
    try:
        _ = create_spectrum2d(coordinates_wrong_order, variance_density)
    except ValueError:
        pass

    # Coordinate size much match size of variance_density shape
    try:
        _ = create_spectrum2d(coordinates_wrong_size, variance_density)
    except ValueError:
        pass


def test_create_parametric_spectrum1d():
    # Check if for normal input we get a spectrum
    freq = np.linspace(0, 1, 100)

    # Check for various different input we get expected behaviour, including errors
    testdata = {
        "test jonswap": {
            "kwargs": {
                "frequencies": freq,
                "waveheight": 1,
                "period": 0.1,
                "shape_name": "jonswap",
            },
            "errorstring": None,
        },
        "test pm": {
            "kwargs": {
                "frequencies": freq,
                "waveheight": 1,
                "period": 0.1,
                "shape_name": "pm",
            },
            "errorstring": None,
        },
        "test phillips": {
            "kwargs": {
                "frequencies": freq,
                "waveheight": 1,
                "period": 0.1,
                "shape_name": "phillips",
            },
            "errorstring": None,
        },
        "test gaussian": {
            "kwargs": {
                "frequencies": freq,
                "waveheight": 1,
                "period": 0.1,
                "shape_name": "phillips",
            },
            "errorstring": None,
        },
        "test name error": {
            "kwargs": {
                "frequencies": freq,
                "waveheight": 1,
                "period": 0.1,
                "shape_name": "ERROR",
            },
            "errorstring": "Unknown frequency shape ERROR",
        },
        "test vector input": {
            "kwargs": {
                "frequencies": freq,
                "waveheight": [1, 2],
                "period": [0.1, 0.2],
                "shape_name": "jonswap",
            },
            "errorstring": None,
        },
        "test different vector length error": {
            "kwargs": {
                "frequencies": freq,
                "waveheight": [0.1, 0.2],
                "period": 1,
                "shape_name": "jonswap",
            },
            "errorstring": "Waveheight and period should be equal length vectors, or scalars.",
        },
    }
    run_input_tests(create_parametric_spectrum1d, testdata)


def test_create_parametric_spectrum2d():
    # Check if for normal input we get a spectrum
    freq = np.linspace(0, 1, 100)

    # Check for various different input we get expected behaviour, including errors
    testdata = {
        "test jonswap cosN": {
            "kwargs": {
                "frequencies": freq,
                "waveheight": 1,
                "period": 0.1,
                "spread": 20,
                "direction": 45,
                "number_of_directions": 36,
                "frequency_shape_name": "jonswap",
                "direction_shape_name": "cosN",
            },
            "errorstring": None,
        },
        "test jonswap cos2N": {
            "kwargs": {
                "frequencies": freq,
                "waveheight": 1,
                "period": 0.1,
                "spread": 20,
                "direction": 45,
                "number_of_directions": 36,
                "frequency_shape_name": "jonswap",
                "direction_shape_name": "cos2N",
            },
            "errorstring": None,
        },
        "test name error": {
            "kwargs": {
                "frequencies": freq,
                "waveheight": 1,
                "period": 0.1,
                "spread": 20,
                "direction": 45,
                "number_of_directions": 36,
                "frequency_shape_name": "jonswap",
                "direction_shape_name": "ERROR",
            },
            "errorstring": "Unknown directional shape ERROR.",
        },
    }
    run_input_tests(create_parametric_spectrum2d, testdata)
