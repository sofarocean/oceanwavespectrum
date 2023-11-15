from roguewavespectrum._directions import (
    convert_unit,
    convert_angle_convention,
)
import numpy as np


def test_convert_unit():
    assert convert_unit(180, "rad", "degree") == np.pi
    assert convert_unit(np.pi / 2, "degree", "rad") == 90
    assert convert_unit(1, "rad", "rad") == 1
    assert convert_unit(1, "degree", "degree") == 1


def test_convert_angle_convention():
    assert (
        convert_angle_convention(180, "oceanographical", "mathematical", "degree")
        == 270
    )
    assert (
        convert_angle_convention(np.pi, "oceanographical", "mathematical", "rad")
        == 3 / 2 * np.pi
    )
    assert (
        convert_angle_convention(180, "oceanographical", "meteorological", "degree")
        == 0
    )
    assert (
        convert_angle_convention(180, "oceanographical", "oceanographical", "degree")
        == 180
    )

    assert (
        convert_angle_convention(180, "meteorological", "mathematical", "degree") == 90
    )
    assert (
        convert_angle_convention(np.pi, "meteorological", "mathematical", "rad")
        == 1 / 2 * np.pi
    )
    assert (
        convert_angle_convention(180, "meteorological", "meteorological", "degree")
        == 180
    )
    assert (
        convert_angle_convention(180, "meteorological", "oceanographical", "degree")
        == 0
    )

    assert (
        convert_angle_convention(180, "mathematical", "mathematical", "degree") == 180
    )
    assert (
        convert_angle_convention(np.pi, "mathematical", "mathematical", "rad") == np.pi
    )
    assert (
        convert_angle_convention(180, "mathematical", "meteorological", "degree") == 90
    )
    assert (
        convert_angle_convention(180, "mathematical", "oceanographical", "degree")
        == 270
    )
