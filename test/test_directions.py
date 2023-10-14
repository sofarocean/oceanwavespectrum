from roguewavespectrum.spectrum._directions import wave_directional_spread, wave_mean_direction, convert_unit, convert_angle_convention
import numpy as np


def test_convert_unit():
    assert convert_unit(180, "rad", "degree") == np.pi
    assert convert_unit(np.pi/2, "degree", "rad") == 90
    assert convert_unit(1, "rad", "rad") == 1
    assert convert_unit(1, "degree", "degree") == 1


def test_convert_angle_convention():
    assert convert_angle_convention(180,'nautical','mathematical', 'degree') == 270
    assert convert_angle_convention(np.pi, 'nautical', 'mathematical', 'rad') == 3 / 2  * np.pi
    assert convert_angle_convention(180, 'nautical', 'meteorological', 'degree') == 0
    assert convert_angle_convention(180, 'nautical', 'nautical', 'degree') == 180

    assert convert_angle_convention(180,'meteorological','mathematical', 'degree') == 90
    assert convert_angle_convention(np.pi, 'meteorological', 'mathematical', 'rad') == 1 / 2  * np.pi
    assert convert_angle_convention(180, 'meteorological', 'meteorological', 'degree') == 180
    assert convert_angle_convention(180, 'meteorological', 'nautical', 'degree') == 0

    assert convert_angle_convention(180,'mathematical','mathematical', 'degree') == 180
    assert convert_angle_convention(np.pi, 'mathematical', 'mathematical', 'rad') == np.pi
    assert convert_angle_convention(180, 'mathematical', 'meteorological', 'degree') == 90
    assert convert_angle_convention(180, 'mathematical', 'nautical', 'degree') == 270
