from roguewavespectrum._geospatial import contains
import numpy as np


def test_contains():
    # Polygon that crosses the 180 meridian
    polygon_0 = np.array([(0, 10), (0, 350), (1, 350), (1, 10), (0, 10)])

    # Same, but degrees from -180 to 180
    polygon_0b = np.array([(0, 10), (0, -10), (1, -10), (1, 10), (0, 10)])

    # Polygon that does not cross the 180 meridian, but has a longitudal span larger than 180 degrees.
    polygon_1 = np.array(
        [(0, 10), (0, 190), (0, 350), (1, 350), (1, 190), (1, 10), (0, 10)]
    )

    lat0 = 0.5
    lon0 = 179

    lat1 = 0.5
    lon1 = -1

    assert not contains(lat0, lon0, polygon_0)
    assert contains(lat1, lon1, polygon_0)

    assert not contains(lat0, lon0, polygon_0b)
    assert contains(lat1, lon1, polygon_0b)

    assert contains(lat0, lon0, polygon_1)
    assert not contains(lat1, lon1, polygon_1)
