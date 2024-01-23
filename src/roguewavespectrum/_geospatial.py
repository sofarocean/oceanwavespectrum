import numpy as np
import matplotlib.path


def contains(point_lats, points_lons, polygon: np.ndarray):
    """
    Check if a geopspatial point is inside a geospatial polygon - accounting for crossing the 180 meridian. Note that
    we assume that the polygon is closed (i.e. the first and last vertices are the same). Further, the check if the
    point is inside the polygon is done using the matplotlib.path.Path.contains_points method with Euclidean geometry
    (not accounting for great circles).

    :param lon: longitude of the point in signed decimal degrees
    :param lat: latitude of the point in signed decimal degrees
    :param polygon: polygon vertices as a numpy array of shape (n, 2) where n is the number of vertices
    :return: True if the point is inside the polygon, False otherwise
    """

    # Unwrap the polygon longitudes to avoid issues with the 180 meridian
    polygon = polygon.copy()
    polygon[:, 1] = np.unwrap(polygon[:, 1], discont=180, period=360)

    # The new origin is the minimum longitude in the unwrapped polygon.
    new_origin = polygon[:, 1].min()
    polygon[:, 1] -= new_origin

    # Our polygon is now defined on the interval [0, 360]. But we need to redefine the points longitudes as well
    points_lons = (points_lons - new_origin) % 360

    points = np.column_stack((point_lats, points_lons))
    return matplotlib.path.Path(polygon, closed=True).contains_points(points)
