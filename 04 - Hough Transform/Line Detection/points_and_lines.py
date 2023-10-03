import numpy as np


def _ensure_shape(xy: np.ndarray) -> np.ndarray:
    # #print(xy.shape)
    # #print(xy)
    # #print(xy.ndim)
    # ?--> 1st array is 1 D, 2nd is 2D so they pass, 3rd array is 3D hence ValueError
    if xy.ndim == 1:

        # Change things of shape (2,) to shape (1, 2) so that they conform to the (m,2) specification in all functions
        # below.
        xy = xy[np.newaxis, :]
        # #print(xy.shape)
        # #print(xy.ndim)
        # #print(xy)
    elif xy.ndim >= 3:
        raise ValueError(
            f"xy must have 1 or 2 dimensions, but has {xy.ndim} dimensions"
        )
    return xy


def cart2pol(xy: np.ndarray) -> np.ndarray:
    """Given m points in the Cartesian plane, return the polar coordinates of those points.

    :param xy: numpy array of shape (m, 2) where each row is (x, y)
    :return: numpy array of shape (m, 2) where each row is (rho, theta)
    """
    #print("Cart2Pol")
    xy = _ensure_shape(xy)
    # getting x and y coordinates
    x, y = xy[:, 0], xy[:, 1]
    #print(x, y)
    # cartesian product  in an iterative manner
    rho = np.sqrt(x ** 2 + y ** 2)
    # angle in polar coordinate in an iterative manner
    theta = np.arctan2(y, x)
    #print(rho ,"\n", theta)
    #print(np.column_stack((rho, theta)))
    return np.column_stack((rho, theta))
    # ====



def pol2cart(rho_theta: np.ndarray) -> np.ndarray:
    """Given m points in the polar plane, return the Cartesian coordinates of those points.

    :param rho_theta: numpy array of shape (m, 2) where each row is (rho, theta)
    :return: numpy array of shape (m, 2) where each row is (x, y)
    """

    rho_theta = _ensure_shape(rho_theta)
    #separate for magnitude and polar angle in iterative manner
    rho, theta = rho_theta[:, 0], rho_theta[:, 1]
    # calculate the x and y axis in iterative manner
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    # send back the array with x and y together
    return np.column_stack((x, y))
    # ====



def unit_vector(v: np.ndarray):
    """Given m vectors in the Cartesian plane, return the unit vectors of those vectors.

    :param v: numpy array of shape (m, 2) where each row is (x, y)
    :return: numpy array of shape (m, 2) where each row is (x', y')
    """
    v = _ensure_shape(v)
    #print(v)
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    #print(norm)
    #print(v / norm)
    return v / norm



def point_on_line_closest_to_origin(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Given two points in the Cartesian plane, return the point on the line between those points which is closest to
    the origin. This is the projection of the origin onto the line. Vectorized to handle m pairs of points at once.

    :param a: numpy array of shape (m, 2) where each row is (x, y)
    :param b: numpy array of shape (m, 2) where each row is (x, y)
    :return: numpy array p of shape (m, 2) such that p[i,:] is the projection of the origin onto the line a[i,:]--b[i,:]
    """
    a, b = _ensure_shape(a), _ensure_shape(b)
    # vector along a and b
    ab = b - a
    #print(ab)
    #
    ao = -a
    #print(ao)
    # product and add x and y values
    dot_product = np.sum(ab * ao, axis=1, keepdims=True)
    #print(dot_product)
    #
    ab_length_squared = np.sum(ab * ab, axis=1, keepdims=True)
    #print(ab_length_squared)
    t = dot_product / ab_length_squared
    #print(t)
    closest_point = a + t * ab
    #print(closest_point)
    return closest_point


def distance_point_to_normal_line(rho_theta: np.ndarray, xy: np.ndarray):
    """Given a line in normal form (rho, theta) and a point in the Cartesian plane (x, y), return the distance from the
    point to the line. Vectorized to handle m lines and points at once.

    :param rho_theta: numpy array of shape (m, 2) where each row is (rho, theta) and specificies a line in normal form
    :param xy: numpy array of shape (m, 2) where each row is (x, y) and specifies a point in the Cartesian plane
    :return: numpy array of shape (m,) where each element is the distance from xy[i] to the line defined by rho_theta[i]
    """

    rho_theta, xy = _ensure_shape(rho_theta), _ensure_shape(xy)
    x, y = xy[:, 0], xy[:, 1]
    rho, theta = rho_theta[:, 0], rho_theta[:, 1]
    return np.abs(x * np.cos(theta) + y * np.sin(theta) - rho)



def get_all_lines_through_points(xy: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Given a set of m points xy in the Cartesian plane and an array of n angles, return a (m,n,2) array of lines in
    normal form. For example, if the output is stored in a variable 'lines', then lines[i,j,:] is (rho, theta) for the
    line through xy[i,:] at angle angles[j] (thus theta is equal to angles[j]).

    The definition of a line in normal form is

    .. math::
        \\rho = x \\cos(\\theta) + y \\sin(\\theta)

    This function essentially solves for rho in the above equation for each point in xy and each angle in angles.

    :param xy: numpy array of shape (m, 2) where each row is (x, y) and specifies a point in the Cartesian plane
    :param angles: numpy array of shape (n,) where each element is an angle in radians
    :return: numpy array of shape (m,n,2) where each [i,j,:] is (rho, theta) and specifies a line in normal form passing
        through xy[i,:] at angle angles[j]
    """

    xy = _ensure_shape(xy)
    x = xy[:, 0].reshape(-1, 1)  # Reshape to (m, 1)
    y = xy[:, 1].reshape(-1, 1)  # Reshape to (m, 1)

    # Calculate rho for all combinations of points and angles
    rho = x * np.cos(angles) + y * np.sin(angles)

    # combining the parameters
    param = np.zeros((xy.shape[0], angles.shape[0], 2))
    param[:, :, 0] = rho  # Set rho values
    param[:, :, 1] = angles  # Set angle values

    return param

    # iterative approach
    # xy = _ensure_shape(xy)
    # x, y = xy[:, 0], xy[:, 1]
    #
    # angles = angles.reshape(1, -1)
    #
    # rho = (x[:, np.newaxis] * np.cos(angles)).T
    #
    # rho1 = (y[:, np.newaxis] * np.sin(angles)).T
    # rho = rho +  rho1
    #
    # arrayx= []
    # for i in range(rho.shape[1]):
    #     arrayx.append(rho[:, i])
    #
    # array_comb = [array for array in arrayx]
    # # Create a list to store the repeated elements
    # result = []
    #
    # # Loop through the original_matrix and repeat the elements
    # for row in array_comb:
    #     repeated_row = []
    #     i = 0
    #
    #     for value in row:
    #         #print(i)
    #         repeated_row.append([value,angles[0][i] ])
    #         # repeated_row.extend()
    #         i += 1
    #     result.append(repeated_row)
    #
    # # Convert the result to a numpy array
    # result = np.array(result)
    #
    #
    # return result