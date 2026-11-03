# coding: utf-8

"""
Module for 2D geometry

It brings utilitary functions for various computation in a 2D euclidean space such as :
    - computing the distance between a point and a line
    - computing the projection of a point on a line
    - computing the projection of a point on a segment

You can download geom2d.tar.gz_ and install it with the following command:

.. code-block:: bash

    tar -zxvf geom2d.tar.gz
    python3 -m pip install geom2d

and then test if the module is correctly installed with the following command:

.. code-block:: bash

    python3 -c "import geom2d; print(geom2d.__version__)"

.. _geom2d.tar.gz: https://ia2vr.gitlabpages.inria.fr/iliar/_static/geom2d.tar.gz

"""

# Standard imports
# External imports
import numpy as np

__version__ = "0.1.0"


class Segment:
    """
    A segment is a subset of a line defined by two points

    Arguments:
     - pointA : numpy.ndarray
     - pointB : numpy.ndarray
    """

    def __init__(self, pointA: np.array, pointB: np.array):
        assert pointA.shape == (2,)
        assert pointB.shape == (2,)

        self.pointA = pointA
        self.pointB = pointB

    @staticmethod
    def from_origin_vector(origin: np.array, vector: np.array):
        """
        Builds a line from two points
        """
        return Segment(origin, origin + vector)

    @property
    def vector(self):
        """
        Computes and returns a unit vector along the segment
        """
        dp = self.pointB - self.pointA
        norm = np.linalg.norm(dp)
        return dp / norm


def distance_point_line(
    point: np.array, segment: Segment, forward_direction: np.array = None
):
    """
    Compute the distance between a point and a line to which the segment belongs

    Arguments
    ---------
    point : numpy.ndarray
        The point in 2D space
    segment : Segment
        The segment which is a subset of the line
    forward_direction : numpy.ndarray. Optional
                        If specified, the distance is signed with respect
                        to the forward direction

    Returns
    -------
    float
        The distance between the point and the line

    """
    # Normalize the line vector
    dp = point - segment.pointA

    # line vector is normalized, so the distance of the point
    # to the line is the cross product of the line vector and the origin -> point
    v = segment.vector
    cross_prod = dp[0] * v[1] - dp[1] * v[0]

    distance = np.abs(cross_prod)

    if forward_direction is not None:
        # Compute the point on the line that is the closest to the point
        projection = project_point_on_line(point, segment)

        # Compute the vector from the point to the projection
        v = projection - point

        # Compute the sign of the distance
        cross_prod = v[0] * forward_direction[1] - v[1] * forward_direction[0]
        sign = -np.sign(cross_prod)
        distance *= sign

    # Compute the distance
    return distance


def project_point_on_line(point: np.array, segment: Segment):
    """
    Compute the projection of a point on a line to which the segment belongs

    Parameters
    ----------
    point : numpy.ndarray
        The point in 2D space
    segment : Segment
        The segment which is a subset of the line

    Returns
    -------
    numpy.ndarray
        The projection of the point on the line

    """
    dp = point - segment.pointA
    return segment.pointA + np.dot(dp, segment.vector) * segment.vector


def project_point_on_segment(point: np.array, segment: np.array):
    """
    Compute the projection of a point on a segment

    Parameters
    ----------
    point : numpy.ndarray
        The point in 2D space
    segment : Segment
        The segment on which the projection is computed

    Returns
    -------
    numpy.ndarray
        The projection of the point on the segment

    """
    projection = project_point_on_line(point, segment)

    # Check that the projection is on the segment
    pa_proj = projection - segment.pointA
    pa_pb = segment.pointB - segment.pointA
    v = segment.vector
    lbd = np.dot(pa_proj, v)

    if lbd < 0:
        return segment.pointA
    elif lbd > np.linalg.norm(pa_pb):
        return segment.pointB
    else:
        return projection


def test_distance_point_line():
    """
    Test the distance_point_line function
    """
    line = Segment(np.array([0, 0]), np.array([1, 1]))
    point = np.array([1, 0])

    assert np.isclose(distance_point_line(point, line), 1 / np.sqrt(2))


def test_project_point_on_line():
    """
    Test the project_point_on_line function
    """
    line = Segment(np.array([0, 0]), np.array([1, 1]))
    point = np.array([1, 0])

    projection = project_point_on_line(point, line)
    assert np.allclose(projection, np.array([0.5, 0.5]))


def test_projection_point_on_line2():
    import matplotlib.pyplot as plt

    segment = Segment(np.array([0, 0]), np.array([2, 2]))

    plt.figure()
    for x in np.linspace(-5, 5):
        point = np.array([x, 1])
        projection = project_point_on_line(point, segment)
        plt.plot([point[0], projection[0]], [point[1], projection[1]], "r")
        plt.plot(point[0], point[1], "bo")
        plt.plot(projection[0], projection[1], "go")
    plt.plot(
        [segment.pointA[0], segment.pointB[0]],
        [segment.pointA[1], segment.pointB[1]],
        "k",
    )
    plt.gca().set_aspect("equal")
    plt.savefig("projection_point_on_line.png")
    # plt.show()


def test_project_point_on_segment():
    """
    Test the project_point_on_segment function
    """
    segment = Segment(np.array([0, 0]), np.array([1, 1]))
    point = np.array([1, 0])

    projection = project_point_on_segment(point, segment)

    # Check that the projection is on the segment
    pa_proj = projection - segment.pointA
    v = segment.vector
    lbd = np.dot(pa_proj, v)
    assert np.allclose(projection, segment.pointA + lbd * v)

    point = np.array([1, 2])
    projection = project_point_on_segment(point, segment)
    assert np.allclose(projection, segment.pointB)

    point = np.array([1, -1])
    projection = project_point_on_segment(point, segment)
    assert np.allclose(projection, segment.pointA)


def test_projection_point_on_segment2():
    import matplotlib.pyplot as plt

    segment = Segment(np.array([0, 0]), np.array([2, 2]))

    plt.figure()
    for x in np.linspace(-5, 5):
        point = np.array([x, 1])
        projection = project_point_on_segment(point, segment)
        plt.plot([point[0], projection[0]], [point[1], projection[1]], "r")
        plt.plot(point[0], point[1], "bo")
        plt.plot(projection[0], projection[1], "go")
    plt.plot(
        [segment.pointA[0], segment.pointB[0]],
        [segment.pointA[1], segment.pointB[1]],
        "k",
    )
    plt.gca().set_aspect("equal")
    plt.savefig("projection_point_on_segment.png")
    # plt.show()


if __name__ == "__main__":
    test_distance_point_line()
    test_project_point_on_line()
    test_projection_point_on_line2()
    test_project_point_on_segment()
    test_projection_point_on_segment2()
    print("All tests passed")
