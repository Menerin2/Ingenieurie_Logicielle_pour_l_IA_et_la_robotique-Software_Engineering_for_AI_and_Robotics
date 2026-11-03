#! coding: utf-8

"""
This node computes the traveled distance of the robot along a path. 
The distance is computed from the projected position of the robot on the path

Subscriptions:
    - path (nav_msgs/Path): The path to follow

Publications:
    - traveled_dist (std_msgs/Float64): The traveled distance along the path
    - curv_abscissa (std_msgs/Float64): The curvilinear abscissa of the robot on the path

Parameters:
    - base_frame (string): The frame of the robot
"""

# External imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker
from tf2_ros.transform_listener import TransformListener
from tf2_ros.buffer import Buffer
from tf2_ros import LookupException

from iliar_solution import geom2d
import numpy as np


class TraveledDistance(Node):
    """
    This node computes the traveled distance of the robot along a path.
    The distance is computed from the projected position of the robot on the path
    """

    def __init__(self):
        super().__init__("traveled_distance")

        # The base frame of the robot
        self.declare_parameter("base_frame", "base_link")

        # The TF objects required to request the transforms
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self.last_valid_position = None
        self.total_traveled_distance = 0.0

        self.last_s = None
        self.path_lengths = None
        self.total_length = None
        self.s_vertices = None

        # TODO
        # You must instantiante your publisher and subscriber here
        self.create_subscription(Path, "path", self.dist_callback, 10)

        self.dist_pub = self.create_publisher(Float64, "traveled_dist", 10)
        self.curv_pub = self.create_publisher(Float64, "curv_abscissa", 10)
        self.pub_point = self.create_publisher(Marker, "closest_point", 1)

    def dist_callback(self, path_msg: Path):
        """Callback to compute the traveled distance along the path."""

        # Extract the 2D (x, y) positions and go into the numpy world
        path = [
            np.array([pstamped.pose.position.x, pstamped.pose.position.y])
            for pstamped in path_msg.poses
        ]
        path_frame = path_msg.header.frame_id

        # Compute curvilinear abscissa data once
        if self.path_lengths is None:
            self.path_lengths = []
            self.s_vertices = [0.0]

            for a, b in zip(path[:-1], path[1:]):
                L = np.linalg.norm(b - a)
                self.path_lengths.append(L)
                self.s_vertices.append(self.s_vertices[-1] + L)

            L = np.linalg.norm(path[0] - path[-1])
            self.path_lengths.append(L)
            self.s_vertices.append(self.s_vertices[-1] + L)

            self.total_length = self.s_vertices[-1]
            self.s_vertices = [s / self.total_length for s in self.s_vertices]

        # Get the base frame of the robot
        base_frame = self.get_parameter("base_frame").get_parameter_value().string_value

        try:
            # Get the pose in the path frame of reference
            trans = self._tf_buffer.lookup_transform(
                path_frame, base_frame, rclpy.time.Time()
            )
            # self.get_logger().info(f"Got transform {trans}")
            car_position = np.array(
                [trans.transform.translation.x, trans.transform.translation.y]
            )
        except LookupException as e:
            self.get_logger().error("failed to get transform {} \n".format(repr(e)))
            return

        # Iterate over the segments in order to determine the
        # projection of the car onto the polygon
        car_projection = None
        min_dist = float("inf")
        best_idx = None
        best_t = None

        for p1_idx, (p1, p2) in enumerate(zip(path[:-1], path[1:])):
            segment = geom2d.Segment(p1, p2)
            projected_point = geom2d.project_point_on_segment(car_position, segment)
            d = np.linalg.norm(car_position - projected_point)

            if d < min_dist:
                min_dist = d
                car_projection = projected_point
                best_idx = p1_idx
                best_t = np.linalg.norm(projected_point - p1) / np.linalg.norm(p2 - p1)

        # Do not forget the last segment
        p1 = path[-1]
        p2 = path[0]
        segment = geom2d.Segment(p1, p2)
        projected_point = geom2d.project_point_on_segment(car_position, segment)
        d = np.linalg.norm(car_position - projected_point)

        if d < min_dist:
            car_projection = projected_point
            best_idx = len(path) - 1
            best_t = np.linalg.norm(projected_point - p1) / np.linalg.norm(p2 - p1)

        s_current = (
            self.s_vertices[best_idx]
            + best_t * self.path_lengths[best_idx] / self.total_length
        )

        # TODO
        # Update the traveled distance
        # and publish it
        if self.last_valid_position is not None and self.last_s is not None:
            delta = np.linalg.norm(car_projection - self.last_valid_position)

            if (
                s_current > self.last_s
                or (self.last_s > 0.8 and s_current < 0.2)
            ):
                self.total_traveled_distance += delta

        self.last_valid_position = car_projection
        self.last_s = s_current

        msg = Float64()
        msg.data = self.total_traveled_distance
        self.dist_pub.publish(msg)

        s_msg = Float64()
        s_msg.data = s_current
        self.curv_pub.publish(s_msg)

        marker = Marker()
        marker.header.frame_id = path_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = car_projection[0]
        marker.pose.position.y = car_projection[1]
        marker.pose.position.z = 1.0
        marker.scale.x = 0.4
        marker.scale.y = 0.4
        marker.scale.z = 0.4
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        self.pub_point.publish(marker)


def main(args=None):
    """Instantiation node and class."""
    rclpy.init(args=args)

    # Create the node
    traveled_distance = TraveledDistance()

    # Spin the node
    try:
        rclpy.spin(traveled_distance)
    except KeyboardInterrupt:
        pass

    rclpy.try_shutdown()


if __name__ == "__main__":
    main()


