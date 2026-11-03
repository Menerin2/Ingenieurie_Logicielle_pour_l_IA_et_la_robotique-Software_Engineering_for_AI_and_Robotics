#! coding: utf-8

"""
Compute the distance between the robot and the path. The path is provided as a
sequence of control points. The distance is computed as the distance between
the robot and the segment connecting the two closest points of the path.

Subscriptions:
    - `/path` (nav_msgs/Path): The path to follow

Publications:
    - `/dist_to_path` (std_msgs/Float64): The distance to the path

Parameters:
    - `base_frame` (str): The base frame of the robot

"""

# Standard imports
import sys

# External imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from std_msgs.msg import Header
from tf2_ros.transform_listener import TransformListener
from tf2_ros.buffer import Buffer
from tf2_ros import LookupException
import numpy as np
from iliar_solution import geom2d

# from tf_transformations import euler_from_quaternion


class DistToPath(Node):

    def __init__(self):
        super().__init__("dist_to_path")

        # The base frame of the robot
        self.declare_parameter("base_frame", "prius")

        # Boolean to decide whether or not to compute a signed distance to
        # the path
        self.declare_parameter("signed_distance", False)

        # The TF objects required to request the transforms
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # TODO
        # You must instantiante your subscriber and publisher here
        # Subscriber to the path
        self.sub_path = self.create_subscription(
            Path,
            "/lanes/track",      
            self.dist_callback,
            10
        )

        # Publisher for distance
        self.pub_dist = self.create_publisher(
            Float64,
            "/dist_to_path",
            10
        )

    def dist_callback(self, msg):
        path = [pstamped.pose for pstamped in msg.poses]
        path_frame = msg.header.frame_id

        base_frame = self.get_parameter("base_frame").value
        signed_distance = self.get_parameter("signed_distance").value

        try:
            # Get the pose in the path frame of reference
            trans = self._tf_buffer.lookup_transform(
                path_frame, base_frame, rclpy.time.Time()
            )
            # self.get_logger().info(f"Got transform {trans}")
            object_position = trans.transform.translation
            q = trans.transform.rotation
            object_forward = np.array(
                [
                    q.w**2 + q.x**2 - q.y**2 - q.z**2,
                    2 * q.w * q.z + 2 * q.x * q.y,
                ]
            )
        except LookupException as e:
            self.get_logger().error("failed to get transform {} \n".format(repr(e)))
            return

        # Compute the two closest points of the path to the object
        fdist = (
            lambda posa, posb: (posa.x - posb.x) ** 2
            + (posa.y - posb.y) ** 2
            + (posa.z - posb.z) ** 2
        )

        # Determine the path point closest to the current pose of the car
        min_dist_2 = sys.float_info.max
        min_idx = 0
        for i, pose in enumerate(path):
            dist_2 = fdist(pose.position, object_position)
            if dist_2 < min_dist_2:
                min_dist_2 = dist_2
                min_idx = i

        # Get, from the path, the closest point and the previous one
        # Helpfull for computing the tangent to the curve
        closest_point = path[min_idx].position

        # The second closest point is either the point before or after the closest point
        prev_point = path[min_idx - 1 if min_idx != 0 else len(path) - 1].position
        next_point = path[min_idx + 1 if min_idx != len(path) - 1 else 0].position

        if fdist(prev_point, object_position) < fdist(next_point, object_position):
            second_closest_point = prev_point
        else:
            second_closest_point = next_point

        # Compute the distance between the object and the segment
        # connecting the two closest points

        # TODO
        # Construire le segment entre les deux points les plus proches sur la route        
        A = np.array([closest_point.x, closest_point.y])
        B = np.array([second_closest_point.x, second_closest_point.y])

        # Position de la voiture dans le même repère que le chemin
        C = np.array([object_position.x, object_position.y])

        # Segment OA
        segment = geom2d.Segment(A, B)

        # Dist entre C et OA
        proj = geom2d.project_point_on_segment(C, segment)
        distance = np.linalg.norm(C - proj)

        # Publication de la distance sur le topic /dist_to_path
        msg_out = Float64()
        msg_out.data = float(distance)
        self.pub_dist.publish(msg_out)




def main(args=None):
    """Instantiation node and class."""
    rclpy.init(args=args)

    # Create the node
    dist_to_path = DistToPath()

    # Run
    try:
        rclpy.spin(dist_to_path)
    except KeyboardInterrupt:
        pass

    # End

    rclpy.try_shutdown()


if __name__ == "__main__":
    main()