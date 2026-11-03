#! coding: utf-8

"""
Follow a path using a PID controller

Subscriptions:
    - dist (std_msgs/Float64): The distance to the path we want to minimize.
    The PID will try to keep this distance to the target_dist_to_centerline

Publications:
    - throttle (std_msgs/Float64): The throttle command to send to the car
    - steering (std_msgs/Float64): The steering command to send to the car

Parameters:
    - target_dist_to_centerline (float): The distance we want to keep from the centerline of the path
    - Kp (float): The proportional gain of the PID
    - Kd (float): The derivative gain of the PID
    - Ki (float): The integral gain of the PID

"""

# Standard imports

# External imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64


class PID:
    """Simple PID implementation"""

    def __init__(self, gains):
        """
        Builds a PID

        Arguments:
            gains (dict): with keys Kp, Kd, Ki
        """
        self.gains = gains.copy()
        self.errors = {"error": 0.0, "d_error": 0.0, "i_error": 0.0, "time": None}
        if not ("Kp" in gains and "Kd" in gains and "Ki" in gains):
            raise RuntimeError(
                """Some keys are missing in the gains, we expect Kp, Kd and Ki"""
            )

    def reset(self):
        """
        Reset the errors of the PID
        """
        self.errors = {"error": 0.0, "d_error": 0.0, "i_error": 0.0, "time": None}

    def update(self, time: rclpy.time.Time, error: float):
        """
        Update the error, its derivative and integral
        """
        prev_error = self.errors["error"]
        prev_time = self.errors["time"]
        if prev_time is None:
            self.errors["error"], self.errors["time"] = error, time
            return False
        dt = (time - prev_time).nanoseconds * 1e-9
        if dt == 0:
            return False

        self.errors["error"] = error
        self.errors["time"] = time
        self.errors["i_error"] += dt * (prev_error + error) / 2.0
        self.errors["d_error"] = (error - prev_error) / dt
        return True

    @property
    def command(self):
        """
        Computes and returns the command of the PID
        """
        return (
            self.gains["Kp"] * self.errors["error"]
            + self.gains["Kd"] * self.errors["d_error"]
            + self.gains["Ki"] * self.errors["i_error"]
        )


class PathFollower(Node):
    """
    Follow a path using a PID controller
    """

    def __init__(self):
        super().__init__("path_follower")

        # TODO
        self.declare_parameter("target_dist_to_centerline", 0.0)
        # Adds the publisher and subscriber here

        # Paramtres PID
        self.declare_parameter("Kp", 1.0)
        self.declare_parameter("Ki", 0.1)
        self.declare_parameter("Kd", 0.1)

        # Initialisation PID
        gains = {
            "Kp": self.get_parameter("Kp").value,
            "Ki": self.get_parameter("Ki").value,
            "Kd": self.get_parameter("Kd").value,
        }
        self.pid = PID(gains)

        # Subscriber
        self.sub_error = self.create_subscription(
            Float64,
            "/dist_to_path",
            self.control_callback,
            10
        )

        # Publisher
        self.pub_steer = self.create_publisher(Float64, "/audibot/steering_cmd", 10)
        self.pub_throttle = self.create_publisher(Float64, "/audibot/throttle_cmd", 10)
        self.pub_brake = self.create_publisher(Float64, "/audibot/brake_cmd", 10)


    def control_callback(self, msg: Float64):
        """
        Callback to control the car triggered by an input message
        giving the distance to the path
        """

        target_dist_to_centerline = self.get_parameter(
            "target_dist_to_centerline"
        ).value

        # TODO
        # Given the received distance and the target distance
        # We must update the PID and send the computed commands to the car

        # Calcul  de la difference
        dist = msg.data
        error = dist - target_dist_to_centerline

        # Temps actuel
        time_now = self.get_clock().now()

        # Mise à jour du PID
        update = self.pid.update(time_now, error)

        # Ignorer le premier appel
        if not update:
            return

        # Calcul de la commande de braquage
        steering = self.pid.command

        # Braquage
        steering = max(min(steering, 0.5), -0.5)

        # Publication de la commande de dir
        msg_steer = Float64()
        msg_steer.data = steering
        self.pub_steer.publish(msg_steer)

        # Publier la commande d'accélération
        msg_throttle = Float64()
        msg_throttle.data = 0.3
        self.pub_throttle.publish(msg_throttle)

        # On enleve le frein
        msg_brake = Float64()
        msg_brake.data = 0.0
        self.pub_brake.publish(msg_brake)



def main(args=None):
    """Instantiation node and class."""
    rclpy.init(args=args)

    # Create the node
    path_follower = PathFollower()

    # Spin the node so the callback is called
    try:
        rclpy.spin(path_follower)
    except KeyboardInterrupt:
        pass

    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
