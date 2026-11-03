"""Définition d'un noeud qui permet la téléopération via une manette, 
de son lancement et des tests associés.

Pour le mappong des touches: voir le fichier CommandsJoycon.txt (!! A modifer !!)
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Int32, Float64, UInt8
from sensor_msgs.msg import Joy


"""Classe créant un noeud qui permet la téléopération via une manette

Publie sur les topics:
- 'throttle_cmd' (std_msgs.msg.Float32) : commande de vitesse
- 'steering_cmd' (std_msgs.msg.Float32) : commande de direction
"""

class TeleopNode(Node):
    
    joy_sub = None
    steering_pub = None
    throttle_pub = None
    max_steer = None
    max_throttle = None
    max_brake = None
    steering_axis = None
    throttle_axis = None
    deadman_button = 4
    drive_button = 0
    reverse_button = 1

    def __init__(self, max_steer=90.0, max_throttle=1.0, max_brake=8000, steering_axis=0, movement_axis=4):
        super().__init__('teleopNode')
        self.joy_sub = self.create_subscription(Joy, 'joy', self.joy_cb, 1)
        self.steering_pub = self.create_publisher(Float64, 'teleop_steering', 1)
        self.forward_pub = self.create_publisher(Float64, 'teleop_throttle', 1)
        self.stop_pub = self.create_publisher(Float64, 'teleop_brake', 1)
        self.gear_pub = self.create_publisher(UInt8, 'teleop_gear', 1)
        self.max_steer = max_steer
        self.max_throttle = max_throttle
        self.max_brake = max_brake
        self.steering_axis = steering_axis
        self.movement_axis = movement_axis
        self.current_gear = 0
        self.last_drive = 0
        self.last_reverse = 0

    def joy_cb(self, msg: Joy):
        # Deadman button doit être préssé pour que les commandes soient envoyées
        if msg.buttons[self.deadman_button] != 1:
            return

        drive = msg.buttons[self.drive_button]
        reverse = msg.buttons[self.reverse_button]

        if drive == 1 and self.last_drive == 0:
            self.current_gear = 0

        if reverse == 1 and self.last_reverse == 0:
            self.current_gear = 1

        self.last_drive = drive
        self.last_reverse = reverse

        gear_msg = UInt8()
        gear_msg.data = self.current_gear
        self.gear_pub.publish(gear_msg)

        movement_msg = Float64()
        steering_msg = Float64()

        if msg.axes[self.movement_axis] > 0: # Forward
            movement_msg.data = msg.axes[self.movement_axis]
            self.forward_pub.publish(movement_msg)

        elif msg.axes[self.movement_axis] < 0:
            movement_msg.data = abs(msg.axes[self.movement_axis]) * self.max_brake
            self.stop_pub.publish(movement_msg)

        else: # Stopping
            self.forward_pub.publish(Float64(data=0.0))
            self.stop_pub.publish(Float64(data=0.0))
        # Left joystick horizontal adapted from [-1,1] to [-max_steering, max_steering]
        steering_msg.data = -(msg.axes[self.steering_axis] * self.max_steer)
        self.steering_pub.publish(steering_msg)


def main():
    rclpy.init()
    node = TeleopNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
