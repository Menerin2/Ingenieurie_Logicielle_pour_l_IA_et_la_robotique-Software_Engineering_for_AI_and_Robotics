import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, UInt8

"""
Multiplexer that selects commands from teleoperation or autopilot based on recent activity

Subscriptions:
    - 'throttle' (std_msgs.msg.Float64): commande throttle
    - 'steering' (std_msgs.msg.Float64): commande steering
    - 'brake' (std_msgs.msg.Float64): commande brake
    - 'gear' (std_msgs.msg.UInt8): commande gear

Publications:
    - 'throttle_out' (std_msgs.msg.Float64)
    - 'steering_out' (std_msgs.msg.Float64)
    - 'brake_out' (std_msgs.msg.Float64)
    - 'gear_out' (std_msgs.msg.UInt8)
"""



class CommandMux(Node):
    
    def __init__(self, time_max=500_000_000):
        super().__init__('command_mux')
        
        self.clock = self.get_clock()

        self.time_max = time_max
        self.last_teleop_time = self.clock.now()
        
        # Teleop subscriptions
        self.teleop_throttle_sub = self.create_subscription(Float64, 'teleop_throttle', self.teleop_throttle_cb, 1)
        self.teleop_steering_sub = self.create_subscription(Float64, 'teleop_steering', self.teleop_steering_cb, 1)
        self.teleop_brake_sub = self.create_subscription(Float64, 'teleop_brake', self.teleop_brake_cb, 1)
        self.teleop_gear_sub = self.create_subscription(UInt8, 'teleop_gear', self.teleop_gear_cb, 1)

        # Autopilot subscriptions
        self.autopilot_throttle_sub = self.create_subscription(Float64, 'autopilot_throttle', self.autopilot_throttle_cb, 1)
        self.autopilot_steering_sub = self.create_subscription(Float64, 'autopilot_steering', self.autopilot_steering_cb, 1)
        
        # Publishers
        self.throttle_out_pub = self.create_publisher(Float64, 'throttle_out', 1)
        self.steering_out_pub = self.create_publisher(Float64, 'steering_out', 1)
        self.brake_out_pub = self.create_publisher(Float64, 'brake_out', 1)
        self.gear_out_pub = self.create_publisher(UInt8, 'gear_out', 1)
        
        # Stockage des dernières commandes
        self.last_throttle = Float64(data=0.0)
        self.last_steering = Float64(data=0.0)
        self.last_brake = Float64(data=0.0)
        self.last_gear = UInt8(data=0)
    
    def teleop_throttle_cb(self, msg: Float64):
        self.last_throttle = msg
        self.last_teleop_time = self.clock.now()
        self.publish_commands()
    
    def teleop_steering_cb(self, msg: Float64):
        self.last_steering = msg
        self.last_teleop_time = self.clock.now()
        self.publish_commands()
    
    def teleop_brake_cb(self, msg: Float64):
        self.last_brake = msg
        self.last_teleop_time = self.clock.now()
        self.publish_commands()
    
    def teleop_gear_cb(self, msg: UInt8):
        self.last_gear = msg
        self.last_teleop_time = self.clock.now()
        self.publish_commands()
    
    def autopilot_throttle_cb(self, msg: Float64):
        current_time = self.clock.now()
        # Only update if teleop hasn't been used recently
        if (current_time - self.last_teleop_time).nanoseconds >= self.time_max:
            self.last_throttle = msg
        self.publish_commands()
    
    def autopilot_steering_cb(self, msg: Float64):
        current_time = self.clock.now()
        # Only update if teleop hasn't been used recently
        if (current_time - self.last_teleop_time).nanoseconds >= self.time_max:
            self.last_steering = msg
        self.publish_commands()
    
    def publish_commands(self):
        self.throttle_out_pub.publish(self.last_throttle)
        self.steering_out_pub.publish(self.last_steering)
        self.brake_out_pub.publish(self.last_brake)
        self.gear_out_pub.publish(self.last_gear)



def main():
    rclpy.init()
    node = CommandMux()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()