#! coding: utf-8

"""
Send throttle and steering commands predicted by the model to the simulation

Subscriptions:
    - 

Publications:
    - throttle (std_msgs/Float64): The throttle command to send to the car
    - steering (std_msgs/Float64): The steering command to send to the car

Parameters:
"""

import sys

# Imports externes
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from message_filters import ApproximateTimeSynchronizer, Subscriber
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float64
import numpy as np
import onnxruntime
import cv2

class autopilot(Node):

    def __init__(self, onnx_filename):
        super().__init__('autopilot')
        self.throttle_publisher = self.create_publisher(Float64, 'autopilot_throttle', 1)
        self.steering_publisher = self.create_publisher(Float64, 'autopilot_steering', 1)

        # Chargement du modèle Onnx
        print(onnx_filename)
        self.model = onnxruntime.InferenceSession(onnx_filename)

        # Profil QoS minimal pour les images compressées
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Abonnements aux caméras
        self.sub_left = Subscriber(
            self, CompressedImage,
            "/audibot/left_camera/image_raw/compressed",
            qos_profile=qos_profile
        )
        self.sub_front = Subscriber(
            self, CompressedImage,
            "/audibot/front_camera/image_raw/compressed",
            qos_profile=qos_profile
        )
        self.sub_right = Subscriber(
            self, CompressedImage,
            "/audibot/right_camera/image_raw/compressed",
            qos_profile=qos_profile
        )

        # Synchronisation approximative des trois caméras
        self.sync = ApproximateTimeSynchronizer(
            [self.sub_left, self.sub_front, self.sub_right],
            queue_size=10,
            slop=0.1,  # Tolérance temporelle
        )
        self.sync.registerCallback(self.record_cb)

    def record_cb(
        self,
        left_msg: CompressedImage,
        front_msg: CompressedImage,
        right_msg: CompressedImage,
    ):
        """
        Callback appelé lorsque les trois images sont synchronisées.
        On enregistre alors ces images + la dernière commande de direction.
        Lorsque le nombre d'échantillons atteint chunk_size, on sauvegarde le tout.
        """

        # Conversion des messages ROS → tableaux numpy
        bridge = CvBridge()
        left_img = bridge.compressed_imgmsg_to_cv2(left_msg, "rgb8")
        front_img = bridge.compressed_imgmsg_to_cv2(front_msg, "rgb8")
        right_img = bridge.compressed_imgmsg_to_cv2(right_msg, "rgb8")

        # Resize images to 224x224 (model expected size)
        left_img = cv2.resize(left_img, (224, 224))
        front_img = cv2.resize(front_img, (224, 224))
        right_img = cv2.resize(right_img, (224, 224))

        # Normalize to [0, 1] and transpose to (C, H, W)
        left_img = left_img.astype(np.float32) / 255.0
        front_img = front_img.astype(np.float32) / 255.0
        right_img = right_img.astype(np.float32) / 255.0

        left_img = np.transpose(left_img, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        front_img = np.transpose(front_img, (2, 0, 1))
        right_img = np.transpose(right_img, (2, 0, 1))

        # Concatenate the 3 images along channel dimension: (9, 224, 224)
        concatenated = np.concatenate([left_img, front_img, right_img], axis=0)

        # Add batch dimension: (1, 9, 224, 224)
        input_tensor = np.expand_dims(concatenated, axis=0)

        # Run inference
        inputs = {"input": input_tensor}
        output = self.model.run(None, inputs)

        print(output)
        # Extract scalar value from output
        steering_value = float(output[0][0]) if isinstance(output[0], np.ndarray) else float(output[0])
        self.steering_publisher.publish(Float64(data=steering_value))
        self.throttle_publisher.publish(Float64(data=0.1))

def main():
    rclpy.init()
    node = autopilot(sys.argv[1])
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

