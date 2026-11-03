#! coding: utf-8

"""
Écoute les images et la commande de direction, puis les enregistre sur le disque.

Abonnements :
        - `/audibot/left_camera/image_raw/compressed` (sensor_msgs/CompressedImage)
        - `/audibot/front_camera/image_raw/compressed` (sensor_msgs/CompressedImage)
        - `/audibot/right_camera/image_raw/compressed` (sensor_msgs/CompressedImage)
        - `/audibot/steering_cmd` (std_msgs/Float64)

Publications : néant

Paramètres : néant
"""

# Imports externes
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from message_filters import ApproximateTimeSynchronizer, Subscriber
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float64
import numpy as np
import os

class DatasetRecorder(Node):
    """
    Node chargé d’écouter les images des caméras et la commande de direction,
    puis d’enregistrer ces données dans des fichiers .npz.
    Ce node sert à constituer une base d'entraînement pour le réseau neuronal.
    """

    def __init__(self):
        super().__init__("dataset_recorder")

        self.get_logger().info("Enregistreur lancé !")

        # Taille d’un bloc : on sauvegarde toutes les 500 données
        self.chunk_size = 500
        self.chunk_idx = 0

        # Buffers temporaires avant l’écriture sur disque
        self.frames = []
        self.steerings = []

        # Dernière commande de direction reçue (non horodatée)
        self.last_steering = None

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

        # Abonnement à la commande de direction
        self.sub_steering = self.create_subscription(
            Float64,
            "/audibot/steering_cmd",
            self.on_steering,
            qos_profile
        )

        # Synchronisation approximative des trois caméras
        self.sync = ApproximateTimeSynchronizer(
            [self.sub_left, self.sub_front, self.sub_right],
            queue_size=10,
            slop=0.1,  # Tolérance temporelle
        )
        self.sync.registerCallback(self.record_cb)

    def on_steering(self, msg: Float64):
        """ Callback de réception de la commande de direction. """
        self.last_steering = msg.data

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
        # Si aucune commande reçue → on n’enregistre rien
        if self.last_steering is None:
            return

        # Conversion des messages ROS → tableaux numpy
        bridge = CvBridge()
        left_img = bridge.compressed_imgmsg_to_cv2(left_msg, "rgb8")
        front_img = bridge.compressed_imgmsg_to_cv2(front_msg, "rgb8")
        right_img = bridge.compressed_imgmsg_to_cv2(right_msg, "rgb8")

        # Enregistrement dans les buffers
        self.frames.append(
            {
                "left": left_img,
                "front": front_img,
                "right": right_img,
            }
        )
        self.steerings.append(self.last_steering)

        # Si on atteint la taille d’un bloc → sauvegarde sur disque
        if len(self.frames) == self.chunk_size:

            # Répertoire de sauvegarde
            save_dir = os.path.expanduser("~/ros2_ws/dataset")
            os.makedirs(save_dir, exist_ok=True)

            # Chemin du fichier .npz
            chunk_path = os.path.join(
                save_dir,
                f"chunk_{self.chunk_idx}.npz"
            )

            # Sauvegarde compressée (format numpy)
            np.savez_compressed(chunk_path, frames=self.frames, steerings=self.steerings)
            self.get_logger().info(f"Bloc {self.chunk_idx} sauvegardé sur disque.")

            # Réinitialisation des buffers
            self.chunk_idx += 1
            self.steerings = []
            self.frames = []


def main(args=None):
    """ Initialisation et exécution du node. """
    rclpy.init(args=args)

    dataset_recorder = DatasetRecorder()

    try:
        rclpy.spin(dataset_recorder)
    except KeyboardInterrupt:
        pass

    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
