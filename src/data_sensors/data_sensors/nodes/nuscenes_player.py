from pathlib import Path

import rclpy
from rclpy.node import Node

from nuscenes.nuscenes import NuScenes

from data_sensors.utils.camera_utils import CameraUtils
from data_sensors.utils.radar_utils import RadarUtils
from data_sensors.utils.calibration_utils import CalibrationUtils
from data_sensors.utils.radar_preprocessor import RadarPreprocessor


class NuScenesPlayer(Node):

    def __init__(self):

        super().__init__("nuscenes_player")

        self.get_logger().info("NuScenes Player started.")

        #
        # Dataset
        #

        self._dataroot = str(Path.home() / "datasets" / "nuscenes")

        self._nusc = NuScenes(
            version="v1.0-mini",
            dataroot=self._dataroot,
            verbose=False
        )

        #
        # Utils
        #

        self.camera = CameraUtils(
            self._nusc,
            self._dataroot
        )

        self.radar = RadarUtils(
            self._nusc,
            self._dataroot
        )

        self.calibration = CalibrationUtils(
            self._nusc
        )

        self.radar_preprocessor = RadarPreprocessor()

        #
        # Scene
        #

        self.scene = self._nusc.scene[0]

        self.sample = self._nusc.get(
            "sample",
            self.scene["first_sample_token"]
        )


def main(args=None):

    rclpy.init(args=args)

    node = NuScenesPlayer()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()