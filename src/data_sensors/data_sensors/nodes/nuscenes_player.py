from pathlib import Path
import rclpy
from rclpy.node import Node

from nuscenes.nuscenes import NuScenes
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy

from data_sensors.utils.camera_utils import CameraUtils
from data_sensors.utils.radar_utils import RadarUtils
from data_sensors.utils.radar_preprocessor import RadarPreprocessor

from sensor_msgs.msg import Image, PointCloud2
from data_sensors.utils.pointcloud_utils import PointCloudUtils

from builtin_interfaces.msg import Time
import cv_bridge
from std_msgs.msg import Header

class NuScenesPlayer(Node):

    def __init__(self):
        super().__init__("nuscenes_player")

        self.get_logger().info("NuScenes Player started.")

        # Dataset
        self._dataroot = str(Path.home() / "semida"/"licenta" /"data" / "nuscenes")

        self._nusc = NuScenes(
            version="v1.0-mini",
            dataroot=self._dataroot,
            verbose=False
        )

        # Utils
        self.camera = CameraUtils(
            self._nusc,
            self._dataroot
        )
        self.radar = RadarUtils(
            self._nusc,
            self._dataroot
        )
        self.radar_preprocessor = RadarPreprocessor()
        # Scene
        # print(f" len scenes {self._nusc.scene.__len__()}")
        self.scene = self._nusc.scene[1]
        self.current_sample = self._nusc.get(
            "sample",
            self.scene["first_sample_token"]
        )
        self.cv_to_ros = cv_bridge.CvBridge()
        self.qos_profile = QoSProfile(reliability = QoSReliabilityPolicy.BEST_EFFORT, 
                                      history=QoSHistoryPolicy.KEEP_LAST,
                                       depth = 3)
        self.camera_publisher = self.create_publisher(Image, '/ros/camera/images_sender', qos_profile=self.qos_profile)
        self.radar_publisher = self.create_publisher(PointCloud2, '/ros/radar/points', qos_profile=self.qos_profile)

        _ = self.create_timer(0.17, self.callback)

    def callback(self):
        stamp = self.__nuscenes_timestamp_to_ros(self.current_sample["timestamp"] )
        print(f"Sample timestamp: {stamp}")
        # 1. Camera
        image = self.camera.load_image(self.current_sample)
        # 2. Radar
        radar_points = self.radar.load_point_cloud(self.current_sample)
        radar_points = self.radar.get_pointcloud_points(radar_points)
        # 3. Radar preprocessing
        radar_points = self.radar_preprocessor.preprocess(radar_points)

        ######### DEBUG
        camera_token = self.current_sample["data"]["CAM_FRONT"]
        radar_token = self.current_sample["data"]["RADAR_FRONT"]
        camera_data = self._nusc.get("sample_data",camera_token)
        radar_data = self._nusc.get("sample_data", radar_token)
        camera_stamp = self.__nuscenes_timestamp_to_ros( camera_data["timestamp"])
        radar_stamp = self.__nuscenes_timestamp_to_ros(radar_data["timestamp"])

        ########### END DEBUG
        # 4. ROS conversion
        image_msg = self.cv_to_ros.cv2_to_imgmsg(image, encoding='bgr8')
        image_msg.header.stamp = camera_stamp
        image_msg.header.frame_id = 'map'
        header = Header()
        header.frame_id = 'map'
        header.stamp = radar_stamp
        # print(f"Radar Points: {radar_points} and radar points type {type(radar_points)}")
        pointcloud_msg = PointCloudUtils.create_pointcloud2(header, radar_points)
        pointcloud_msg.header.stamp = radar_stamp
        # 5. Publish
        self.camera_publisher.publish(image_msg)
        self.radar_publisher.publish(pointcloud_msg)
        # 6. Next sample
        self.__advance_sample()

    def __advance_sample(self):
        # if self.current_sample["next"] == "":
        #     self.current_sample = self._nusc.get(
        #         "sample",
        #          self.scene["first_sample_token"]
        #          )
        #     return

        self.current_sample = self._nusc.get(
            "sample",
            self.current_sample["next"]
        )

    def  __nuscenes_timestamp_to_ros(self, timestamp_us):
        stamp = Time()
        stamp.sec = int(timestamp_us // 1_000_000)
        stamp.nanosec = int((timestamp_us % 1_000_000) * 1000)

        return stamp

def main(args=None):
    rclpy.init(args=args)
    node = NuScenesPlayer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()