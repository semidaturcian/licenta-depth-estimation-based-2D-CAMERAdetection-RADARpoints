import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSHistoryPolicy,
    QoSReliabilityPolicy
)

from radar_fusion.transformation_utils import TransformationUtils
from radar_fusion.calibration_utils import CalibrationUtils
from bounding_box_msgs.msg import BoundingBoxList
from radar_fusion.KPI import KPI

from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2

from nuscenes.nuscenes import NuScenes

import numpy as np
from cv_bridge import CvBridge
import cv2

from radar_fusion.fusion_utils import depth_estimation
import os

from radar_fusion.visualization import Visualization


class Projection(Node):

    def __init__(self):
        super().__init__('PointCloudProjection')
        self.get_logger().info("Projection Node started...")
        DATAROOT = "/home/danitur2/semida/licenta/data/nuscenes"
        self.nusc = NuScenes(
            version="v1.0-mini",
            dataroot=str(DATAROOT),
            verbose=False
        )
        self.kpi = KPI(DATAROOT)

        self.calib = CalibrationUtils(self.nusc)

        # Folosim primul sample doar pentru calibrare.
        # Calibrarea senzorilor nu este dependenta de sample.
        scene = self.nusc.scene[0]

        calibration_sample = self.nusc.get(
            "sample",
            scene["first_sample_token"]
        )

        self.camera_calib = self.calib.get_camera_calibration(
            calibration_sample
        )

        self.camera_intrinsic = self.calib.get_camera_intrinsic(
            calibration_sample
        )

        self.radar_calib = self.calib.get_radar_calibration(
            calibration_sample
        )

        self.camera_timestamp_to_sample = {}
        self.radar_timestamp_to_sample = {}

        for sample in self.nusc.sample:

            camera_token = sample["data"]["CAM_FRONT"]
            radar_token = sample["data"]["RADAR_FRONT"]

            camera_data = self.nusc.get(
                "sample_data",
                camera_token
            )

            radar_data = self.nusc.get(
                "sample_data",
                radar_token
            )

            self.camera_timestamp_to_sample[
                camera_data["timestamp"]
            ] = sample["token"]

            self.radar_timestamp_to_sample[
                radar_data["timestamp"]
            ] = sample["token"]

        self.bridge = CvBridge()
        self.path_output = (
            '/home/danitur2/semida/licenta/data/results'
        )
        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=3
        )

        self.radar_buffer = {}
        self.bbox_buffer = {}

        self.max_buffer_size = 5

        # Imaginea este folosita doar pentru debug
        self.image_buffer = {}

        self.pub_deb = self.create_publisher(
            Image,
            "/projection/debug/img_debug",
            qos_profile=self.qos_profile
        )
        # Subscribers
        self.radar_sub_ = self.create_subscription(
            PointCloud2,
            "/ros/radar/points",
            self.radar_callback,
            qos_profile=self.qos_profile
        )

        self.img_sub_ = self.create_subscription(
            Image,
            "/ros/camera/images_sender",
            self.img_callback,
            qos_profile=self.qos_profile
        )

        self.bb_sub_ = self.create_subscription(
            BoundingBoxList,
            "/processing/bounding_boxes",
            self.boundingbox_callback,
            qos_profile=self.qos_profile
        )
        self.counter = 0

    def boundingbox_callback(self, bb_msg):
        bb_timestamp = self.ros_stamp_to_us(
            bb_msg.header.stamp
        )
        print(f"Numar obiecte detectate in frame: {len(bb_msg.boxes)}")
        sample_token = self.camera_timestamp_to_sample.get(
            bb_timestamp
        )
        if sample_token is None:
            self.get_logger().warn(
                f"Nu exista sample pentru "
                f"camera timestamp {bb_timestamp}"
            )
            return
        radar_msg = self.radar_buffer.get(sample_token)
        if radar_msg is None:
            print(
                f"Radarul pentru sample "
                f"{sample_token} nu a sosit inca."
            )

            # Pastram BBox-ul
            self.bbox_buffer[sample_token] = bb_msg
            self.cleanup_buffer( self.bbox_buffer)
            return
        del self.radar_buffer[sample_token]
        
        self.process_data(
            bb_msg,
            radar_msg,
            sample_token)

    def radar_callback(self, radar_msg):
        radar_timestamp = self.ros_stamp_to_us(
            radar_msg.header.stamp
        )
        sample_token = self.radar_timestamp_to_sample.get(
            radar_timestamp
        )
        if sample_token is None:
            self.get_logger().warn(
                f"Nu exista sample pentru "
                f"radar timestamp {radar_timestamp}"
            )
            return
        self.radar_buffer[sample_token] = radar_msg
        self.cleanup_buffer(
            self.radar_buffer)

        bb_msg = self.bbox_buffer.get(
            sample_token)

        if bb_msg is None:
            return
        del self.radar_buffer[sample_token]

        self.process_data(
            bb_msg,
            radar_msg,
            sample_token)

    def img_callback(self, img_msg):

        timestamp = self.ros_stamp_to_us(
            img_msg.header.stamp)
        self.image_buffer[timestamp] = img_msg
        self.cleanup_buffer(
            self.image_buffer)
        
    def process_data(
        self,
        bb_msg,
        radar_msg,
        sample_token):
        self.get_logger().info("RADAR si Camera detection synced.")
        points_raw_data = self.from_pointcloud2(radar_msg)
        camera_points = (
            TransformationUtils.transform_points(
                points_raw_data,
                self.camera_calib,
                self.radar_calib
            )
        )
        u, v = (
            TransformationUtils.point_cloud_to_pixel(
                camera_points,
                self.camera_intrinsic
            )
        )
        bb_box_values = bb_msg.boxes
        (
            distances_based_mean,
            distances_based_median,
            distances_based_min,
            distances_based_filter
        ) = depth_estimation(
            bbox=bb_box_values,
            p_u=u,
            p_v=v,
            cam_points=camera_points
        )
        # for index, bb in enumerate(bb_box_values):
        #     print(
        #         f"Center_x = {bb.center_x}\n"
        #         f"Center_y = {bb.center_y}\n"
        #         f"distance based centers = "
        #         f"{distances_based_filter[index]}"
        #         f"distance based min = "
        #         f"{distances_based_min[index]}"
        #         f"distance based mean = "
        #         f"{distances_based_mean[index]}"
        #         f"distance based median = "
        #         f"{distances_based_median[index]}"
        #     )
        
        sample = self.nusc.get(
            "sample",
            sample_token
        )

        gt_boxes = self.kpi.get_gt(
            sample
        )
        print(f"Length of gt_boxes: {len(gt_boxes)}")
        errors = self.kpi.calculate_error(
            bb_yolo=bb_box_values,
            estimated_distances=distances_based_median,
            bb_gt=gt_boxes
        )
        print(
            f"Erorile bazate pe estimarea mediana : {errors}"
            if errors is not None
            else "Error is none"
        )
        errors = self.kpi.calculate_error(
            bb_yolo=bb_box_values,
            estimated_distances=distances_based_filter,
            bb_gt=gt_boxes
        )
        print(
            f"Erorile bazate pe estimarea center : : {errors}"
            if errors is not None
            else "Error is none"
        )
        errors = self.kpi.calculate_error(
            bb_yolo=bb_box_values,
            estimated_distances=distances_based_mean,
            bb_gt=gt_boxes
        )
        print(
            f"Erorile bazate pe estimarea mean : {errors}"
            if errors is not None
            else "Error is none"
        )
        errors = self.kpi.calculate_error(
            bb_yolo=bb_box_values,
            estimated_distances=distances_based_min,
            bb_gt=gt_boxes
        )
        print(
            f"Erorile bazate pe estimarea min : {errors}"
            if errors is not None
            else "Error is none"
        )

        image_timestamp = None
        image_timestamp = self.ros_stamp_to_us(
            bb_msg.header.stamp
        )
        image_msg = self.image_buffer.get(image_timestamp)
        if image_msg is None:
            self.get_logger().warn("Nu am gasit imaginea pentru debug.")
            return
        debug_img = self.bridge.imgmsg_to_cv2(image_msg,
            desired_encoding="bgr8"
        )

        all_points_img = (
            Visualization.draw_point_cloud(debug_img, u, v)
        )
        debug_imagine = (
            Visualization.draw_detection(all_points_img, distances_based_filter, bb_box_values, u,v)
        )
        save_path = os.path.join(
            self.path_output,
            str(self.counter) + ".jpg"
        )
        cv2.imwrite(
            save_path,
            debug_imagine
        )
        self.counter += 1
        all_points_img2ros = (
            self.bridge.cv2_to_imgmsg(
                debug_imagine,
                encoding="bgr8"
            )
        )
        all_points_img2ros.header = image_msg.header
        self.pub_deb.publish(
            all_points_img2ros
        )

    def from_pointcloud2(self, msg) -> np.ndarray:
        point = point_cloud2.read_points(
            msg,
            field_names=(
                "x",
                "y",
                "z",
                "vx",
                "vy",
                "rcs"
            ),
            skip_nans=True
        )
        points = np.array(
            [
                [
                    p["x"],
                    p["y"],
                    p["z"],
                    p["vx"],
                    p["vy"],
                    p["rcs"]
                ]
                for p in point
            ],
            dtype=np.float32
        )
        return points

    @staticmethod
    def ros_stamp_to_us(stamp):
        return (
            stamp.sec * 1_000_000
            + stamp.nanosec // 1000
        )

    def cleanup_buffer(self, buffer):
        while len(buffer) > self.max_buffer_size:
            oldest_key = next(iter(buffer))
            del buffer[oldest_key]


def main():
    rclpy.init(args=None)
    node = Projection()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(
            f"Projection stopped: {e}"
        )
        rclpy.shutdown()