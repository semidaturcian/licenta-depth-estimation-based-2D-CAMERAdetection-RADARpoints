import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
from radar_fusion.transformation_utils import TransformationUtils
from radar_fusion.calibration_utils import CalibrationUtils

from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2

from nuscenes.nuscenes import NuScenes
import numpy as np



class Projection(Node):
    def __init__(self):
        super().__init__('PointCloudProjection')

        DATAROOT = "/home/danitur2/semida/licenta/data/nuscenes"

        self.nusc = NuScenes(
            version="v1.0-mini",
            dataroot=str(DATAROOT),
            verbose=False
        )
        self.calib = CalibrationUtils(self.nusc)
        self.qos_profile = QoSProfile(reliability = QoSReliabilityPolicy.BEST_EFFORT, 
                                      history=QoSHistoryPolicy.KEEP_LAST,
                                       depth = 3)
        self.sub_ = self.create_subscription(PointCloud2, "/ros/radar/points", self.callback, self.qos_profile )

    def callback(self, msg):
        scene = self.nusc.scene[0]
        sample = self.nusc.get("sample", scene["first_sample_token"])

        camera_calib = self.calib.get_camera_calibration(sample)
        radar_calib =  self.calib.get_radar_calibration(sample)

        points_raw_data = self.from_pointcloud2(msg)
        # print(f"Raw point data shape {points_raw_data.shape}")
        # for point_data in points_raw_data:
        #     print(f"Point data shape {point_data.shape}")
        camera_points = TransformationUtils.transform_points(points_raw_data,
                                                                 camera_calib,
                                                                 radar_calib)
        cp = camera_points[:, :3]
        mask = cp[:, 2] > 0

        print(f"Visible points: {np.sum(mask)/len(mask)}")

    def from_pointcloud2(self, msg) -> np.ndarray:
            """
            Convert a PointCloud2 message into a Nx6 numpy array.

            Output format:

                [[x, y, z, vx, vy, rcs],
                [x, y, z, vx, vy, rcs],
                ...]
            """
            point = point_cloud2.read_points(
                        msg,
                        field_names=("x", "y", "z", "vx", "vy", "rcs"),
                        skip_nans=True
                    )
            
            points = np.array([
                                    [p["x"], p["y"], p["z"], p["vx"], p["vy"], p["rcs"]]
                                    for p in point
                                ],
                                dtype=np.float32)
            return points 

def main():
    rclpy.init(args=None)
    node = Projection()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
   try:
       main()
   except:
       print("BYE BYE!")
       rclpy.shutdown()


