import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy

from radar_fusion.transformation_utils import TransformationUtils
from radar_fusion.calibration_utils import CalibrationUtils
from bounding_box_msgs.msg import BoundingBox, BoundingBoxList

from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2

from nuscenes.nuscenes import NuScenes
import numpy as np
from cv_bridge import CvBridge
from radar_fusion.fusion_utils import depth_estimation

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
        self.bridge = CvBridge()
        self.calib = CalibrationUtils(self.nusc)
        
        scene = self.nusc.scene[0]
        sample = self.nusc.get("sample", scene["first_sample_token"])
        self.camera_calib = self.calib.get_camera_calibration(sample)
        self.camera_intrinsic = self.calib.get_camera_intrinsic(sample)
        self.radar_calib =  self.calib.get_radar_calibration(sample)

        self.qos_profile = QoSProfile(reliability = QoSReliabilityPolicy.BEST_EFFORT, 
                                      history=QoSHistoryPolicy.KEEP_LAST,
                                       depth = 3)
        self.last_image = None
        self.last_radar = None
        self.last_bbox = None
        self.pub_deb = self.create_publisher(Image, "/projection/debug/img_debug", qos_profile = self.qos_profile)
        self.radar_sub_ = self.create_subscription(PointCloud2, "/ros/radar/points", self.radar_callback, qos_profile = self.qos_profile)
        self.img_sub_ = self.create_subscription(Image, "/ros/camera/images_sender", self.img_callback, qos_profile = self.qos_profile)
        self.bb_sub_ = self.create_subscription(BoundingBoxList, "/processing/bounding_boxes", self.boundingbox_callback, qos_profile = self.qos_profile)
        
    def boundingbox_callback(self, bb_msg):
        self.last_bbox = bb_msg
        self.process_data()

    def radar_callback(self, radar_msg):
        self.last_radar = radar_msg
        self.process_data()

    def img_callback(self, img_msg):
        self.last_image = img_msg
        self.process_data()
    
    def process_data(self):
        if self.last_bbox is None or self.last_radar is None:
            print("EXIT")
            return
        
        self.get_logger().info("RADAR si Camera detection synced.")
        points_raw_data = self.from_pointcloud2(self.last_radar)
        camera_points = TransformationUtils.transform_points(points_raw_data,
                                                                 self.camera_calib,
                                                                 self.radar_calib)
        u,v = TransformationUtils.point_cloud_to_pixel(camera_points, self.camera_intrinsic)

        bb_box_values = self.last_bbox.boxes
        distances = depth_estimation(bbox=bb_box_values, p_u=u, p_v=v, cam_points=camera_points)

        for index, bb in enumerate(bb_box_values):
            print(f"Center_x = {bb.center_x} \n Center_y = {bb.center_y} \n distance {distances[index]}")
                  
        if self.last_image is not None:
            debug_img = self.bridge.imgmsg_to_cv2(
                self.last_image,
                desired_encoding="bgr8"
                )
            all_points_img = Visualization.draw_point_cloud(debug_img, u, v)
            all_points_img2ros = self.bridge.cv2_to_imgmsg(all_points_img, encoding = "bgr8")
            self.pub_deb.publish(all_points_img2ros)


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


