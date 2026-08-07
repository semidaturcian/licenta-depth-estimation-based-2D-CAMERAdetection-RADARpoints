#!/usr/lib/python3
import rclpy
from rclpy.node import Node
from rclpy import time
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
from bounding_box_msgs.msg import BoundingBox, BoundingBoxList
from sensor_msgs.msg import Image
from boundingbox_detection.model_inference import InferenceYoloModel
from cv_bridge import CvBridge
from boundingbox_detection.visualization import Visualization


class DetectionNode(Node):
    def __init__(self):
        super().__init__('detection_node', allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        # self.output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        # self.input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        self.qos_profile = QoSProfile(reliability = QoSReliabilityPolicy.BEST_EFFORT, 
                                      history=QoSHistoryPolicy.KEEP_LAST,
                                       depth = 1)
        self.model_run = InferenceYoloModel()
        self.bridge = CvBridge()

        self.publisher_ = self.create_publisher(BoundingBoxList, "/processing/bounding_boxes", qos_profile=self.qos_profile)
        self.pub_debug = self.create_publisher(Image, "/debug/image_with_bbox", qos_profile = self.qos_profile)
        self.subsription_ = self.create_subscription(Image, "/ros/camera/images_sender", self.callback, qos_profile=self.qos_profile)

    def callback(self, msg:Image):
        img = self.preprocessing(msg) # Transform imaginea din formatul ROS in openCV (cv2)
        bbox = self.model_run.model_inference(img)

        img_d = Visualization.draw_detection(img, bbox)
        img_debug = self.bridge.cv2_to_imgmsg(img_d, encoding = "bgr8")
        self.pub_debug.publish(img_debug)
        message_from_detector = self.extract_data_for_ros(bbox)
        self.publisher_.publish(message_from_detector)

    def preprocessing(self, image):
        img = self.bridge.imgmsg_to_cv2(
            image,
            desired_encoding="bgr8"
             )

        return img

    def extract_data_for_ros(self, boundingboxes):
        ros_boundingbox = BoundingBoxList()
        for bb in boundingboxes:
            bb_ros = BoundingBox()
            bb_ros.center_x = bb.center_x
            bb_ros.center_y = bb.center_y
            bb_ros.height = bb.width
            bb_ros.width =  bb.height
            ros_boundingbox.boxes.append(bb_ros)
        ros_boundingbox.header.stamp = self.get_clock().now().to_msg()
        # ros_boundingbox.header.frame_id = map

        return ros_boundingbox

def main():
    rclpy.init(args=None)
    node = DetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
   try:
       main()
   except:
       print("BYE BYE!")
       rclpy.shutdown()
