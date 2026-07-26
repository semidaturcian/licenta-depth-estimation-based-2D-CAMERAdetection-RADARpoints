import rclpy
from rclpy.node import Node
from rclpy import time
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
from bounding_box_msgs import BoundingBox, BoundingBoxList
from sensor_msgs import Image
from model_inference import InferenceYoloModel


class DetectionNode(Node):
    def __init__(self):
        super().__init__('detection_node', allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        self.output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        self.input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        self.qos_profile = QoSProfile(reliability = QoSReliabilityPolicy.BEST_EFFORT, 
                                      history=QoSHistoryPolicy.KEEP_LAST,
                                       depth = 1)
        self.model_run = InferenceYoloModel()
        self.publisher_ = self.create_publisher(BoundingBox, self.output_topic, qos_profile=self.qos_profile)

        self.subsription_ = self.create_subscription(Image, self.input_topic, self.callback, qos_profile=self.qos_profile)

    def callback(self, msg:Image):
        img = self.preprocessing(msg) 
        bbox = self.model_run(img)

        message_from_detector = self.extract_data_for_ros(bbox)
        self.publishers_.publish(message_from_detector)

    def preprocessing(self, image):
        header = image.header
        h = image.height
        w = image.width

        encoding = image.enconding
        step = image.step
        img = image.data

        return img

    def extract_data_for_ros(self, boundingboxes):
        ros_boundingbox = BoundingBoxList()
        for bb in boundingboxes:
            bb_ros = BoundingBox()
            bb_ros.center_x = bb[0]
            bb_ros.center_y = bb[1]
            bb_ros.height = bb[2]
            bb_ros.width =  bb[3]
            ros_boundingbox.boxes.append(bb_ros)

        ros_boundingbox.header.timestamp = time()
        return ros_boundingbox
