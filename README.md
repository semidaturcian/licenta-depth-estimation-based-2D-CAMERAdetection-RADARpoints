# Depth Estimation based on 2D Camera detection (YOLOv11) and Radar points.

The project aims to build a tool for depth estimation based on the 2D detection from YOLOv11 and the RADAR points.
All the code is running on ROS2 Humble. There are multiple ros2 nodes, for sending the Camera and Radar data from a dataset on a ROS2 topic; 
The inference of the YOLO model is on a different node.
