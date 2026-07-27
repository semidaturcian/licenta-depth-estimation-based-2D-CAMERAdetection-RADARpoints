cmake_minimum_required(VERSION 3.10)
project(detection_yolo )

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)

install(DIRECTORY config
        DESTINATION share/$(PROJECT_NAME))

install(DIRECTORY launch
        DESTINATION share/$(PROJECT_NAME))

install(DIRECTORY src
        DESTINATION lib/$(PROJECT_NAME))

install(PROGRAMS src/detection_node.py
        DESTINATION lib/$(PROJECT_NAME))

ament_package()



 