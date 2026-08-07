from setuptools import find_packages, setup

package_name = "boundingbox_detection"

setup(
    name=package_name,
    version="0.0.1",

    packages=find_packages(),

    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        (
            "share/" + package_name,
            ["package.xml"],
        ),
    ],

    install_requires=["setuptools"],
    zip_safe=True,

    maintainer="semidaturcian",
    maintainer_email="semidaturcian@gmail.com",

    description="YOLO model inference and bounding boxes publisher",
    license="TODO",

    entry_points={
        "console_scripts": [
            "detection_node = boundingbox_detection.detection_node:main",
        ],
    },
)