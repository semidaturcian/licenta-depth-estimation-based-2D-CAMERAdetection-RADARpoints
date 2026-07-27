from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
import numpy as np


class PointCloudUtils:
    """
    Utility class for converting radar detections to ROS2 PointCloud2 messages.
    """

    @staticmethod
    def create_pointcloud2(
        header: Header,
        points : np.ndarray
    ) -> PointCloud2:
        """
        Convert radar detections into a PointCloud2 message.

        Expected point format:

            [[x, y, z, vx, vy, rcs],
             [x, y, z, vx, vy, rcs],
             ...]
        """

        fields = [

            PointField(
                name="x",
                offset=0,
                datatype=PointField.FLOAT32,
                count=1
            ),

            PointField(
                name="y",
                offset=4,
                datatype=PointField.FLOAT32,
                count=1
            ),

            PointField(
                name="z",
                offset=8,
                datatype=PointField.FLOAT32,
                count=1
            ),

            PointField(
                name="vx",
                offset=12,
                datatype=PointField.FLOAT32,
                count=1
            ),

            PointField(
                name="vy",
                offset=16,
                datatype=PointField.FLOAT32,
                count=1
            ),

            PointField(
                name="rcs",
                offset=20,
                datatype=PointField.FLOAT32,
                count=1
            ),
        ]

        return point_cloud2.create_cloud(
            header,
            fields,
            points
        )