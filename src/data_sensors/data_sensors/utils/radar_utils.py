from pathlib import Path
from typing import Any

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud


class RadarUtils:
    """
    Helper class for accessing radar data from the nuScenes dataset.

    Responsibilities:
        - Retrieve radar metadata.
        - Retrieve radar file paths.
        - Load radar point clouds.
        - Extract useful radar attributes.
        - Prepare radar points for ROS2 PointCloud2 messages.
    """

    def __init__(self, nusc: NuScenes, dataroot: str) -> None:
        """
        Initialize the radar helper.

        Args:
            nusc: Initialized nuScenes object.
            dataroot: Root directory of the nuScenes dataset.
        """

        self._nusc = nusc
        self._dataroot = Path(dataroot)

    def get_radar_data(
        self,
        sample: dict[str, Any],
        radar_name: str = "RADAR_FRONT"
    ) -> dict[str, Any]:
        """
        Retrieve the sample_data dictionary corresponding to a radar.

        Args:
            sample: nuScenes sample dictionary.
            radar_name: Radar channel.

        Returns:
            sample_data dictionary.
        """

        radar_token = sample["data"][radar_name]

        return self._nusc.get("sample_data", radar_token)

    def get_radar_path(
        self,
        sample: dict[str, Any],
        radar_name: str = "RADAR_FRONT"
    ) -> Path:
        """
        Return the absolute path of a radar file.

        Args:
            sample: nuScenes sample dictionary.
            radar_name: Radar channel.

        Returns:
            Absolute radar file path.
        """

        radar_data = self.get_radar_data(sample, radar_name)

        return self._dataroot / radar_data["filename"]

    def load_point_cloud(
        self,
        sample: dict[str, Any],
        radar_name: str = "RADAR_FRONT"
    ) -> RadarPointCloud:
        """
        Load a radar point cloud from the selected radar.

        Args:
            sample: nuScenes sample dictionary.
            radar_name: Radar channel.

        Returns:
            RadarPointCloud object.
        """

        radar_path = self.get_radar_path(sample, radar_name)

        return RadarPointCloud.from_file(str(radar_path))

    def get_xyz(
        self,
        point_cloud: RadarPointCloud
    ) -> np.ndarray:
        """
        Return XYZ coordinates.

        Shape:
            (3, N)
        """

        return point_cloud.points[0:3, :]

    def get_velocity(
        self,
        point_cloud: RadarPointCloud
    ) -> np.ndarray:
        """
        Return ego-motion compensated velocity.

        Shape:
            (2, N)
        """

        return point_cloud.points[8:10, :]

    def get_rcs(
        self,
        point_cloud: RadarPointCloud
    ) -> np.ndarray:
        """
        Return Radar Cross Section (RCS).

        Shape:
            (N,)
        """

        return point_cloud.points[5, :]

    def get_pointcloud_points(
        self,
        point_cloud: RadarPointCloud
    ) -> list[list[float]]:
        """
        Prepare radar detections for a ROS2 PointCloud2 message.

        Output format:

            [
                [x, y, z, vx, vy, rcs],
                ...
            ]
        """

        xyz = self.get_xyz(point_cloud)
        velocity = self.get_velocity(point_cloud)
        rcs = self.get_rcs(point_cloud)

        ros_points = []

        number_of_points = xyz.shape[1]

        for i in range(number_of_points):

            ros_points.append(
                [
                    float(xyz[0, i]),
                    float(xyz[1, i]),
                    float(xyz[2, i]),
                    float(velocity[0, i]),
                    float(velocity[1, i]),
                    float(rcs[i]),
                ]
            )

        return ros_points