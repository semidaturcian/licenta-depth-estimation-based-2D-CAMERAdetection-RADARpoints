from typing import Any
from nuscenes.nuscenes import NuScenes


class CalibrationUtils:
    """
    Helper class for accessing sensor calibration data from the nuScenes dataset.

    Responsibilities:
        - Retrieve camera calibration.
        - Retrieve radar calibration.
        - Retrieve ego pose.
    """

    def __init__(self, nusc: NuScenes) -> None:
        """
        Initialize the calibration helper.

        Args:
            nusc: Initialized nuScenes object.
        """

        self._nusc = nusc

    def get_camera_calibration(
        self,
        sample: dict[str, Any],
        camera_name: str = "CAM_FRONT"
    ) -> dict[str, Any]:
        """
        Return the calibration dictionary of a camera.
        """

        camera_token = sample["data"][camera_name]

        camera_data = self._nusc.get("sample_data", camera_token)

        return self._nusc.get(
            "calibrated_sensor",
            camera_data["calibrated_sensor_token"]
        )

    def get_radar_calibration(
        self,
        sample: dict[str, Any],
        radar_name: str = "RADAR_FRONT"
    ) -> dict[str, Any]:
        """
        Return the calibration dictionary of a radar.
        """

        radar_token = sample["data"][radar_name]

        radar_data = self._nusc.get("sample_data", radar_token)

        return self._nusc.get(
            "calibrated_sensor",
            radar_data["calibrated_sensor_token"]
        )

    def get_camera_intrinsic(
        self,
        sample: dict[str, Any],
        camera_name: str = "CAM_FRONT"
    ):
        """
        Return the intrinsic matrix of the selected camera.
        """

        calibration = self.get_camera_calibration(
            sample,
            camera_name
        )

        return calibration["camera_intrinsic"]

    def get_ego_pose(
        self,
        sample: dict[str, Any],
        sensor_name: str = "CAM_FRONT"
    ) -> dict[str, Any]:
        """
        Return the ego pose of the selected sensor.
        """

        sensor_token = sample["data"][sensor_name]

        sensor_data = self._nusc.get(
            "sample_data",
            sensor_token
        )

        return self._nusc.get(
            "ego_pose",
            sensor_data["ego_pose_token"]
        )

