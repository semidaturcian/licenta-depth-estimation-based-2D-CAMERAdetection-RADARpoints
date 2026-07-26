from pathlib import Path
from typing import Any

import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes


class CameraUtils:
    """
    Helper class for accessing camera data from the nuScenes dataset.

    Responsibilities:
        - Retrieve camera metadata.
        - Retrieve image paths.
        - Load RGB images.
    """

    def __init__(self, nusc: NuScenes, dataroot: str) -> None:
        """
        Initialize the camera helper.

        Args:
            nusc: Initialized nuScenes object.
            dataroot: Root directory of the nuScenes dataset.
        """

        self._nusc = nusc
        self._dataroot = Path(dataroot)

    def get_camera_data(
        self,
        sample: dict[str, Any],
        camera_name: str = "CAM_FRONT"
    ) -> dict[str, Any]:
        """
        Retrieve the sample_data dictionary corresponding to a camera.

        Args:
            sample: nuScenes sample dictionary.
            camera_name: Camera channel.

        Returns:
            sample_data dictionary.
        """

        camera_token = sample["data"][camera_name]

        return self._nusc.get("sample_data", camera_token)

    def get_image_path(
        self,
        sample: dict[str, Any],
        camera_name: str = "CAM_FRONT"
    ) -> Path:
        """
        Return the absolute path of a camera image.

        Args:
            sample: nuScenes sample dictionary.
            camera_name: Camera channel.

        Returns:
            Absolute image path.
        """

        camera_data = self.get_camera_data(sample, camera_name)

        return self._dataroot / camera_data["filename"]

    def load_image(
        self,
        sample: dict[str, Any],
        camera_name: str = "CAM_FRONT"
    ) -> np.ndarray:
        """
        Load an image using OpenCV.

        Args:
            sample: nuScenes sample dictionary.
            camera_name: Camera channel.

        Returns:
            OpenCV image in BGR format.
        """

        image_path = self.get_image_path(sample, camera_name)

        image = cv2.imread(str(image_path))

        if image is None:
            raise RuntimeError(
                f"Unable to load image: {image_path}"
            )

        return image

    @staticmethod
    def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
        """
        Convert an OpenCV image from BGR to RGB.

        Args:
            image: OpenCV image.

        Returns:
            RGB image.
        """

        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)