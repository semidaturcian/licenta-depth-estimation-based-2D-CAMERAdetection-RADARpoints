from __future__ import annotations

import numpy as np


class RadarPreprocessor:
    """
    Radar preprocessing utilities.

    Responsibilities:
        - Remove invalid detections.
        - Filter detections.
        - Normalize radar attributes.
        - Prepare radar data for sensor fusion.
    """

    def __init__(self) -> None:
        """
        Initialize the radar preprocessor.
        """

        pass

    def remove_invalid_points(
        self,
        points: np.ndarray
    ) -> np.ndarray:
        """
        Remove invalid radar detections.

        Currently:
            Placeholder implementation.

        Returns:
            Filtered radar points.
        """

        return points

    def filter_by_distance(
        self,
        points: np.ndarray,
        min_distance: float = 0.0,
        max_distance: float = 100.0
    ) -> np.ndarray:
        """
        Filter radar detections by distance.

        Placeholder implementation.
        """

        return points

    def filter_by_rcs(
        self,
        points: np.ndarray,
        min_rcs: float = -10.0
    ) -> np.ndarray:
        """
        Remove weak radar detections.

        Placeholder implementation.
        """

        return points

    def filter_by_velocity(
        self,
        points: np.ndarray,
        min_velocity: float = 0.0
    ) -> np.ndarray:
        """
        Filter radar detections according to velocity.

        Placeholder implementation.
        """

        return points


    def select_features(
        self,
        points: np.ndarray
    ) -> np.ndarray:
        """
        Select only the radar features required by the
        fusion algorithm.

        Output format:

            x
            y
            z
            vx
            vy
            rcs
        """

        return points
    
    def preprocess(
        self,
        points: np.ndarray
    ) -> np.ndarray:
        """
        Complete radar preprocessing pipeline.
        """

        points = self.remove_invalid_points(points)

        points = self.filter_by_distance(points)

        points = self.filter_by_rcs(points)

        points = self.filter_by_velocity(points)
        # print(f"Radar Points inside radar preprocessing: {points} and radar points type {type(points)}")
        
        return points
