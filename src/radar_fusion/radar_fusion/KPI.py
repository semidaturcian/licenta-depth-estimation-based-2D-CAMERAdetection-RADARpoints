from pathlib import Path

import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from collections import dataclass

@dataclass
class Bbox_GT:
    center_x: float
    center_y: float
    width: float
    height: float
    class_id: int
    distance: float


class KPI:
    def __init__(self,  dataroot):
        self.nusc = NuScenes(
            version="v1.0-mini",
            dataroot=dataroot,
            verbose=False
                )

    def get_gt(self, sample):

        gt_boxes = []
        cam_token = sample["data"]["CAM_FRONT"]
        cam_data = self.nusc.get(
            "sample_data",
            cam_token
        )
        ego_pose = self.nusc.get(
            "ego_pose",
            cam_data["ego_pose_token"]
        )
        camera_calib = self.nusc.get(
            "calibrated_sensor",
            cam_data["calibrated_sensor_token"]
        )
        global_to_ego = transform_matrix(
            ego_pose["translation"],
            Quaternion(ego_pose["rotation"]),
            inverse=True
        )

        ego_to_camera = transform_matrix(
            camera_calib["translation"],
            Quaternion(camera_calib["rotation"]),
            inverse=True
        )
        global_to_camera = ego_to_camera @ global_to_ego
        K = np.array(
            camera_calib["camera_intrinsic"],
            dtype=np.float64
        )

        for ann_token in sample["anns"]:
            annotation = self.nusc.get(
                "sample_annotation",
                ann_token
            )
            global_point = np.array([
                annotation["translation"][0],
                annotation["translation"][1],
                annotation["translation"][2],
                1.0
            ])

            camera_point = global_to_camera @ global_point

            X, Y, Z = camera_point[:3]

            # Obiectul este în spatele camerei
            if Z <= 0:
                continue

            # Proiectare în imagine
            pixel = K @ np.array([X, Y, Z])

            u = pixel[0] / pixel[2]
            v = pixel[1] / pixel[2]

            gt_box = Bbox_GT(
                center_x=u,
                center_y=v,
                width=0.0,
                height=0.0,
                class_id=self.get_class_id(
                    annotation["category_name"]
                ),
                distance=np.sqrt(X**2 + Y**2 + Z**2)
            )
            gt_boxes.append(gt_box)
            return gt_boxes

    def calculate_error(self, bb_yolo, estimated_distances, bb_gt):

        threshold = 0.5
        errors = []
        for yolo_index, bb_y in enumerate(bb_yolo):
            if bb_y.class_id not in [2, 5, 7]:
                continue
            best_iou = 0.0
            best_gt = None
            for gt in bb_gt:
                if gt.class_id != bb_y.class_id:
                    continue
                iou = self.compute_iou(
                    bb_y,
                    gt
                )
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt

            if best_gt is None:
                continue

            if best_iou < threshold:
                continue

            estimated = estimated_distances[yolo_index]
            ground_truth = best_gt.distance
            error = estimated - ground_truth
            errors.append(error)
            print(
                f"class={bb_y.class_id} | "
                f"IoU={best_iou:.2f} | "
                f"estimated={estimated:.2f} m | "
                f"GT={ground_truth:.2f} m | "
                f"error={error:.2f} m"
            )
        return errors

    def get_class_id(self, category):
        mapping = {
            "vehicle.car": 2,
            "vehicle.truck": 7,
            "vehicle.bus.rigid": 5,
            "human.pedestrian.adult": 0,
        }

        return mapping.get(category, -1)

    @staticmethod
    def compute_iou(bb_yolo, bb_gt):
        """
        Compute Intersection over Union between two bounding boxes.

        Both boxes are represented as:
            center_x, center_y, width, height
        """

        # YOLO bbox
        x1_y = bb_yolo.center_x - bb_yolo.width / 2
        y1_y = bb_yolo.center_y - bb_yolo.height / 2
        x2_y = bb_yolo.center_x + bb_yolo.width / 2
        y2_y = bb_yolo.center_y + bb_yolo.height / 2

        # Ground Truth bbox
        x1_g = bb_gt.center_x - bb_gt.width / 2
        y1_g = bb_gt.center_y - bb_gt.height / 2
        x2_g = bb_gt.center_x + bb_gt.width / 2
        y2_g = bb_gt.center_y + bb_gt.height / 2

        # Intersection
        x_left = max(x1_y, x1_g)
        y_top = max(y1_y, y1_g)
        x_right = min(x2_y, x2_g)
        y_bottom = min(y2_y, y2_g)

        if x_right <= x_left or y_bottom <= y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)

        # Areas
        area_yolo = (x2_y - x1_y) * (y2_y - y1_y)
        area_gt = (x2_g - x1_g) * (y2_g - y1_g)

        union = area_yolo + area_gt - intersection

        if union == 0:
            return 0.0

        return intersection / union


