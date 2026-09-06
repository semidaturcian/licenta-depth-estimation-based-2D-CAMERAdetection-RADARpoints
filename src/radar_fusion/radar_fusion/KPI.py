from pathlib import Path

import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from dataclasses import dataclass
from nuscenes.utils.data_classes import Box

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
            box = Box(
                annotation["translation"],
                annotation["size"],
                Quaternion(annotation["rotation"])
            )
            corners_global = box.corners()

            corners_global_h = np.vstack([
                corners_global,
                np.ones((1, corners_global.shape[1]))
            ])

            corners_camera = (
                global_to_camera @ corners_global_h
            )

            X = corners_camera[0, :]
            Y = corners_camera[1, :]
            Z = corners_camera[2, :]

            if np.all(Z <= 0):
                continue

            pixels = K @ corners_camera[:3, :]

            u = pixels[0, :] / pixels[2, :]
            v = pixels[1, :] / pixels[2, :]

            x_min = np.min(u)
            x_max = np.max(u)

            y_min = np.min(v)
            y_max = np.max(v)

            width = x_max - x_min
            height = y_max - y_min

            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2

            global_center = np.array([
                annotation["translation"][0],
                annotation["translation"][1],
                annotation["translation"][2],
                1.0
            ])

            camera_center = (
                global_to_camera @ global_center
            )

            X_center = camera_center[0]
            Y_center = camera_center[1]
            Z_center = camera_center[2]

            distance = np.sqrt(
                X_center**2 +
                Y_center**2 +
                Z_center**2
            )

            gt_box = Bbox_GT(
                center_x=float(center_x),
                center_y=float(center_y),
                width=float(width),
                height=float(height),
                class_id=self.get_class_id(
                    annotation["category_name"]
                ),
                distance=float(distance)
            )

            gt_boxes.append(gt_box)
        return gt_boxes

    def calculate_error(self, bb_yolo, estimated_distances, bb_gt):
        threshold = 0.5
        errors = []
        for yolo_index, bb_y in enumerate(bb_yolo):
            if bb_y.class_id not in [2]:
                continue
            best_iou = 0.0
            best_gt = None
            # min_error = float('inf')
            min_dist = float('inf')
            estimated = estimated_distances[yolo_index]
            if estimated == np.inf:
                continue
            for gt in bb_gt:
                # er = gt.distance - estimated
                # if min_error > abs(er):
                #     ground_truth = gt.distance
                #     min_error = er
                # dist = np.sqrt((gt.center_x - bb_y.center_x)**2 + (gt.center_y - bb_y.center_y)**2)
                # if min_dist > dist:
                #     min_dist = dist
                #     best_gt = gt
                #     print(f"Best GT: {best_gt}")
                
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
            # else:
            #     print("We have a matched GT")

            if best_iou < threshold:
                continue

            ground_truth = best_gt.distance
            error = estimated - ground_truth
            errors.append(error)
            
            # errors.append(min_error)
            # print(
            #     f"class={bb_y.class_id} | "
            #     f"estimated={estimated:.2f} m | "
            #     f"GT={ground_truth:.2f} m | "
            #     f"error={error:.2f} m"
            #     # f"error={min_error:.2f} m"
            # )
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


