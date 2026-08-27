from pathlib import Path

import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix

# Calea către dataset
DATAROOT = str(Path.home() / "semida"/"licenta" /"data" / "nuscenes")

nusc = NuScenes(
    version="v1.0-mini",
    dataroot=str(DATAROOT),
    verbose=False
)

scene = nusc.scene[0]
sample = nusc.get("sample", scene["first_sample_token"])

annotations = []

for ann_token in sample["anns"]:

    annotation = nusc.get(
        "sample_annotation",
        ann_token
    )

    annotations.append(annotation)

print("=" * 60)
print("GROUND TRUTH ANNOTATIONS")
print("=" * 60)

cam_token = sample["data"]["CAM_FRONT"]

cam_data = nusc.get(
    "sample_data",
    cam_token
)

ego_pose = nusc.get(
    "ego_pose",
    cam_data["ego_pose_token"]
)

print("=" * 60)
print("CAMERA EGO POSE")
print("=" * 60)

print("translation:")
print(ego_pose["translation"])

print("rotation:")
print(ego_pose["rotation"])
print("=" * 60)
print("Transformare obj 2:")

camera_calib = nusc.get(
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
K = np.array(camera_calib["camera_intrinsic"])
global_to_camera = ego_to_camera @ global_to_ego

for index, annotation in enumerate(annotations):

    print(f"\nObject {index}")
    print("-" * 40)

    print(f"token:       {annotation['token']}")
    print(f"category:    {annotation['category_name']}")
    print(f"translation: {annotation['translation']}")
    print(f"size:        {annotation['size']}")
    print(f"rotation:    {annotation['rotation']}")

    global_point = np.array([
    annotation["translation"][0],
    annotation["translation"][1],
    annotation["translation"][2],
    1.0
    ])

    camera_point = global_to_camera @ global_point
    camera_xyz = camera_point[:3]
    print("Camera point: ")
    print(camera_xyz)
    pixel = K @ camera_xyz

    u = pixel[0] / pixel[2]
    v = pixel[1] / pixel[2]

    print("Pixel:", u, v)


