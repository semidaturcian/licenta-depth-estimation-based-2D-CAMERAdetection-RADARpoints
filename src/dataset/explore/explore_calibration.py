from nuscenes.nuscenes import NuScenes

DATAROOT = "/home/danitur2/semida/licenta/data/nuscenes"

nusc = NuScenes(
    version="v1.0-mini",
    dataroot=str(DATAROOT),
    verbose=False
)

scene = nusc.scene[0]
sample = nusc.get("sample", scene["first_sample_token"])

# CAMERA
cam_token = sample["data"]["CAM_FRONT"]
cam_data = nusc.get("sample_data", cam_token)

cam_calib = nusc.get(
    "calibrated_sensor",
    cam_data["calibrated_sensor_token"]
)

print("=" * 60)
print("CAMERA CALIBRATION")
print("=" * 60)

for key, value in cam_calib.items():
    print(f"{key}:")
    print(value)
    print()

# RADAR
radar_token = sample["data"]["RADAR_FRONT"]
radar_data = nusc.get("sample_data", radar_token)

radar_calib = nusc.get(
    "calibrated_sensor",
    radar_data["calibrated_sensor_token"]
)

print("=" * 60)
print("RADAR CALIBRATION")
print("=" * 60)

for key, value in radar_calib.items():
    print(f"{key}:")
    print(value)
    print()

ego_token = sample["data"]["CAM_FRONT"]
sensor_data = nusc.get("sample_data", ego_token)
ego_calib = nusc.get("calibrated_sensor", sensor_data["ego_pose_token"])

print("=" * 60)
print("Ego CALIBRATION")
print("=" * 60)

for key, value in ego_calib.items():
    print(f"{key}:")
    print(value)
    print()