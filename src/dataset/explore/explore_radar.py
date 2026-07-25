from pathlib import Path

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud

# Calea către dataset
DATAROOT = Path.home() / "datasets" / "nuscenes"

# Încarcă datasetul
nusc = NuScenes(
    version="v1.0-mini",
    dataroot=str(DATAROOT),
    verbose=False
)

# Prima scenă
scene = nusc.scene[0]

# Primul sample
sample = nusc.get("sample", scene["first_sample_token"])

# Token-ul radarului frontal
radar_token = sample["data"]["RADAR_FRONT"]

# Informațiile radarului
radar_data = nusc.get("sample_data", radar_token)

print("=" * 60)
print("RADAR_FRONT information")
print("=" * 60)

for key, value in radar_data.items():
    print(f"{key}: {value}")

# Fișierul radar
radar_path = DATAROOT / radar_data["filename"]

print("\nRadar file:")
print(radar_path)

# Încarcă punctele radar
point_cloud = RadarPointCloud.from_file(str(radar_path))

print("\n==============================================")
print("Radar Point Cloud")
print("==============================================")

print(f"Shape: {point_cloud.points.shape}")
print(f"Number of radar points: {point_cloud.points.shape[1]}")
print(f"Number of attributes: {point_cloud.points.shape[0]}")
print("\nFirst radar point:")
print("=" * 60)

for i, value in enumerate(point_cloud.points[:, 0]):
    print(f"Attribute {i:2d}: {value}")